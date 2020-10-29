from __future__ import division
from __future__ import print_function

import time
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
import ptwse.scope as scope


def flatten_xla_tensors(data):
  tensors = []

  def select_fn(value):
    return type(value) == torch.Tensor and xm.is_xla_tensor(value)

  def collect_fn(value):
    tensors.append(value)

  xu.for_each_instance(data, select_fn, collect_fn)
  return tensors


FORCE_RUNONWSE = True


def run(
    loader,
    device,
    batch_size,
    minibatch_size,
    megabatch_multiplier,
    steps_per_epoch,
    closure,
    closure_args=(),
    output_closure=None,
    output_closure_args=(),
    pre_closure=None,
    **kwargs,
):
  para_loader = pl.ParallelLoader(
      loader, [device],
      fixed_batch_size=True,
      minibatch_size=minibatch_size,
      **kwargs)

  # Disabling the mark step is necessary because otherwise it
  # will try to sync the outputs that we pruned
  device_loader = para_loader.per_device_loader(
      device,
      enable_minibatch=minibatch_size != 0,
  )

  prev_hash = None
  handle_map = dict()
  steady_graph = None
  outputs = None
  tensors = None
  minibatch = None

  step = 0
  batch_count = 0

  has_been_steady = False
  did_execute_on_proxy = False

  # TODO: if not steady_graph, can next() return us a batch-sized view of the minibatch?
  for batch in device_loader:
    step += 1
    print(f'step={step}')

    if minibatch_size > batch_size:
      minibatch = device_loader.next_mini_batch_item()

    def send_batch(batch, steady_graph):
      assert torch_xla._XLAC._xla_was_previous_mark_step_on_proxy()
      if batch:
        print('Sending batch...')
        outputs = torch_xla._XLAC._xla_execute_compiled_graph(
          flatten_xla_tensors(batch), steady_graph)
        # Could actually output closure here
        print('Batch sent')

    did_execute_on_proxy = torch_xla._XLAC._xla_was_previous_mark_step_on_proxy()
    if did_execute_on_proxy:
      print(f'----------------------------------------')
      print(f'{batch_count}')
      print(f'----------------------------------------')
      has_been_steady = True
      if output_closure is not None and outputs is not None:
        output_closure(outputs, *output_closure_args)
      assert torch_xla._XLAC._xla_was_previous_mark_step_on_proxy()
      if pre_closure:
        # Set outputs
        pre_closure(tensors)
      outputs = torch_xla._XLAC._xla_execute_compiled_graph(
        flatten_xla_tensors(batch), steady_graph)
      assert torch_xla._XLAC._xla_was_previous_mark_step_on_proxy()
      batch_count += 1
      if minibatch is not None:
        send_batch(minibatch, steady_graph)
        batch_count += minibatch_size
      handle_map.clear()

      #tensors = closure(batch, *closure_args)

    else: # or not torch_xla._XLAC._xla_was_previous_mark_step_on_proxy():
      print('unsteady graph fork')
      if has_been_steady:
        print('what went wrong?')
        has_been_steady = False
      if did_execute_on_proxy:
        print('WHAT WENT WRONG???')
        has_been_steady = False
      tensors = closure(batch, *closure_args)
      if pre_closure:
        # Set outputs
        pre_closure(tensors)
      flattened_tensors = flatten_xla_tensors(batch)
      graph_dict = torch_xla._XLAC._xla_compile_execute_graph(
        flattened_tensors, tensors, str(device), [], handle_map)
      batch_count += 1
      if graph_dict is None:
        raise RuntimeError('Unable to accelerate graph execution')
      chash = graph_dict['hash']
      if torch_xla._XLAC._xla_was_previous_mark_step_on_proxy():
        did_execute_on_proxy = True
        xm.master_print("WSE STEADY GRAPH")
        steady_graph = graph_dict['graph']
        #handle_map = None
        #handle_map.clear()
        if minibatch is not None:
          send_batch(minibatch, steady_graph)
          batch_count += minibatch_size
      else:
        xm.master_print("UNSTEADY GRAPH")
        prev_hash = chash
        handle_map = graph_dict['handle_map']
      outputs = graph_dict['outputs']

      # Release the compile graph dictionary to make sure we do not hold two
      # copies of it while reaching stable compilations.
      graph_dict = None

    if pre_closure and torch_xla._XLAC._xla_was_previous_mark_step_on_proxy():
        # Set outputs
        pre_closure(list(outputs))

    xm.mark_step_trail()
    if step >= steps_per_epoch:
      break

  if tensors is not None and pre_closure:
    # Set outputs
    pre_closure(tensors)

  return step


def run_almost_original(loader,
                 device,
                batch_size,
                minibatch_size,
                megabatch_multiplier,
                 closure,
                 closure_args=(),
                 output_closure=None,
                 output_closure_args=(),
                 **kwargs):
  para_loader = pl.ParallelLoader(loader, [device], fixed_batch_size=True)
  device_loader = para_loader.per_device_loader(device)
  prev_hash = None
  handle_map = dict()
  steady_graph = None
  outputs = None
  for batch in device_loader:
    if output_closure is not None and outputs is not None:
      output_closure(outputs, *output_closure_args)
    if steady_graph:
      outputs = torch_xla._XLAC._xla_execute_compiled_graph(
        flatten_xla_tensors(batch), steady_graph)
    else:
      tensors = closure(batch, *closure_args)
      graph_dict = torch_xla._XLAC._xla_compile_execute_graph(
        flatten_xla_tensors(batch), tensors, str(device), [], handle_map)
      if graph_dict is None:
        raise RuntimeError('Unable to accelerate graph execution')
      chash = graph_dict['hash']
      if chash == prev_hash:
        print("STEADY GRAPH")
        steady_graph = graph_dict['graph']
        handle_map = None
      else:
        print("UNSTEADY GRAPH")
        prev_hash = chash
        handle_map = graph_dict['handle_map']
      outputs = graph_dict['outputs']
      # Release the compile graph dictionary to make sure we do not hold two
      # copies of it while reaching stable compilations.
      graph_dict = None
    xm.mark_step_trail()


def run_original(loader,
                 device,
                 closure,
                 closure_args=(),
                 output_closure=None,
                 output_closure_args=()):
  para_loader = pl.ParallelLoader(loader, [device], fixed_batch_size=True)
  device_loader = para_loader.per_device_loader(device)
  prev_hash = None
  handle_map = dict()
  steady_graph = None
  outputs = None
  for batch in device_loader:
    if output_closure is not None and outputs is not None:
      output_closure(outputs, *output_closure_args)
    if steady_graph:
      outputs = torch_xla._XLAC._xla_execute_compiled_graph(
          flatten_xla_tensors(batch), steady_graph)
    else:
      tensors = closure(batch, *closure_args)
      graph_dict = torch_xla._XLAC._xla_compile_execute_graph(
          flatten_xla_tensors(batch), tensors, str(device), [], handle_map)
      if graph_dict is None:
        raise RuntimeError('Unable to accelerate graph execution')
      chash = graph_dict['hash']
      if chash == prev_hash:
        print("STEADY GRAPH")
        steady_graph = graph_dict['graph']
        handle_map = None
      else:
        print("UNSTEADY GRAPH")
        prev_hash = chash
        handle_map = graph_dict['handle_map']
      outputs = graph_dict['outputs']
      # Release the compile graph dictionary to make sure we do not hold two
      # copies of it while reaching stable compilations.
      graph_dict = None
    xm.mark_step_trail()
