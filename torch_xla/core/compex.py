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
      **kwargs)

  # Disabling the mark step is necessary because otherwise it
  # will try to sync the outputs that we pruned
  device_loader = para_loader.per_device_loader(
      device,
      disable_mark_step_after_first=True,
  )

  prev_hash = None
  handle_map = dict()
  steady_graph = None
  outputs = None
  tensors = None
  #megabatch = None

  megabatch_size = kwargs.get('megabatch_size', 0)

  step = 0
  batch_count = 0

  # TODO: if not steady_graph, can next() return us a batch-sized view of the megabatch?
  for batch in device_loader:
    step += 1
    print(f'step={step}')

    if megabatch_size:
      megabatch = device_loader.next_mini_batch_item()

    if torch_xla._XLAC._xla_was_previous_mark_step_on_proxy():
      print(f'----------------------------------------')
      print(f'{batch_count}')
      print(f'----------------------------------------')
      if output_closure is not None and outputs is not None:
        output_closure(outputs, *output_closure_args)
      assert torch_xla._XLAC._xla_was_previous_mark_step_on_proxy()
      if pre_closure:
        # Set outputs
        pre_closure(tensors)
      outputs = torch_xla._XLAC._xla_execute_compiled_graph(
        flatten_xla_tensors(batch), steady_graph)
      batch_count += 1
      if megabatch_size:
        assert False  # not running atm
        print('Sending megabatch')
        outputs = torch_xla._XLAC._xla_execute_compiled_graph(
          flatten_xla_tensors(megabatch), steady_graph)
        # Could actually output closure here
        print('Megabatch sent')
        batch_count += megabatch_size
      # if pre_closure:
      #   # Set outputs
      #   pre_closure(tensors)
    else: # or not torch_xla._XLAC._xla_was_previous_mark_step_on_proxy():
      print('unsteady graph fork')
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
        xm.master_print("WSE STEADY GRAPH")
        steady_graph = graph_dict['graph']
        handle_map = None
      # elif chash == prev_hash and not torch_xla._XLAC._xla_was_previous_mark_step_on_proxy():
      #   assert False
      #   xm.master_print("STEADY GRAPH")
      #   steady_graph = graph_dict['graph']
      #   handle_map = None
      else:
        xm.master_print("UNSTEADY GRAPH")
        prev_hash = chash
        handle_map = graph_dict['handle_map']
      outputs = graph_dict['outputs']

      # Release the compile graph dictionary to make sure we do not hold two
      # copies of it while reaching stable compilations.
      graph_dict = None

    # else:
    #   print('steady graph fork')
    #   assert steady_graph
    #   if output_closure is not None and outputs is not None:
    #     output_closure(outputs, *output_closure_args)
    #   assert torch_xla._XLAC._xla_was_previous_mark_step_on_proxy()
    #   outputs = torch_xla._XLAC._xla_execute_compiled_graph(
    #       flatten_xla_tensors(batch), steady_graph)
    #   batch_count += 1
    #   if megabatch_size:
    #     if torch_xla._XLAC._xla_was_previous_mark_step_on_proxy():
    #       print('Last step was on proxy, so executing minibatch')
    #       #megabatch = device_loader.next_mini_batch_item()
    #       print('Sending megabatch')
    #       torch_xla._XLAC._xla_execute_compiled_graph(
    #         flatten_xla_tensors(megabatch), steady_graph)
    #       print('Megabatch sent')
    #       batch_count += megabatch_size
    #     else:
    #       print('Last step was on proxy, no minibatch')
    #
    #   if pre_closure:
    #     # Set outputs
    #     pre_closure(tensors)

    xm.mark_step_trail()

  if tensors is not None and pre_closure:
    # Set outputs
    pre_closure(tensors)

  return step


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
