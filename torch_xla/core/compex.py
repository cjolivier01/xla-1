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
):
  para_loader = pl.ParallelLoader(
      loader, [device],
      fixed_batch_size=True,
      loader_prefetch_size=100,
      device_prefetch_size=100)
  device_loader = para_loader.per_device_loader(
      device, disable_mark_step_after_first=True)
  prev_hash = None
  handle_map = dict()
  steady_graph = None
  outputs = None

  start_time = None
  last_step_timed = 0
  step = 0
  wse_comp = False
  for batch in device_loader:
    step += 1
    #if step == 1 or step == 2 or step == 3 or step == 4 or step == 5:
    if not steady_graph:
      tensors = closure(batch, *closure_args)
      if pre_closure:
        # Set outputs
        pre_closure(tensors)
      graph_dict = torch_xla._XLAC._xla_compile_execute_graph(
          flatten_xla_tensors(batch), tensors, str(device), [], handle_map)
      if graph_dict is None:
        raise RuntimeError('Unable to accelerate graph execution')
      chash = graph_dict['hash']
      if chash == prev_hash and step != 3:
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
      if pre_closure:
        # Set outputs
        pre_closure(tensors)

    else:
      assert steady_graph
      if output_closure is not None and outputs is not None:
        output_closure(outputs, *output_closure_args)
      outputs = torch_xla._XLAC._xla_execute_compiled_graph(
          flatten_xla_tensors(batch), steady_graph)
      # if step > 0 and step % log_steps == 0:
      #   now_time = time.time()
      #   if start_time:
      #     per_step_time = (now_time - start_time) / (step - last_step_timed)
      #     steps_per_second = 1 / per_step_time
      #     print(
      #         f'Round-trip step time: {per_step_time} seconds, steps per second: {steps_per_second}'
      #     )
      #   print(f'BEGIN Train step {step}')
      #   start_time = time.time()
      #   last_step_timed = step

    xm.mark_step_trail()
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
