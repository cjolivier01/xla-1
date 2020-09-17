from __future__ import division
from __future__ import print_function

from six import iteritems, itervalues
import threading
import torch
import torch_xla
import torch_xla.utils.keyd_queue as kq
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm


class PerDeviceQueue(object):

  def __init__(self, device, loader_prefetch_size, device_prefetch_size):
    self.device = device
    self.loader_queue = kq.Queue(maxsize=loader_prefetch_size)
    self.queue = kq.Queue(maxsize=device_prefetch_size)


class PerDeviceLoader(object):

  def __init__(self, loader, device, disable_mark_step_after_first=None):
    self._loader = loader
    self._device = device
    self._disable_mark_step_after_first = disable_mark_step_after_first

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def __len__(self):
    return self._loader.per_device_samples()

  def next(self):
    if self._disable_mark_step_after_first is None:
      xm.mark_step()
    elif self._disable_mark_step_after_first:
      xm.mark_step()
      self._disable_mark_step_after_first = 0

    item = self._loader.next_item(self._device)
    if item is None:
      raise StopIteration
    return item

  def next_megabatch_item(self):
    return self._loader.next_megabatch_item(self._device)


class ParallelLoader(object):
  """Wraps an existing PyTorch DataLoader with background data upload.

  Args:
    loader (:class:`torch.utils.data.DataLoader`): The PyTorch DataLoader to be
      wrapped.
    devices (`torch.device`...): The list of devices where the data has to be
      sent. The i-th sample returned by the `loader` will be sent to `devices[i
      % len(devices)]`.
    batchdim (int, optional): The dimension which is holding the batch size.
      Default: 0
    fixed_batch_size (bool, optional): Ensures that all the batch sizes sent to
      the devices are of the same size. The original `loader` iteration stops as
      soon as a not matching batch size is found.
      Default: False
    loader_prefetch_size (int, optional): The max capacity of the queue used by
      the thread which is reading samples from the `loader`, to be processed by
      the worker threads which upload data to the devices.
      Default: 8
    device_prefetch_size (int, optional): The max size of the per-device queues,
      where the worker threads deposit tensors which have already been sent to
      devices.
      Default: 4
  """

  def __init__(self,
               loader,
               devices,
               batchdim=0,
               fixed_batch_size=False,
               loader_prefetch_size=8,
               device_prefetch_size=4,
               megabatch_size=0):
    self._loader = loader
    self._devices = [torch.device(x) for x in devices]
    self._batchdim = batchdim
    self._fixed_batch_size = fixed_batch_size
    self._per_device_samples = len(loader) // len(devices)
    self._done = False
    self._queues = dict()
    self._mgb_queues = dict()
    self._mgb_size = megabatch_size
    self._batch_dim = 0
    for device in self._devices:
      self._queues[device] = PerDeviceQueue(device, loader_prefetch_size,
                                            device_prefetch_size)
      if self._mgb_size > 1:
        self._mgb_queues[device] = PerDeviceQueue(device, loader_prefetch_size,
                                                  device_prefetch_size)
      else:
        self._mgb_queues[device] = None
    thread = threading.Thread(target=self._loader_worker)
    thread.daemon = True
    thread.start()
    for dqueue, mgbqueue in zip(
        itervalues(self._queues), itervalues(self._mgb_queues)):
      thread = threading.Thread(
          target=self._worker, args=(
              dqueue,
              mgbqueue,
          ))
      thread.daemon = True
      thread.start()

  def per_device_loader(self, device, **kwargs):
    """Retrieves the loader iterator object for the given device.

    Args:
      device (`torch.device`): The device whole loader is being requested.

    Returns:
      The loader iterator object for the `device`. This is not a
      `torch.utils.data.DataLoader` interface, but a Python iterator which
      returns the same tensor data structure as returned by the wrapped
      `torch.utils.data.DataLoader`, but residing on XLA devices.
    """
    return PerDeviceLoader(self, torch.device(device), **kwargs)

  def per_device_samples(self):
    return self._per_device_samples

  def next_item(self, device):
    dqueue = self._queues[device]
    return dqueue.queue.get()

  def next_megabatch_item(self, device):
    dqueue = self._mgb_queues[device]
    if dqueue is not None:
      return dqueue.queue.get()
    return None

  def close(self):
    self._done = True
    for dqueue in itervalues(self._queues):
      dqueue.queue.close()
      dqueue.loader_queue.close()

  def _get_batch_size(self, data, dim):
    size = []

    def fn(v):
      csize = v.size()[dim]
      if not size:
        size.append(csize)
      else:
        assert csize == size[0]

    xu.for_each_instance(data, lambda x: type(x) == torch.Tensor, fn)
    return size[0] if size else None

  def _loader_worker(self):
    queues = list(self._queues.values())
    mgb_queues = list(self._mgb_queues.values())
    data_iter = enumerate(self._loader)
    batch_size = None
    batch = []
    mgb_batch = []
    field_towers = list()
    has_mgb = self._mgb_size > 1
    mgb_data = None
    tmp_mgb_batch = []
    while not self._done:
      try:
        _, data = next(data_iter)

        if has_mgb:
          for i in range(self._mgb_size):
            _, more_data = next(data_iter)
            # make towers of each field, self._mgb_size deep
            for input_index, field in enumerate(more_data):
              if input_index == len(field_towers):
                field_towers.append([field])
              else:
                assert input_index < len(field_towers)
                field_towers[input_index].append(field)
          # now make them fat tensors
          for tensor_list in field_towers:
            fat_tensor = torch.cat(tensor_list, dim=self._batch_dim)
            tmp_mgb_batch.append(fat_tensor)
          mgb_data = tuple(tmp_mgb_batch)
          tmp_mgb_batch.clear()
          field_towers.clear()

      except StopIteration:
        break
      if self._fixed_batch_size:
        if batch_size is None:
          batch_size = self._get_batch_size(data, self._batchdim)
        elif batch_size != self._get_batch_size(data, self._batchdim):
          break
      batch.append(data)
      if mgb_data:
        mgb_batch.append(mgb_data)
      if len(batch) == len(self._devices):
        for queue_no, device_batch in enumerate(batch):
          queues[queue_no].loader_queue.put(device_batch)
          if mgb_queues[queue_no] is not None:
            mgb_queues[queue_no].loader_queue.put(mgb_batch[queue_no])
        batch = []
        mgb_batch = []
    for dqueue in queues:
      dqueue.loader_queue.close_write()
    for dqueue in mgb_queues:
      if dqueue is not None:
        dqueue.loader_queue.close_write()

  def _get_batch(self, dqueue):
    batch = []
    while dqueue.queue.max_size() > len(batch):
      item = dqueue.loader_queue.get()
      if item is None:
        break
      batch.append(item)
    return batch

  def _worker(self, dqueue, mgbqueue):
    device = torch.device(dqueue.device)
    while True:
      batch = self._get_batch(dqueue)
      mgb_batch = self._get_batch(mgbqueue) if mgbqueue else None
      if not batch:
        break
      batch = xm.send_cpu_data_to_device(batch, device)
      if mgb_batch:
        mgb_batch = xm.send_cpu_data_to_device(mgb_batch, device)
      if not mgb_batch:
        for data in batch:
          dqueue.queue.put(data)
      else:
        for data, mgb_data in zip(batch, mgb_batch):
          dqueue.queue.put(data)
          mgbqueue.queue.put(mgb_data)

    dqueue.queue.close_write()
    if mgbqueue:
      mgbqueue.queue.close_write()


class MpDeviceLoader(object):
  """Wraps an existing PyTorch DataLoader with background data upload.

  This class should only be using with multi-processing data parallelism.

  Args:
    loader (:class:`torch.utils.data.DataLoader`): The PyTorch DataLoader to be
      wrapped.
    device (`torch.device`...): The device where the data has to be sent.
    kwargs: Named arguments for the `ParallelLoader` constructor.
  """

  def __init__(self, loader, device, **kwargs):
    self._loader = loader
    self._device = device
    self._parallel_loader_kwargs = kwargs

  def __iter__(self):
    parallel_loader = ParallelLoader(self._loader, [self._device],
                                     **self._parallel_loader_kwargs)
    return parallel_loader.per_device_loader(self._device)

  def __len__(self):
    return len(self._loader)
