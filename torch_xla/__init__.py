import os
import re

# if not int(os.environ.get("PTXLA_STANDALONE", 0)):
#   # Load TensorFlow first in order to shared backends
#   import tensorflow

def _setup_grpc():
  # Setup GRPC options to correctly talk to TPU backends.
  options = [
      'grpc.keepalive_time_ms=60000',  # 1 min
      'grpc.keepalive_timeout_ms=14400000',  # 4 hrs
      'grpc.http2.max_pings_without_data=0',  # unlimited
      'grpc.http2.min_ping_interval_without_data_ms=300000',  # 5 min
  ]
  os.environ['TF_GRPC_DEFAULT_OPTIONS'] = ','.join(options)


def _set_missing_flags(flags, sets):
  for name, defval in sets:
    insert = True
    for fval in flags:
      m = re.match(r'(--)?([^=]+)', fval)
      if m and m.group(2) == name:
        insert = False
        break
    if insert:
      flags.append('--{}={}'.format(name, defval))
  return flags


def _setup_xla_flags():
  flags = os.environ.get('XLA_FLAGS', '').split(' ')
  flags = _set_missing_flags(flags, (('xla_cpu_enable_fast_math', 'false'),))
  os.environ['XLA_FLAGS'] = ' '.join(flags)


# These needs to be called before the _XLAC module is loaded.
_setup_grpc()
_setup_xla_flags()

import torch
from .version import __version__
import _XLAC

_XLAC._initialize_aten_bindings()
