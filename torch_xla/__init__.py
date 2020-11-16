import os
import re


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


def _set_missing_env(name, value):
  if name not in os.environ:
    os.environ[name] = value


def _setup_default_env():
  _set_missing_env('TF_CPP_MIN_LOG_LEVEL', '1')


# These needs to be called before the _XLAC module is loaded.
_setup_default_env()
_setup_grpc()
_setup_xla_flags()

import sys
import ctypes
import atexit
import torch
from ._patched_functions import _apply_patches
from .version import __version__

_use_rtld_global = hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags')

if _use_rtld_global:
  _default_dlopen_flags = sys.getdlopenflags()
  RTLD_DEEPBIND = 0x8
  RTLD_NOW = 2
  sys.setdlopenflags(_default_dlopen_flags | RTLD_DEEPBIND)

import _XLAC

if _use_rtld_global:
  sys.setdlopenflags(_default_dlopen_flags)


def _prepare_to_exit():
  _XLAC._prepare_to_exit()


_XLAC._initialize_aten_bindings()
atexit.register(_prepare_to_exit)
_apply_patches()
