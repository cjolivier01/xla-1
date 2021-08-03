#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
XDIR="$CDIR/.."
PTDIR="$XDIR/.."
if [ -z "$PT_INC_DIR" ]; then
  PT_INC_DIR="$PTDIR/build/aten/src/ATen"
fi

if [ -z "$PYTHON_EXE" ]; then
  PYTHON_EXE=python
fi
echo "PYTHON_EXE=$PYTHON_EXE"

echo "PTDIR=$PTDIR"
pushd $PTDIR
$PYTHON_EXE -m tools.codegen.gen_backend_stubs \
  --output_dir="$XDIR/lazy_xla/csrc" \
  --source_yaml="$XDIR/xla_native_functions.yaml"\

popd
