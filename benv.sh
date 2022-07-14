#!/bin/bash
DEBUG=1 \
  BUILD_CPP_TESTS=0 \
  PATH=/home/${USER}/.nvm/versions/node/v18.4.0/bin:/home/${USER}/bin:/usr/local/bin:/home/${USER}/bin:/home/${USER}/scripts:/home/${USER}/.local/bin:/home/${USER}/bin:/home/${USER}/.vscode-server/bin/b06ae3b2d2dbfe28bca3134cc6be65935cdfea6a/bin/remote-cli:/home/${USER}/bin:/home/${USER}/scripts:/home/${USER}/.local/bin:/home/${USER}/bin:/home/${USER}/.conda/envs/dojo/bin:/data/dojo/sw/tools/conda/conda49-py38/condabin:/home/${USER}/bin:/home/${USER}/scripts:/home/${USER}/.local/bin:/home/${USER}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/go/bin:/usr/local/go/bin:/usr/local/go/bin \
  $@
