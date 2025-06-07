#!/bin/bash
set -ex
cmake --build . --target install -j$(nproc)
