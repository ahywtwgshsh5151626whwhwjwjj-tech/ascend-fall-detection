#!/usr/bin/env bash
# 状态机自测 (不需要相机/NPU)
set -u
export PATH=/usr/local/miniconda3/bin:$PATH
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/selftest_detector.py" "$@"
