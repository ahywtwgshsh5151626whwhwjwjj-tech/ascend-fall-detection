#!/usr/bin/env bash
# 批量评测: 新旧判据对比
set -u
export PATH=/usr/local/miniconda3/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/eval_dataset.py" "$@"
