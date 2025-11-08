#!/usr/bin/env bash
# Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
# 运行若干组合以验证与记录压缩比与吞吐。
# 依赖：已训练并生成 artifacts/{tinystories,owt}_vocab.pkl 与 *_merges.pkl
# 用法：bash scripts/tokenizer_experiments.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
PY="uv run python"
EXP="cs336_basics/tokenizer_experiments.py"

cd "$ROOT_DIR"

# (a) 用各自 tokenizer 在各自语料上计算压缩比
$PY $EXP --dataset tinystories --tokenizer tinystories --num_docs 10
$PY $EXP --dataset owt         --tokenizer owt         --num_docs 10

# (b) 交叉编码：用 TinyStories tokenizer 编码 OWT；用 OWT tokenizer 编码 TinyStories
$PY $EXP --dataset owt         --tokenizer tinystories --num_docs 10
$PY $EXP --dataset tinystories --tokenizer owt         --num_docs 10

# (c) 吞吐估计（可按需调整样本数）
$PY $EXP --dataset tinystories --tokenizer tinystories --num_docs 50 --throughput
$PY $EXP --dataset owt         --tokenizer owt         --num_docs 50 --throughput

# (d) 若需进一步生成训练所用的 ID 数组，请在后续单独脚本中完成序列化（本脚本仅做实验与指标输出）。

echo "完成 tokenizer 实验脚本运行。"
