#!/bin/bash
# Marathon wrapper - ensures correct environment
cd /home/chillizu/Projects/Folunar_ || exit 1
source .venv/bin/activate || exit 1
export PYTHONPATH="/home/chillizu/Projects/Folunar_"
export TORCH_NUM_THREADS=4
export OMP_NUM_THREADS=4
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
# NOTE: set DEEPSEEK_API_KEY in your environment before running this script
export PYTHONUNBUFFERED=1
exec python3 -u scripts/marathon_8h.py
