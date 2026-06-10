#!/bin/bash
cd /home/chillizu/Projects/Folunar_
source .venv/bin/activate
export OMP_NUM_THREADS=4
exec nice -n 19 python3 -u scripts/train_continual.py --cycles 1500 --output checkpoints/continual-v2
