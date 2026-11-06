import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT)]
torch.set_num_threads(2)
