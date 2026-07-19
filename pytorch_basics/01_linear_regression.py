import numpy as np
from pathlib import Path
from torch import nn

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "pokemon.csv"

loaded_data = np.loadtxt(datapath)

x_train_np = loaded_data[:, 0:-1]
y_train_np = loaded_data[:, [-1]]
