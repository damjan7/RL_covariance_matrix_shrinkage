import pickle
import pandas as pd
import numpy as np


pth = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL"

with open(pth + f"\permnos100.pickle", 'rb') as f:
    permnos = pickle.load(f)

with open(pth + f"/future_return_matrices_p100.pickle", 'rb') as f:
    fut_ret_mats = pickle.load(f)


print("donedonedone")