import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
import pickle
import plotly.express as px



import helper_functions as hf

file_folder = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets"

df_full = pd.DataFrame(columns=['model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6',
                                'model_7', 'model_8', 'model_9', 'model_10', 'model_11'], dtype=float)

# model 6 is the "opt" theoretical shrinkage model
for i, file in enumerate(os.listdir(file_folder)):
    if file.startswith("factor"):
        pth = file_folder + "/" + file
        with open(pth, 'rb') as f:
            df = pickle.load(f)
            df.columns = df.iloc[0, :]
            df = df.drop(0)
    df_full[f"model_{i+1}"] = df['pf_std'].astype(float)


#
cols = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_7', 'model_8', 'model_9', 'model_10', 'model_11']
df_new = pd.DataFrame(columns=['x', 'y', 'model'])
hist_volas = np.mean(df["hist_vola"].tolist(), axis=1)
for i in range(len(df_full.columns)):
    if (i+1) != 6:
        df_full[f"model_{i+1}"] = df_full[f"model_{i+1}"] - df_full["model_6"]

model_nums = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
df_new = pd.concat([(pd.DataFrame([df_full[col].values, hist_volas, np.full((2499,), model_nums[i])])).T for i, col in enumerate(cols)])
df_new.columns = ['model_std', 'hist_vola', 'model']

fig = px.scatter(df_new, x = "hist_vola", y = "model_std", color="model")
# fig.show()

# count of lowest portfolio std
from collections import Counter
counts = Counter(df_full.idxmin(axis=1).values)


"""
do the same testing but for the fixed shrinkages
"""

path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets\fixed_shrkges_p100.pickle"
with open(path, 'rb') as f:
    df_fixed = pickle.load(f)
    df_fixed.columns = df_fixed.iloc[0, :]
    df_fixed = df_fixed.drop(0)
    for col in range(1, df_fixed.shape[1]):
        df_fixed.iloc[:, col] = df_fixed.iloc[:, col].astype(float) - df_full["model_6"]  # subtract the actual shrk. intensity
    df_fixed.iloc[:, 0] = np.mean(df_fixed["hist_vola"].tolist(), axis=1)

# basic analysis
mins = df_fixed.iloc[:, 1:].idxmin(axis=1).values
min_counts = Counter(mins)

# plot historical volas
n, bins, patches = plt.hist(x=df_fixed.iloc[:, 0].values, bins=10)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

volas_mean = hist_volas.mean()
volas_std = hist_volas.std()

# let's also see in cases where a higher shrkg intensity was better, if we also observed higher volas
# let's look at cases, i.e., where cur_vola > volas_mean + volas_std * 0.5
is_vola_larger = df_fixed.iloc[:, 0] > volas_mean + 0.5 * volas_std  # boolean
df_vola_larger = df_fixed[is_vola_larger]
mins = df_vola_larger.iloc[:, 1:].idxmin(axis=1).values
min_counts_vola_larger = Counter(mins)


print("ok, cool")


"""
# create a figure and axis object
fig, ax = plt.subplots()

# Add a line for each model
for column in df_full.columns:
    ax.plot(df_full[column], label=column)

# Add a legend and set axis labels
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

for model in df_full.columns:
    hf.polyfit((df_full[model].values).astype(float))

"""

print("ok, cool")
