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
    pth = file_folder + "/" + file
    with open(pth, 'rb') as f:
        df = pickle.load(f)
        df.columns = df.iloc[0, :]
        df = df.drop(0)
    df_full[f"model_{i+1}"] = df['pf_std']


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
fig.show()

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
