import pandas as pd
import numpy as np

path1 = "./globalrag_qwen2.5-3b-base.csv"
path2 = "./globalrag_qwen2.5-3b.csv"
path3 = "./globalrag_qwen2.5-7b-base.csv"
path4 = "./globalrag_qwen2.5-7b.csv"

path_list = [path1, path2, path3, path4]
for path in path_list:
    df = pd.read_csv(path)
    df['m'] = 1 / (1 + np.exp((df['step'] - 45) / 10)) * 1.6 + 1
    df['rewards_scaled'] = df['rewards'] / df['m']

    df.to_csv(path, index=False)