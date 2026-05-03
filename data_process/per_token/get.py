import pandas as pd

df = pd.read_csv("./globalrag_qwen2.5-7b_update.csv")

print(df['update_actor_step'][:200].mean() * 100)
