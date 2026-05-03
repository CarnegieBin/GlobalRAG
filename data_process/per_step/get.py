import pandas as pd

df = pd.read_csv("globalrag_qwen2.5-7b.csv")

print(df['step_step'][:200].mean())
