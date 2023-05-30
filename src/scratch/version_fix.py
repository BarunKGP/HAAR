import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])
print(df.head())
df.to_pickle(sys.argv[2])
