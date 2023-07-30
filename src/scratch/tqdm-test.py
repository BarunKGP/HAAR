import os
import pickle
from time import sleep

import pandas as pd
from tqdm import tqdm
from tqdm.auto import trange


loc = "../../data/train_100.pkl"
with open(loc, "rb") as handle:
    df = pickle.load(handle)
    print(df.info())
    print(df.head())
