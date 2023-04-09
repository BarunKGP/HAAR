from tqdm.auto import trange
from tqdm import tqdm
from time import sleep
import pickle
import pandas as pd
import os

# for i in tqdm(range(20), asc=True, desc='1st loop'):
#     for j in tqdm(range(5), desc='2nd loop'):
#         sleep(0.1)

loc = '../../data/train_100.pkl'
with open(loc, 'rb') as handle:
    df = pickle.load(handle)
    print(df.info())
    print(df.head())