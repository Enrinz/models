import os
import pandas as pd

df = pd.read_csv("1-Dataset\\12000\DB-Output12000_FULL.csv",low_memory=False)
dir = "1-Dataset\dataTrain"
files = os.listdir(dir)



filtered_df = df[df["Instance's Name"].isin(files)]

filtered_df.to_csv("1-Dataset\\12000\DB-Output1200_Ttrain.csv", index=False)
