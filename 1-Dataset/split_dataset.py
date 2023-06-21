import pandas as pd
import os
df = pd.read_csv('1-Dataset\\12000\DB-Output1200_Ttrain.csv',engine='python')


def filter_df(df, string):
  mask = df.apply(lambda x: x.str.contains(string, case=False))
  return df[mask]

moves=[]
for i in range(len(df)):
    string_input = df['Moves'][i]
    string_list = string_input[1:-1].split(", ")
    for i in range(len(string_list)):
        string_list_clean=string_list[i].replace("'","")
        moves.append(string_list_clean)

def unique(lst):
  unique_set = set()
  for elem in lst:
    unique_set.add(elem)
  return list(unique_set)

moves_unique=unique(moves)

root='1-Dataset\splitted_dataset12000_Train\\'

dir_name='1-Dataset\splitted_dataset12000_Train'
if not os.path.exists(dir_name):
    # se non esiste, crea la directory
    os.mkdir(dir_name)


for i in range(len(moves_unique)):
    filtered_df = df[df['Moves'].str.contains(moves_unique[i])]
    path=root+moves_unique[i]+".csv"
    filtered_df.to_csv(path, index=False)
    