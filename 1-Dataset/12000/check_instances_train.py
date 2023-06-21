import pandas as pd
import os

# Carica il dataframe
dataframe = pd.read_csv('1-Dataset\\12000\DB-Output1200Ttrain.csv')

# Percorso della cartella
cartella = '1-Dataset\dataTrain'

# Recupera la lista dei file nella cartella
file_nella_cartella = os.listdir(cartella)

# Verifica la presenza degli elementi nel dataframe
tutti_presenti = all(elemento in dataframe['colonna_di_interesse'] for elemento in file_nella_cartella)

if tutti_presenti:
    print("Tutti gli elementi della cartella sono presenti nel dataframe.")
else:
    print("Alcuni elementi della cartella non sono presenti nel dataframe.")
