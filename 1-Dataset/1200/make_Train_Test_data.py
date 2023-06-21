import os
import shutil

# Crea le nuove cartelle
os.makedirs("1-Dataset\dataTrain", exist_ok=True)
os.makedirs("1-Dataset\dataTest", exist_ok=True)

# Elenco dei file da spostare in TestData
test_files = ['c205_21_100.txt', 'c206_21_100.txt', 'c207_21_100.txt','c208_21_100.txt','r208_21_100.txt','r209_21_100.txt','r210_21_100.txt','r211_21_100.txt','rc206_21_100.txt','rc207_21_100.txt','rc208_21_100.txt','c106_21_100.txt','c107_21_100.txt','c108_21_100.txt','c109_21_100.txt','r112_21_100.txt','r111_21_100.txt','r110_21_100.txt','r109_21_100.txt','r108_21_100.txt','r107_21_100.txt','rc108_21_100.txt','rc107_21_100.txt','rc106_21_100.txt',
              'c205_21_50.txt', 'c206_21_50.txt', 'c207_21_50.txt','c208_21_50.txt','r208_21_50.txt','r209_21_50.txt','r210_21_50.txt','r211_21_50.txt','rc206_21_50.txt','rc207_21_50.txt','rc208_21_50.txt','c106_21_50.txt','c107_21_50.txt','c108_21_50.txt','c109_21_50.txt','r112_21_50.txt','r111_21_50.txt','r110_21_50.txt','r109_21_50.txt','r108_21_50.txt','r107_21_50.txt','rc108_21_50.txt','rc107_21_50.txt','rc106_21_50.txt',
              'c205_21_25.txt', 'c206_21_25.txt', 'c207_21_25.txt','c208_21_25.txt','r208_21_25.txt','r209_21_25.txt','r210_21_25.txt','r211_21_25.txt','rc206_21_25.txt','rc207_21_25.txt','rc208_21_25.txt','c106_21_25.txt','c107_21_25.txt','c108_21_25.txt','c109_21_25.txt','r112_21_25.txt','r111_21_25.txt','r110_21_25.txt','r109_21_25.txt','r108_21_25.txt','r107_21_25.txt','rc108_21_25.txt','rc107_21_25.txt','rc106_21_25.txt']
print(len(test_files),test_files)
# Sposta i file in TestData
for file_name in test_files:
    shutil.move(os.path.join("1-Dataset\data", file_name), os.path.join("1-Dataset\dataTest", file_name))

# Sposta il resto dei file in TrainData
for file_name in os.listdir("1-Dataset\data"):
    shutil.move(os.path.join("1-Dataset\data", file_name), os.path.join("1-Dataset\dataTrain", file_name))
