import csv
# Lista dei nomi dei file CSV da unire
file_paths  = ['1-Dataset\\12000\DB-Output_12000_first6.csv', '1-Dataset\\12000\DB-Output_12000_7to10.csv', '1-Dataset\\12000\DB-Output_12000_11to32.csv', '1-Dataset\\12000\DB-Output_12000_33toend.csv','1-Dataset\\12000\DB-Output_12000_remaining.csv','1-Dataset\\12000\DB-Output_12000_remaining2.csv','1-Dataset\\12000\DB-Output_12000_remaining3.csv']

# Definisci il percorso del file CSV risultante
output_file = '1-Dataset\\12000\DB-Output12000_FULL.csv'

# Apri il file CSV risultante in modalità scrittura
with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    header_written = False  # Aggiungi una variabile per tenere traccia se l'intestazione è già stata scritta
    
    # Ciclo sui file di input
    for file_path in file_paths:
        with open(file_path, 'r') as infile:
            reader = csv.reader(infile)
            
            # Copia l'intestazione dal primo file solo se non è stata ancora scritta
            if not header_written:
                header = next(reader)
                writer.writerow(header)
                header_written = True  # Imposta la variabile su True dopo aver scritto l'intestazione
            else:
                next(reader)  # Salta l'intestazione nei file successivi
            
            # Copia i dati da ciascun file nel file di output
            for row in reader:
                writer.writerow(row)

print("Unione completata. Il file risultante è", output_file)