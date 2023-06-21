import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os
import numpy as np
from keras.layers import Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.model_selection import StratifiedKFold
import random
num='12000Train'          #
epochs=20          #
batch_size=32       #

dir = "1-Dataset\splitted_dataset"+str(num)    
files = os.listdir(dir)


for file in files:
    X=[]
    Y=[]
    count_IMPROVE=0  # if OF_Diff>0
    count_NOTIMPROVE=0 # if OF_Diff<=0


    dir_models_bin="6-Convolutional\models_10fold_DEF\\"+str(num)+"\\_"+str(epochs)+"ep_"+str(batch_size)+"bs"
    if not os.path.exists(dir_models_bin):
        os.makedirs(dir_models_bin)


    dir_f="6-Convolutional\\tests_10fold_DEF"
    if not os.path.exists(dir_f):
        os.makedirs(dir_f)

    model_path=dir_models_bin+"\\"+file+'.h5'
    if os.path.exists(model_path):
        continue
    else:
        f = open(dir_f+"\\"+str(num)+"_"+str(epochs)+"ep_"+str(batch_size)+"bs"+".txt", "a")


        #create dataframe
        df = pd.read_csv(dir+"\\"+file) 
        # Calcolo della media degli elementi ripetuti
        grouped = df.groupby('Initial Solution')['OF_Diff'].transform('mean')
        # Determinazione dei valori di partenza
        original_values = df['OF_Diff'].where(grouped <= 1, df['OF_Diff'])
        # Creazione del nuovo dataframe
        new_df = pd.DataFrame({'Instance\'s Name':df['Instance\'s Name'],'Initial Solution': df['Initial Solution'], 'OF_Diff_Average': grouped})
        new_df['OF_Diff_Average'] = new_df['OF_Diff_Average'].fillna(original_values)
        new_df = new_df.drop_duplicates().reset_index(drop=True)
        df=new_df
        df_pos=df[df['OF_Diff_Average']>0]
        num_valori_unici = df_pos["Instance's Name"].nunique()
        num_for_instances=int(len(df_pos)/num_valori_unici)
        
        print(num_for_instances)
        df_neu=df[df['OF_Diff_Average']==0] 


        #negative prese random
        #df_neu_sample = df_neu.sample(n=len(df_pos),replace=True)   #n=len(df_pos),random_state=42
        #df_totale = pd.concat([df_pos, df_neu_sample])

        #negative prese in ordine, le prime negative
        df_neu_sample = df_neu.groupby("Instance's Name").head(num_for_instances)

        while len(df_neu_sample) < len(df_pos):
            random_row = df_neu.sample(n=1)
            if not random_row.values.tolist() in df_neu_sample.values.tolist():
                df_neu_sample = df_neu_sample.append(random_row, ignore_index=True)

        df_totale = pd.concat([df_pos, df_neu_sample])
        print(len(df_neu_sample),len(df_pos))
        df_totale_shuffled = df_totale.sample(frac=1).reset_index(drop=True)
        df=df_totale_shuffled
        df = df.reset_index(drop=True)
        for i in range(len(df)):
            X.append(df['Initial Solution'][i].replace('\'','').replace('[','').replace(']','').replace(',',''))
            if(df['OF_Diff_Average'][i])>0:
                count_IMPROVE+=1
                Y.append(1)
            else: 
                Y.append(0)
                count_NOTIMPROVE+=1

        Y = np.asarray(Y, dtype=np.float32)
        # creazione del tokenizer
        tokenizer = Tokenizer(num_words=150)
        tokenizer.fit_on_texts(X)
        X = tokenizer.texts_to_sequences(X)

        maxlen = 270
        X = pad_sequences(X, padding='post', maxlen=maxlen)


        ############################################################################# Create a StratifiedKFold object
        n_splits = 10  # Number of folds
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)#, shuffle=True, random_state=42

        accuracy_scores = []
        f1_scores=[]
        # Iterate through the StratifiedKFold object
        print("\nDataset: ",file,file=f)
        for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
            print(f"Fold {i+1}:")
            print(f"Fold {i+1}:",file=f)
            # Split the data into training and testing sets
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            print(len(X_train),len(Y_train))
            ##############################################################################

            model = Sequential()
            model.add(Embedding(150, 128, input_length=maxlen))
            model.add(Conv1D(64, 5, activation='relu'))
            model.add(GlobalMaxPooling1D())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))

            # Compilazione del modello
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, Y_train, epochs=epochs,batch_size=batch_size, validation_data=(X_test, Y_test))

            # valutazione del modello sui dati di test
            y_pred = model.predict(X_test)

            # calcolo delle metriche di accuracy e F1
            y_pred_binary = [1 if y >= 0.5 else 0 for y in y_pred]
            tn_test, fp_test, fn_test, tp_test = confusion_matrix(Y_test, y_pred_binary).ravel()
            test_accuracy = accuracy_score(Y_test, y_pred_binary)
            test_f1 = f1_score(Y_test, y_pred_binary)

            accuracy_scores.append(test_accuracy)
            f1_scores.append(test_f1)

            print("Dataset: ",file,file=f)
            print("Test Accuracy:", test_accuracy,file=f)
            print("Test F1-score:", test_f1,file=f)
            print("Test Confusion Matrix:\n", confusion_matrix(Y_test, y_pred_binary),file=f)
            print("Test TP:", tp_test,file=f)
            print("Test TN:", tn_test,file=f)
            print("Test FP:", fp_test,file=f)
            print("Test FN:", fn_test,file=f)
            print("\n###########################################################################\n",file=f)
        model.save(model_path)
        print("\n********************\n")
        print("\n********************\n",file=f)
        print("Dataset: ",file,file=f)
        print("Final test metrics after ",n_splits," folds:\n")
        print("Final test metrics after ",n_splits," folds:\n",file=f)
        print("Average accuracy:", np.mean(accuracy_scores),"\n","Average F1: ",np.mean(f1_scores))
        print("Average accuracy:", np.mean(accuracy_scores),"\n","Average F1: ",np.mean(f1_scores),file=f)
        print("Dev Std accuracy:", np.std(accuracy_scores),"\n","Dev Std F1: ",np.std(f1_scores))
        print("Dev Std accuracy:", np.std(accuracy_scores),"\n","Dev Std F1: ",np.std(f1_scores),file=f)
        print("\n********************\n",file=f)
        print("\n********************\n")