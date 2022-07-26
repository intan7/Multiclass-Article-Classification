# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:14:29 2022

@author: intan
"""

from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model

from Multiclass_Article_Classification_module import ModelDevelopment,ModelEvaluation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import pickle
import json
import os
import re

#%%
URL_PATH='https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
TOKENIZER_SAVE_PATH=os.path.join(os.getcwd(),'saved_models','tokenizer.json')
OHE_SAVE_PATH=os.path.join(os.getcwd(),'saved_models','ohe.pkl')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'saved_models','model.h5')
#%%
#Step 1)Data Loading
df=pd.read_csv(URL_PATH)

#%% Step 2) Data inspection
df.head()
df.info()

print(df['text'][20])
print(df['text'][100])

#No Symbols and HTML Tags to be removed
#there's extra space
#%%Step 3) Data cleaning

df.isna().sum() #no NaNs

text=df['text']
category=df['category']

#to remove extra space
for index,texts in enumerate(text):
  text[index]=re.sub('[^a-zA-Z]',' ',texts).lower().split()

#create backup
text_backup=text.copy()
cat_backup=category.copy()

#%% Step 4) Features selection - nothing to select
#%% Step 5) Data preprocessing 

vocab_size=10000
oov_token='<OOV>'

tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(text) #to learn
word_index=tokenizer.word_index

print(dict(list(word_index.items())[0:10]))

text_int=tokenizer.texts_to_sequences(text) #to convert to number
text_int[100] #to check all convert to number

max_len=np.median([len(text_int[i])for i in range(len(text_int))])

padded_text=pad_sequences(text_int,
                            maxlen=int(max_len),
                            padding='post',
                            truncating='post')

#Y target
ohe=OneHotEncoder(sparse=False)
category=ohe.fit_transform(np.expand_dims(category,axis=-1))

#Split train & test 
X_train,X_test,y_train,y_test=train_test_split(padded_text,category,
                                               test_size=0.3,
                                               random_state=(123))

#%%Model development

input_shape=np.shape(X_train)[1:]
output=len(np.unique(y_train,axis=0))

md=ModelDevelopment()
model=md.simple_MD_model(input_shape,vocab_size,output,nb_node=128,dropout_rate=0.3)

plot_model(model,show_shapes=(True))

#%% Model Training

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['acc'])

#Callbacks
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)#patience-let it wait for 5 times not improving on validation loss


hist = model.fit(X_train, y_train, 
                 epochs=5,
                 validation_data=(X_test, y_test),
                 callbacks=[tensorboard_callback,early_stopping_callback])

#%% Model Evaluation

print(hist.history.keys())

me=ModelEvaluation()
me.Plot_Hist(hist,0,2) #to look for loss & val_loss

me=ModelEvaluation()
me.Plot_Hist(hist,1,3) #to look for acc & val_acc

#%% Model Analysis

y_pred=np.argmax(model.predict(X_test),axis=1)
y_actual=np.argmax(y_test,axis=1)
labels = ['tech','business','sport','entertainment','politics']

#classification report
print(classification_report(y_actual, y_pred, target_names = labels))

#confusion matrix
cm=confusion_matrix(y_actual,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()
#%% Model Saving

#TOKENIZER
token_json=tokenizer.to_json()

with open(TOKENIZER_SAVE_PATH,'w') as file:
    json.dump(token_json,file)
    
#OHE
with open(OHE_SAVE_PATH,'wb') as file:
    pickle.dump(ohe,file)
    
#MODEL
model.save(MODEL_SAVE_PATH)




