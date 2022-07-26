# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:10:00 2022

@author: intan
"""

from tensorflow.keras.layers import LSTM,Dense,Dropout,Embedding,Bidirectional
from tensorflow.keras import Input,Sequential
import matplotlib.pyplot as plt


class ModelDevelopment:
    def simple_MD_model(self,input_shape,vocab_size,nb_class,nb_node=128, dropout_rate=0.3):
        
        model=Sequential()
        model.add(Input(shape=(input_shape)))
        model.add(Embedding(vocab_size,nb_node))
        model.add(Bidirectional(LSTM(nb_node,return_sequences=(True))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(nb_node)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_class, activation='softmax'))
        model.summary()
        
        return model
    
class ModelEvaluation:
    def Plot_Hist(self,hist,loss=0,vloss=2):
        a=list(hist.history.keys())
        plt.figure()
        plt.plot(hist.history[a[loss]])
        plt.plot(hist.history[a[vloss]])
        plt.legend(['training_'+ str(a[loss]), a[vloss]])
        plt.show()
