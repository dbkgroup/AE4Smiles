# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 13:03:42 2018

@author: Steve O'Hagan
"""

#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import SVG, display
from rdkit import Chem

#Import Keras objects
from keras.models import Model
from keras.layers import Input, Masking
from keras.layers import Dense, Bidirectional
from keras.layers import GRU, TimeDistributed
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.vis_utils import model_to_dot

#%%
class smiData:

    def __init__(self,fn='ae4smiles.csv'):
        dat = pd.read_csv(fn)
        #Find unique chars in smiles
        self.smiCodes=set()
        self.smiLen = 0
        for i,p in dat.iterrows():
            s = p.Molecule
            self.smiLen = max(self.smiLen,len(s))
            s = s.ljust(self.smiLen)
            self.smiCodes.update(list(s))
        self.smiCodes = sorted(list(self.smiCodes))
        self.codeLen=len(self.smiCodes)
        rowCount,_= np.shape(dat)
        xs=np.zeros((rowCount,self.smiLen,self.codeLen),'f')
        self.code2int = dict((c,i) for i,c in enumerate(self.smiCodes))
        self.int2code = dict((i,c) for i,c in enumerate(self.smiCodes))
        for i,p in dat.iterrows():
            inP=list(p.Molecule.ljust(128))
            for j,c in enumerate(inP):
                xs[i,j,self.code2int[c]] = 1.0
        self.sTrain, self.sTest = train_test_split(xs, random_state=42)

class aen4smiles:

    def __init__(self,data,LATENT=4,RNN=16):
        self.LATENT = LATENT
        self.RNN = RNN
        self.data=data
        inputs = Input(shape=(data.smiLen,data.codeLen,))
        x = Masking()(inputs)
        x = Bidirectional(GRU(RNN,dropout=0.2, return_sequences=True))(x)
        x = TimeDistributed(Dense(LATENT*2,activation='relu'))(x)
        enc = TimeDistributed(Dense(LATENT,activation='relu'))(x)
        x = TimeDistributed(Dense(LATENT*2,activation='relu'))(enc)
        x = TimeDistributed(Dense(data.codeLen,activation='sigmoid'))(x)

        #inp2 = Input(shape=(smiLen,LATENT,))
        self.aen = Model(inputs,x)

        #decoder = Model(inp2,x)
        self.encoder = Model(inputs,enc)

        self.aen.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['acc'])

    def aeTrain(self,EPOCHS=100,BATCH=64):
        self.model_save = 'smiles_ae_L' + str(self.LATENT) + '_E' + str(EPOCHS) + '_R' + str(self.RNN)+'.hdf5'
        print(self.model_save)
        self.EPOCHS = EPOCHS
        self.BATCH = BATCH
        if not os.path.isfile(self.model_save):
            checkpointer = ModelCheckpoint(filepath = self.model_save, verbose = 1, save_best_only = True)
            reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, min_lr = 0.0001)
            self.aen.fit(data.sTrain, data.sTrain, shuffle = True, epochs = EPOCHS, batch_size = BATCH,
                callbacks = [checkpointer, reduce_lr], validation_split = 0.2)
        self.aen.load_weights(self.model_save)

def plotm(model):
    display(SVG(model_to_dot(model,show_shapes=True).create(prog='dot', format='svg')))

def reverseSS(x,data):
    if (np.ndim(x)==3):
        x=np.reshape(x,(data.smiLen,data.codeLen))
    xx=pd.DataFrame(x)
    xx.columns=data.smiCodes
    xx=list(xx.idxmax(axis=1))
    return "".join(xx)

def cmpSmiles(s1,s2):
    s1=s1.strip()
    s2=s2.strip()
    mx=max(len(s1),len(s2))
    s1=s1.ljust(mx)
    s2=s2.ljust(mx)
    hit=sum([x==y for x,y in zip(s1,s2)])
    return hit/mx

def isGood(smi):
    m = Chem.MolFromSmiles(smi)
    return m is not None

#%%

if __name__ == "__main__":

    data = smiData()

    nn = aen4smiles(data)

    plotm(nn.aen)

    nn.aeTrain()

    #%%
    try:
        h = nn.aen.history
        plt.plot(h.history["loss"], label="Loss")
        plt.plot(h.history["val_loss"], label="Val_Loss")
        plt.yscale("log")
        plt.legend()
    except:
        pass
#%%
    yTest = nn.aen.predict(data.sTest)

#%%
    sm = 0.0
    nr = len(data.sTest)
    perfect = 0
    good = 0.0
    for i in range(nr):
        xt=data.sTest[i,:,:]
        yt=yTest[i,:,:]
        inp=reverseSS(xt,data)
        outp=reverseSS(yt,data)
        hit=cmpSmiles(inp,outp)
        if hit >= 1.0:
            perfect+=1
        if isGood(outp):
            good+=1
        print(hit,isGood(outp))
        print(inp)
        print(outp,flush=True)
        sm=sm+100.0*hit
    print(100*perfect/nr,sm/nr,good/nr)

