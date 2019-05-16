# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:32:51 2019

@author: Steve O'Hagan
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import SVG, display
from rdkit import Chem
from time import time
import pickle

#Import Keras objects
from keras.models import Model
from keras.layers import Input, Masking
from keras.layers import Dense, Bidirectional
from keras.layers import GRU, TimeDistributed
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.vis_utils import model_to_dot

#%%

class SmilesUtil():

    @staticmethod
    def cmpSmiles(s1,s2):
        s1=s1.strip()
        s2=s2.strip()
        mx=max(len(s1),len(s2))
        s1=s1.ljust(mx)
        s2=s2.ljust(mx)
        hit=sum([x==y for x,y in zip(s1,s2)])
        return hit/mx

    @staticmethod
    def isGood(smi):
        m = Chem.MolFromSmiles(smi)
        return m is not None

    def __init__(self,dat):
        #Find unique chars in smiles
        #takes 580s for 6 million items
        self.smiCodes=set()
        self.smiLen = 0
        for i,p in dat.iterrows():
            s = p.Molecule
            if i % 1000 == 0:
                print(i,s)
            self.smiLen = max(self.smiLen,len(s))
            s = s.ljust(self.smiLen)
            self.smiCodes.update(list(s))
        self.smiCodes = sorted(list(self.smiCodes))
        self.codeLen=len(self.smiCodes)
        self.code2int = dict((c,i) for i,c in enumerate(self.smiCodes))
        self.int2code = dict((i,c) for i,c in enumerate(self.smiCodes))


    def to_OH(self,dat):
        rowCount,_= np.shape(dat)
        xs=np.zeros((rowCount,self.smiLen,self.codeLen),'f')
        for i,p in dat.iterrows():
            inP=list(p.Molecule.ljust(self.smiLen))
            for j,c in enumerate(inP):
                xs[i,j,self.code2int[c]] = 1.0
        return xs

    def oh2Smiles(self,oh):
        rslt = map(self.reverseSS,oh)
        return list(rslt)

    def reverseSS(self,x):
        if (np.ndim(x)==3):
            x=np.reshape(x,(self.smiLen,self.codeLen))
        xx=pd.DataFrame(x)
        xx.columns=self.smiCodes
        xx=list(xx.idxmax(axis=1))
        s = "".join(xx)
        return s.strip()

class AE4Smiles:

    def __init__(self,smiObj,LATENT=4,RNN=16):
        self.LATENT = LATENT
        self.RNN = RNN
        self.smiObj=smiObj
        inputs = Input(shape=(smiObj.smiLen,smiObj.codeLen,))
        x = Masking()(inputs)
        x = Bidirectional(GRU(RNN,dropout=0.2, return_sequences=True))(x)
        x = TimeDistributed(Dense(LATENT*2,activation='relu'))(x)
        enc = TimeDistributed(Dense(LATENT,activation='relu'))(x)
        x = TimeDistributed(Dense(LATENT*2,activation='relu'))(enc)
        x = TimeDistributed(Dense(smiObj.codeLen,activation='sigmoid'))(x)

        #inp2 = Input(shape=(smiLen,LATENT,))
        self.aen = Model(inputs,x)

        #decoder = Model(inp2,x)
        self.encoder = Model(inputs,enc)

        self.aen.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])

    def aeTrain(self,name,sTrain,sValid, EPOCHS=1,BATCH=64):
        self.model_save = name + '_SAE' + str(self.LATENT) + '_E' + str(EPOCHS) + '_R' + str(self.RNN)+'.hdf5'
        print(self.model_save)
        self.EPOCHS = EPOCHS
        self.BATCH = BATCH
        if not os.path.isfile(self.model_save):
            checkpointer = ModelCheckpoint(filepath = self.model_save, verbose = 1, save_best_only = True)
            reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, min_lr = 0.0001)
            self.aen.fit(sTrain, sTrain, shuffle = True, epochs = EPOCHS, batch_size = BATCH,
                callbacks = [checkpointer, reduce_lr], validation_data = (sValid,sValid))
        self.aen.load_weights(self.model_save)


def plotm(model):
    display(SVG(model_to_dot(model,show_shapes=True).create(prog='dot', format='svg')))

def getSOH():
    pth='data/6MSmiles.csv'
    dat = pd.read_csv(pth)
    pkFile = 'data/6MSmiles.pkl'
    t0 = time()
    if os.path.exists(pkFile):
        print('Loading SMILES codes.')
        with open(pkFile, 'rb') as f:
            su = pickle.load(f)
    else:
        print('Calculating SMILES codes.')
        su = SmilesUtil(dat)
        with open(pkFile,'wb') as f:
            pickle.dump(su,f)
    t1 = time() - t0
    print('Time:',t1)
    return dat,su


#%%
if __name__ == "__main__":

    dat,su = getSOH()

    kk = 25000

    trnDat = dat[0:kk]

    #2k from end & reindex
    vldDat = dat.iloc[-2000:]
    vldDat = vldDat.reset_index(drop=True)
    tstDat = dat.iloc[-4000:-2000]
    tstDat = tstDat.reset_index(drop=True)

    del dat

    trd = su.to_OH(trnDat)
    vld = su.to_OH(vldDat)
    tsd = su.to_OH(tstDat)

    nn = AE4Smiles(su,LATENT=1)

    plotm(nn.aen)

    nn.aeTrain('25k',trd,vld,EPOCHS=16)

    yTest = nn.aen.predict(tsd)

#%%
    sm = 0.0
    perfect = 0
    good = 0.0
    nr = len(tsd)
    st = su.oh2Smiles(tsd)
    sy = su.oh2Smiles(yTest)
    for x,y in zip(st,sy):
        hit=su.cmpSmiles(x,y)
        if hit >= 1.0:
            perfect+=1
        if su.isGood(y):
            good+=1
        #print(hit,su.isGood(y))
        print(x)
        print(y,flush=True)
        sm=sm+100.0*hit
    print(f'Perfect: {100*perfect/nr:.2f}, Good:{100*good/nr:.2f}, Match:{sm/nr:.2f}')

    tenc = nn.encoder.predict(tsd)


#%%
    try:
        h = nn.aen.history.history
        plt.plot(h["acc"], label="acc")
        plt.plot(h["val_acc"], label="Val_acc")
        #plt.yscale("log")
        plt.legend()
    except:
        pass




