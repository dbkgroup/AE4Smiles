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
from keras.layers import Dense, Bidirectional,RepeatVector
from keras.layers import GRU, TimeDistributed
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.vis_utils import model_to_dot

#%%

class smiUtil:

    def __init__(self,dat):
        #Find unique chars in smiles
        self.smiCodes=set(' ')
        self.smiLen = max([len(s) for s in dat['Molecule']])
        for s in dat['Molecule']:
            self.smiCodes.update(list(s))
        self.smiCodes = sorted(list(self.smiCodes))
        self.codeLen=len(self.smiCodes)
        self.code2int = dict((c,i) for i,c in enumerate(self.smiCodes))
        self.int2code = dict((i,c) for i,c in enumerate(self.smiCodes))

    def smi2OH(self,dat):
        rowCount,_= np.shape(dat)
        xs=np.zeros((rowCount,self.smiLen,self.codeLen),'f')
        for i,p in dat.iterrows():
            inP=list(p.Molecule.ljust(self.smiLen))
            for j,c in enumerate(inP):
                xs[i,j,self.code2int[c]] = 1.0
        return xs

    def reverseSS(self,x):
        if (np.ndim(x)==3):
            x=np.reshape(x,(self.smiLen,self.codeLen))
        xx=pd.DataFrame(x)
        xx.columns=self.smiCodes
        xx=list(xx.idxmax(axis=1))
        return "".join(xx)

    def getScore(self,tstOH,predOH,verbose=False):
        sm = 0.0
        nr = len(tstOH)
        perfect = 0
        good = 0.0
        for i in range(nr):
            xt=tstOH[i,:,:]
            yt=predOH[i,:,:]
            inp=self.reverseSS(xt)
            outp=self.reverseSS(yt)
            hit=self.cmpSmiles(inp,outp)
            if hit >= 1.0:
                perfect+=1
            if self.isGood(outp):
                good+=1
            #print(hit,self.isGood(outp))
            if verbose:
                print(inp)
                print(outp,flush=True)
            sm=sm+100.0*hit
        print(f'{100*perfect/nr:.2f}, {sm/nr:.2f}, {good/nr:.2f}')

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

#%%
class aen4smiles:

    def __init__(self,smt,LATENT=4,RNN=16):
        self.LATENT = LATENT
        self.RNN = RNN
        inputs = Input(shape=(smt.smiLen,smt.codeLen,))
        x = Masking()(inputs)
        x = Bidirectional(GRU(RNN*2,activation='relu',dropout=0.2, return_sequences=True))(x)
        x = GRU(RNN,activation='relu',dropout=0.2, return_sequences=True)(x)
        enc = GRU(LATENT,activation='relu',dropout=0.2, return_sequences=False)(x)
        #enc = TimeDistributed(Dense(LATENT*2,activation='relu'))(x)
        x = RepeatVector(smt.smiLen)(enc)
        x = TimeDistributed(Dense(LATENT*2,activation='relu'))(x)
        x = TimeDistributed(Dense(smt.codeLen,activation='sigmoid'))(x)

        #inp2 = Input(shape=(smiLen,LATENT,))
        self.aen = Model(inputs,x)

        #decoder = Model(inp2,x)
        self.encoder = Model(inputs,enc)

        self.aen.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['acc'])

    def aeTrain(self,trn,EPOCHS=2,BATCH=2000):
        self.model_save = 'smiles_ae_L' + str(self.LATENT) + '_E' + str(EPOCHS) + '_R' + str(self.RNN)+'.hdf5'
        print(self.model_save)
        self.EPOCHS = EPOCHS
        self.BATCH = BATCH
        #if not os.path.isfile(self.model_save):
        checkpointer = ModelCheckpoint(filepath = self.model_save, verbose = 1, save_best_only = True)
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, min_lr = 0.0001)
        self.aen.fit(trn, trn, shuffle = True, epochs = EPOCHS, batch_size = BATCH,
            callbacks = [checkpointer, reduce_lr], validation_split = 0.2)
        #self.aen.load_weights(self.model_save)
#%%
def plotm(model):
    display(SVG(model_to_dot(model,show_shapes=True).create(prog='dot', format='svg')))

#%%

if __name__ == "__main__":

    fn = 'data/6MSmiles.csv'

    dat = pd.read_csv(fn)

    smitool = smiUtil(dat)

    k = 100000
    j = 2000

    trn = dat.iloc[0:k]

    tst = dat.iloc[-j:]
    tst = tst.reset_index(drop=True)

    trnOH = smitool.smi2OH(trn)
    tstOH = smitool.smi2OH(tst)

    del dat

    nn = aen4smiles(smitool,LATENT=64)

    plotm(nn.aen)

    nn.aeTrain(trnOH,EPOCHS=10)

    #%%
    try:
        h = nn.aen.history
        plt.plot(h.history["loss"], label="Loss")
        plt.plot(h.history["val_loss"], label="Val_Loss")
        #plt.yscale("log")
        plt.legend()
    except:
        pass
#%%
    yTest = nn.aen.predict(tstOH)

#%%
    smitool.getScore(tstOH,yTest,verbose=True)




