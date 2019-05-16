# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 13:03:42 2018

@author: Steve O'Hagan
"""

# import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from IPython.display import SVG, display
#from rdkit import Chem
from SmilesTools import smiUtil

#Import Keras objects
from keras.models import Model
from keras.layers import Input, Masking, Flatten, Reshape, Activation,Softmax
from keras.layers import Dense, Bidirectional,RepeatVector
from keras.layers import GRU, TimeDistributed
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.utils.vis_utils import model_to_dot
from keras.utils import multi_gpu_model
from keras.regularizers import l1

import keras.backend as K

import optuna

from time import time

from os import listdir
from os.path import isfile, join

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

K.clear_session()

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

def jacc(yt,yp,sm=1e-6):
    n = yt.shape[0]
    yp=np.reshape(yp,(n,-1))
    yt=np.reshape(yt,(n,-1))
    intr = np.sum(yt*yp,axis=-1)
    sum_ = np.sum(yt + yp,axis=-1)
    jac = (intr + sm) / (sum_ - intr + sm)
    return np.mean(jac)

def jaccard(yt, yp, sm=1e-6):
    yt=K.batch_flatten(yt)
    yp=K.batch_flatten(yp)
    intr = K.sum(yt*yp,axis=-1)
    sum_ = K.sum(yt + yp,axis=-1)
    jac = (intr + sm) / (sum_ - intr + sm)
    return K.mean(jac)

def npbxe(yt,yp):
    yt=yt.reshape(-1)
    yp=yp.reshape(-1)
    bce = -(yt*np.log(yp) + (1.0 - yt)*np.log(1.0 - yp))
    bce = np.mean(bce)
    return bce


def bxe(yt, yp):
    yt=K.flatten(yt)
    yp=K.flatten(yp)
    bce = -(yt*K.log(yp) + (1.0 - yt)*K.log(1.0 - yp))
    bce = K.mean(bce)
    return bce

def jaccard_loss(y_true, y_pred, sm=1e-6):
    return (1.0 - jaccard(y_true,y_pred,sm))

def dice(y_true, y_pred, sm=1e-6):
    yp=K.batch_flatten(y_pred)
    yt=K.batch_flatten(y_true)
    intr = K.sum(yt*yp,axis=-1)
    sum_ = K.sum(K.square(yt)+K.square(yp),axis=-1) + sm
    dce = (2.0 * intr + sm) / sum_
    return K.mean(dce)

def dice_loss(y_true, y_pred, sm=1e-6):
    return 1.0 - dice(y_true, y_pred, sm)

class FCAE:

    @staticmethod
    def getNList(strt,end,cnt,pwr=0.6):
        p2 = 1.0/pwr
        t = np.linspace(strt,end**p2,cnt+1,endpoint=False)[1:]
        t = list(map(lambda x: int(np.power(x,pwr)),t))
        return t

    def __init__(self,smiLen,codeLen,lrep=4, nEL=1,nDL=1,opt='adam',p=0.6,ngpu=1,reg=0.001):
        self.ngpu=ngpu
        inputs = Input(shape=(smiLen,codeLen,),name='IN1')
        x = Flatten(name='F1')(inputs)
        NL = self.getNList(smiLen*codeLen,lrep,nEL,p)
        for L in range(nEL):
            nm=f'DE{L+1}'
            x = Dense(NL[L],activation='relu',name=nm)(x)
        enc = Dense(lrep,activation='relu',activity_regularizer=l1(reg),name=f'Enc')(x)
        NL = self.getNList(smiLen*codeLen,lrep,nDL,p)
        NL.reverse()
        for L in range(nDL):
            if L==0:
                x = Dense(NL[L],activation='relu',name='DD1')(enc)
            else:
                x = Dense(NL[L],activation='relu',name=f'DD{L+1}')(x)
        x = Dense(smiLen*codeLen,activation='relu',name='Out')(x)
        x = Reshape((smiLen,codeLen),name='RS1')(x)
        op = Softmax(axis=-1)(x)

        self.aen = Model(inputs,op)
        self.enc = Model(inputs,enc)
        N = 3 + nDL

        inp2 = Input(shape=(lrep,),name='IN2')
        deco = self.aen.layers[-N](inp2)
        for L in range(N-1):
            deco = self.aen.layers[L-N+1](deco)
        self.dec = Model(inp2,deco)
        if self.ngpu>1:
            self.mgm = ModelMGPU(self.aen,gpus=self.ngpu)
            self.mgm.compile(optimizer=opt, loss='binary_crossentropy',metrics=['acc',jaccard])
        else:
            self.aen.compile(optimizer=opt, loss='binary_crossentropy',metrics=['acc',jaccard])
            self.mgm = None

    def aeTrain(self,trn,fn,epo=2,bat=500,vb=0):
        modsave = f'data/{fn}.hdf5'
        chkptr = ModelCheckpoint(filepath = modsave, verbose = 0, save_best_only = True)
        rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, min_lr = 0.0001)
        tb = TensorBoard(log_dir=f'logs/{fn}')
        cblist=[chkptr,rlr,tb]
        if self.mgm is None:
            self.aen.fit(trn, trn, shuffle = True, epochs = epo, batch_size = bat,
                         callbacks = cblist, validation_split = 0.2,verbose=vb)
            self.aen.load_weights(modsave)
        else:
            self.mgm.fit(trn, trn, shuffle = True, epochs = epo, batch_size = bat,
                         callbacks = cblist, validation_split = 0.2,verbose=vb)
            self.mgm.load_weights(modsave)
        return modsave

    def loadw(self,modsave):
        if self.mgm is None:
            self.aen.load_weights(modsave)
        else:
            self.mgm.load_weights(modsave)

    def evaluate(self,x):
        if self.mgm is None:
            sc = self.aen.evaluate(x,x)
        else:
            sc = self.mgm.evaluate(x,x)
        return sc[0] #jaccard loss

    def jscore(self,xt):
        xp = self.aen.predict(xt)
        return jacc(xt,xp)

def plotm(model):
    display(SVG(model_to_dot(model,show_shapes=True).create(prog='dot', format='svg')))

def getFileList(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

#%%
if __name__ == "__main__":

    dfn = 'data/6MSmiles.csv'

    dat = pd.read_csv(dfn)

    smt = smiUtil(dat)

    k = 250000 #train size
    j = 2000    #tst size

    trn = dat[0:k]

    tst = dat[-j:]
    tst = tst.reset_index(drop=True)

    trnOH = smt.smi2OH(trn)
    tstOH = smt.smi2OH(tst)

    del dat

    pfix = 'test250kBXE_'

    def doHPOpt(trial):
        K.clear_session()
        gpus = 1
        batch = 512 * gpus
        EPO = 20
        pw = 0.6
        opt = 'adam'
        rg = 0.001
        lRep = trial.suggest_int('lrep',128,128)
        nE = trial.suggest_int('nEL',1,2)
        nD = trial.suggest_int('nDL',1,2)
        rg = trial.suggest_categorical('reg',[0.01,0.001,0.0001])
        batch = trial.suggest_categorical('batch',[64,128,256,512])
        trial.set_user_attr('opt',opt)
        trial.set_user_attr('p',pw)
        trial.set_user_attr('ngpu',gpus)
        #trial.set_user_attr('reg',rg)
        #pw = trial.suggest_uniform('pwr',0.5,1.0)

        fcn = FCAE(smt.smiLen,smt.codeLen,lrep=lRep,nEL=nE,nDL=nD,opt=opt,p=pw,ngpu=gpus,reg=rg)

        fcn.aeTrain(trnOH,f'{pfix}{trial.number}',EPO,batch,vb=1)
        bxe = fcn.evaluate(tstOH)
        print(f'BXE loss: {bxe}')
        return bxe

    study = optuna.create_study()

    t0 = time()
    study.optimize(doHPOpt,n_trials=20)
    t1 = time() - t0
    print(f'Time: {t1:.1f} sec')
    df = study.trials_dataframe()

    df2 = df['params'].copy()

    df2['bxe'] = df['value']

    df2['nD+E'] = df2['nDL'] + df2['nEL']

    df2.to_csv(f'data/{pfix}.csv')

    bp = { **study.best_params, **study.best_trial.user_attrs}

    bp.pop('batch')

    bt = study.best_trial.number

    bcn = FCAE(smt.smiLen,smt.codeLen, **bp)

    fn = f'data/{pfix}{bt}.hdf5'

    bcn.loadw(fn)

    xp = bcn.aen.predict(tstOH)

    bcn.evaluate(tstOH)

    smt.getScore(tstOH,xp,True)

    logdir = f'logs/{pfix}{bt}/'

    fn = getFileList(logdir)

    fn = fn[0]

    eacc = EventAccumulator(logdir+fn)

    eacc.Reload()

    print(eacc.Tags())

    tj = eacc.Scalars('loss')

    vj = eacc.Scalars('val_loss')

    steps = len(tj)

    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = tj[i][2] # value
        y[i, 1] = vj[i][2]

    plt.plot(x, y[:,0], label='training loss')
    plt.plot(x, y[:,1], label='validation loss')

    plt.xlabel("Steps")
    plt.ylabel("BXE Loss")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()


#    plotm(bcn.aen)
#    plotm(fcn.enc)
#    plotm(fcn.dec)

