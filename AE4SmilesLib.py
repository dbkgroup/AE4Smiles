# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 13:03:42 2018

@author: Steve O'Hagan
"""

import numpy as np
import matplotlib.pyplot as plt

from IPython.display import SVG, display

#Import Keras objects
from keras.models import Model
from keras.layers import Input, Flatten, Reshape, Softmax
from keras.layers import Dense, UpSampling1D
from keras.layers import Conv1D, MaxPool1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.utils.vis_utils import model_to_dot
from keras.utils import multi_gpu_model
from keras.regularizers import l1
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import keras.backend as K

from os import listdir
from os.path import isfile, join

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

def jaccard(yt, yp, sm=1e-6):
    yt=K.batch_flatten(yt)
    yp=K.batch_flatten(yp)
    intr = K.sum(yt*yp,axis=-1)
    sum_ = K.sum(yt + yp,axis=-1)
    jac = (intr + sm) / (sum_ - intr + sm)
    return K.mean(jac)

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

def Wo(W,F=2,S=2):
    P = W % 2
    return (W-F+2*P)//S+1

class CNAE:

    def __init__(self,smt,lrep=4, nEL=1,opt='adam',
                 ngpu=1,reg=0.001,kern=5,flt=32,batch=256,EPO=1):
        self.ngpu=ngpu
        self.EPO=EPO
        self.batch=batch

        smiLen = smt.smiLen
        codeLen = smt.codeLen

        k = smiLen

        inputs = Input(shape=(smiLen,codeLen,),name='IN1')
        for L in range(nEL):
            if L==0:
                x = Conv1D(flt,kern,name='C1',activation='relu',padding='same')(inputs)
            else:
                x = Conv1D(flt,kern,name=f'C{L+1}',activation='relu',padding='same')(x)
            x = MaxPool1D(2,padding='same')(x)
            k = Wo(k)
        x = Conv1D(flt,kern,name=f'C{nEL+1}',activation='relu',padding='same')(x)
        x = MaxPool1D(2,padding='same')(x)
        k = Wo(k)

        x = Flatten()(x)
        enc = Dense(lrep,name='Encoded',activation='relu',activity_regularizer=l1(reg))(x)
        x = Dense(k*flt)(enc)
        x = Reshape((k,flt,))(x)
        k2 = k
        for L in range(nEL):
            x = Conv1D(flt,kern,name=f'D{L+1}',activation='relu',padding='same')(x)
            x = UpSampling1D(2)(x)
            k2 = k2 * 2
        x = Conv1D(smt.codeLen,kern,name=f'D{nEL+1}',activation='relu',padding='same')(x)
        x = UpSampling1D(2)(x)
        k2 = k2 * 2
        f = k2 - smt.smiLen + 1
        x = Conv1D(smt.codeLen,f,name='outp',padding='valid')(x)
        op = Softmax(axis=-1)(x)

        self.enc = Model(inputs,enc)
        self.aen = Model(inputs,op)

        N = 6 + nEL * 2

        inp2 = Input(shape=(lrep,),name='IN2')
        for L in range(N):
            if L==0:
                deco = self.aen.layers[L-N](inp2)
            else:
                deco = self.aen.layers[L-N](deco)

        self.dec = Model(inp2,deco)

        if self.ngpu>1:
            self.mgm = ModelMGPU(self.aen,gpus=self.ngpu)
            self.mgm.compile(optimizer=opt, loss='binary_crossentropy',metrics=['acc',jaccard,dice_loss])
        else:
            self.aen.compile(optimizer=opt, loss='binary_crossentropy',metrics=['acc',jaccard,dice_loss])
            self.mgm = None

    def aeTrain(self,trn,fn,vb=0,mdelta=0.0002,esp=10,EPO=None):
        if EPO is None:
            epo = self.EPO
        else:
            epo = EPO
        bat=self.batch
        modsave = f'data/{fn}.hdf5'
        chkptr = ModelCheckpoint(filepath = modsave, verbose = 0, save_best_only = True)
        rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 2, min_lr = 0.000001)
        lgd = f'logs/{fn}'
        tb = TensorBoard(log_dir=lgd)
        estp = EarlyStopping(patience=esp,restore_best_weights=True,min_delta=mdelta)
        cblist=[chkptr,estp,tb,rlr]
        if self.mgm is None:
            self.aen.fit(trn, trn, shuffle = True, epochs = epo, batch_size = bat,
                         callbacks = cblist, validation_split = 0.2,verbose=vb)
            self.aen.load_weights(modsave)
        else:
            self.mgm.fit(trn, trn, shuffle = True, epochs = epo, batch_size = bat,
                         callbacks = cblist, validation_split = 0.2,verbose=vb)
            self.mgm.load_weights(modsave)
        return modsave,lgd

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
        return sc #0 -> loss

def plotm(model):
    display(SVG(model_to_dot(model,show_shapes=True).create(prog='dot', format='svg')))

def getFileList(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

def tbHistoryPlot(lgd):
    fn = getFileList(lgd)
    fn = fn[-1]
    eacc = EventAccumulator(lgd+'/'+fn)
    eacc.Reload()
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
    plt.ylabel("Loss")
    plt.title("Re-Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()
        
#%%
if __name__ == "__main__":

    pass
