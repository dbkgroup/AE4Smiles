# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 13:03:42 2018

@author: Steve O'Hagan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from SmilesTools import smiUtil as SU

from AE4SmilesLib import CNAE

import keras.backend as K

import optuna

from time import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


#%%
if __name__ == "__main__":

    dfn = 'data/6MSmiles.csv'

    dat = pd.read_csv(dfn)

    smt = SU(dat)

    k = 250000 #train size
    j = 2000    #tst size

    trn = dat[0:k]

    tst = dat[-j:]
    tst = tst.reset_index(drop=True)

    trnOH = smt.smi2OH(trn)
    tstOH = smt.smi2OH(tst)

    del dat

    pfix = 'test250kCNNv3C'

    def doHPOpt(trial):
        K.clear_session()

        pp = {
            'lrep' : trial.suggest_int('lrep',4,256),
            'nEL' : trial.suggest_int('nEL',1,1),
            'reg' : trial.suggest_loguniform('reg',1.0e-8,1.0e-2),
            'flt' : trial.suggest_categorical('flt',[32,64,128,256]),
            'kern' : trial.suggest_categorical('kern',[5]),
            'opt' : trial.suggest_categorical('opt',['adam']),
            'ngpu' : trial.suggest_categorical('ngpu',[1]),
            'batch' : trial.suggest_categorical('batch',[256]),
            'EPO' : trial.suggest_categorical('EPO',[20])
            }

        fcn = CNAE(smt,**pp)

        fcn.aeTrain(trnOH,f'{pfix}{trial.number}',vb=1)
        loss,acc,jacc,dloss = fcn.evaluate(tstOH)
        trial.set_user_attr('Acc',acc)
        trial.set_user_attr('Jaccard',jacc)
        trial.set_user_attr('DiceLoss',dloss)
        #print(f'BXE loss: {bxe}')
        return loss

    study = optuna.create_study()

#%%
    t0 = time()
    study.optimize(doHPOpt,n_trials=20)
    t1 = time() - t0
    print(f'Time: {t1:.1f} sec')
    df = study.trials_dataframe()

    df2 = df['params'].copy()

    df2['loss'] = df['value']

    df3 = df['user_attrs']

    df2 = pd.concat([df2,df3],axis=1)

    df2.to_csv(f'data/{pfix}.csv')

    bp = study.best_params

    bt = study.best_trial.number

    bcn = CNAE(smt, **bp)

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

    #print(eacc.Tags())

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
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()

#=====================================================

