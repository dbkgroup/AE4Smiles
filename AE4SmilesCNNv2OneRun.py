# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 13:03:42 2018

@author: Steve O'Hagan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K

from time import time

from AE4SmilesLib import CNAE, getFileList

from SmilesTools import smiUtil as SU

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

#%%
if __name__ == "__main__":

    dfn = 'data/6MSmiles.csv'

    dat = pd.read_csv(dfn)

    smt = SU(dat)

    k = 5000000 #train size
    j = 2000    #tst size

    trn = dat[0:k]

    tst = dat[-j:]
    tst = tst.reset_index(drop=True)

    trnOH = smt.smi2OH(trn)
    tstOH = smt.smi2OH(tst)

    del dat

    pfix = 'test5MCNNv3Co1'

    t0 = time()

    bp = {
            'lrep' : 145,
            'nEL' : 1,
            'reg' : 1.0e-8,
            'flt' : 32,
            'kern' : 5,
            'opt' : 'adam',
            'ngpu' : 1,
            'batch' : 256,
            'EPO' : 60
            }

    K.clear_session()
    bcn = CNAE(smt,**bp)
    fn,lgd = bcn.aeTrain(trnOH,f'{pfix}',vb=1)

    t1 = time() - t0
    print(f'Time: {t1:.1f} sec')

#%%
    #bcn.loadw(fn)

    xp = bcn.aen.predict(tstOH)

    bcn.evaluate(tstOH)

    smt.getScore(tstOH,xp,True)

    fn = getFileList(lgd)

    fn = fn[0]

    eacc = EventAccumulator(lgd+'/'+fn)

    eacc.Reload()

    #print(eacc.Tags())
#%%
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

