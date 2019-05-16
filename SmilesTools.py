# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 13:03:42 2018

@author: Steve O'Hagan
"""

import numpy as np
import pandas as pd

from rdkit import Chem

from sys import getsizeof

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def sizeof(obj):
    m = getsizeof(obj)
    return sizeof_fmt(m)

class smiUtil:

    @staticmethod
    def findUniqueCodes(dat,cname='Molecule'):
        smiCodes=set(' ')
        smiLen = max([len(s) for s in dat[cname]])
        for s in dat[cname]:
            smiCodes.update(list(s))
        smiCodes = sorted(list(smiCodes))
        return smiCodes, smiLen

    def __init__(self,codes=' #()+-/1234567=@BCFHINOS[\\]clnoprs',maxLen=125):
        self.smiCodes = sorted(list(codes))
        self.smiLen = maxLen
        self.codeLen=len(self.smiCodes)
        self.code2int = dict((c,i) for i,c in enumerate(self.smiCodes))
        self.int2code = dict((i,c) for i,c in enumerate(self.smiCodes))
        self.vfunc = np.vectorize(lambda x: self.int2code[x])
        self.trnDat = None
        self.tstDat = None
        self.trnOH = None
        self.tstOH = None

    def filterGood(self,dat,cname='Molecule'):
        chkd = dat.copy()
        validCodes = set(self.smiCodes)
        flagList=[]
        for idx, p in chkd.iterrows():
            n = len(p[cname])
            inP = set(list(p[cname]))
            if (n<self.smiLen) and (inP <= validCodes):
                flagList.append('OK')
            else:
                flagList.append('bad')
        chkd['flag'] = flagList
        idx = chkd[chkd['flag']=='bad'].index
        chkd.drop(idx,inplace=True)
        chkd.drop(columns=['flag'],inplace=True)
        chkd.reset_index(inplace=True,drop=True)
        return chkd
    
    def smi2OH(self,dat,cname='Molecule'):
        rowCount,_= np.shape(dat)
        xs=np.zeros((rowCount,self.smiLen,self.codeLen),dtype=np.int8)
        for i,p in dat.iterrows():
            inP=list(p[cname].ljust(self.smiLen))
            for j,c in enumerate(inP):
                xs[i,j,self.code2int[c]] = 1.0
        return xs

    def oh2smi(self,x):
        zc = x.argmax(axis=-1)
        cz = self.vfunc(zc)
        cc = ["".join(row) for row in cz]
        return cc

    def getScore(self,tstOH,predOH,verbose=False):
        nr = len(tstOH)
        c1 = self.oh2smi(tstOH)
        c2 = self.oh2smi(predOH)
        perfect = 0
        good = 0.0
        sm = 0.0
        for s1,s2 in zip(c1,c2):
            s1=s1.strip()
            s2=s2.strip()
            mx=max(len(s1),len(s2))
            s1=s1.ljust(mx)
            s2=s2.ljust(mx)
            hit=sum([x==y for x,y in zip(s1,s2)])
            hit = hit / mx
            if hit >= 1.0:
                perfect+=1
            if self.isGood(s2):
                good+=1
            sm += 100*hit
            if verbose:
                print(s1)
                print(s2)
        p=100*perfect/nr
        g=100*good/nr
        f=sm/nr
        print(f'Perfect: {p:.2f} Good: {g:.2f} Match: {f:.2f}')
        return p,g,f

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

    @staticmethod
    def getDataFromCSV(dfn = 'data/6MSmiles.csv',trnSz=25000,tstSz=2000,shuffle=True,codesFromData=False):
        dat = pd.read_csv(dfn)
        n = len(dat)
        assert n >= trnSz+tstSz

        if codesFromData:
            cds = smiUtil.findUniqueCodes(dat)
            smt = smiUtil(codes=cds)
        else:
            smt = smiUtil()

        if shuffle:
            dat = dat.sample(frac=1).reset_index(drop=True)

        smt.trnDat = dat[0:trnSz]

        smt.tstDat = dat[-tstSz:]
        smt.tstDat = smt.tstDat.reset_index(drop=True)

        smt.trnOH = smt.smi2OH(smt.trnDat)
        smt.tstOH = smt.smi2OH(smt.tstDat)

        return smt


#%%
if __name__ == "__main__":
    pass






