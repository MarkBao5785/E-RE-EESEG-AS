import numpy as np
import argparse
import pickle
import os
import time
import torch
import pandas as pd 
import scipy as sp
import torch.nn as nn
import torch.optim as optim
import random
from collections import defaultdict
import scipy.io as sio
import scipy.sparse as spp
import scipy as sp


if torch.cuda.is_available():  
    dev = "cuda:1" 
else:  
    dev = "cpu" 
device = torch.device(dev)

class Linearucb:
    # Brute-force Linear TS with full inverse
    def __init__(self, dim, lamdba=0.001, nu=1, style='ts'):
        self.dim = dim
        self.U = lamdba * np.eye(dim)
        self.Uinv = 1 / lamdba * np.eye(dim)
        self.nu = nu
        self.jr = np.zeros((dim, ))
        self.mu = np.zeros((dim, ))
        self.lamdba = lamdba
        self.style = style

    def select(self, context):
        sig = np.diag(np.matmul(np.matmul(context, self.Uinv), context.T))
        r = np.dot(context, self.mu) + np.sqrt(self.lamdba * self.nu) * sig
        return np.argmax(r)
        
    
    def train(self, context, reward):
        self.jr += reward * context
        self.U += np.matmul(context.reshape((-1, 1)), context.reshape((1, -1)))
        # fast inverse for symmetric matrix
        zz , _ = sp.linalg.lapack.dpotrf(self.U, False, False)
        Linv, _ = sp.linalg.lapack.dpotri(zz)
        self.Uinv = np.triu(Linv) + np.triu(Linv, k=1).T
        self.mu = np.dot(self.Uinv, self.jr)
        return 0
    

    
    
    