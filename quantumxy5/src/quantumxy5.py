import numpy as np
from matplotlib import pyplot as plt
import itertools as it
from numpy import linalg as LA
import bitarray as bt
import functools
import copy




def get_bin(x,n):
    return format(x,'b').zfill(n)




def Basis(N):  #returns string of bits
    basisList = []
    for i in range(0,2**N):
        basisList.append(get_bin(i,N))
    return basisList




def BasisVector(N):   #returns array of bits
    aN=[]
    ss=Basis(N)
    for i in range(0,2**N):
        aN.append(bt.bitarray(ss[i]))
        aN2 = aN[::-1]
    return aN2




def Hamiltonian1(N,J,h):
    BS = BasisVector(N)
    M = len(BS)
    H0 = np.zeros((M,M))
    # diagonal elements
    for i in range(0,M):
        #a = BS[i]
        for j in range(0,N):
            if j<N-1:
                a = copy.copy(BS[i])
                if a[j]!=a[j+1]:
                    a[j],a[j+1]=a[j+1],a[j]
                    H0[i][BS.index(a)]+=J
                elif a[j]==a[j+1]:
                    H0[i][BS.index(a)]=0
            elif j == N-1:
                a = copy.copy(BS[i])
                if a[j]!=a[0]:
                    a[j],a[0]=a[0],a[j]
                    H0[i][BS.index(a)]+=J
                elif a[j]==a[0]:
                    H0[i][BS.index(a)]=0

    HB = np.zeros((M,M))
    for i in range(0,M):
        HB[i][i]+= h*(BS[i].count(1)-BS[i].count(0))
    return -1*(H0+HB)


def rotate(l, n):
    return l[n:] + l[:n]


def Hamiltonian2(N, J, h):
    Id = np.array([[1,0], [0,1]])
    Sp = np.array([[0,1], [0,0]])
    Sm = np.array([[0,0], [1,0]])
    Sz = np.array([[1,0],[0,-1]])
    
    # vector of operators: [σᶻ, σᶻ, id, ...]
    fst_term_ops = [Id]*N
    fst_term_ops[0] = Sp
    fst_term_ops[1] = Sm
    
    fst_term_conj = [Id]*N
    fst_term_conj[0] = Sm
    fst_term_conj[1] = Sp
    
    # vector of operators: [σˣ, id, ...]
    snd_term_ops = [Id]*N
    snd_term_ops[0] = Sz
    
    H0 = np.zeros((2**N, 2**N))
    foldl = lambda func,acc:functools.reduce(func,acc)
    
    for counter in range(0,N):
        H0-=J*foldl(np.kron,fst_term_ops)
        fst_term_ops=rotate(fst_term_ops,1)
    
    for counter in range(0,N):
        H0-=J*foldl(np.kron,fst_term_conj)
        fst_term_conj=rotate(fst_term_conj,1)

    #magnetic field    
    for counter in range(0,N):
        H0-=h*foldl(np.kron,snd_term_ops)
        snd_term_ops=rotate(snd_term_ops,1)
    
    return H0





def magnetisation(state,basis):
    M=0
    for (i, bstate) in enumerate(basis):
        bstate_M=0
        for spin in bstate:
            bstate_M += ((state[i])**2 * (1 if spin else -1))/len(bstate)
        M+=bstate_M
    return M



def correlation(state, basis, r):
    CC = 0
    for (i, bstate) in enumerate(basis):
        bstate_CC = 0
        for (s,s1) in zip(bstate, rotate(bstate,-r)):
            bstate_CC += (state[i]**2 * (1 if s else -1) * (1 if s1 else -1))/len(bstate)
        CC += bstate_CC
        
    MM = magnetisation(state,basis)    
    
    return CC-(MM*MM)






