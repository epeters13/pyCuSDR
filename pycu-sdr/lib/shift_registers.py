# Copyright: (c) 2021, Edwin G. W. Peters

import numpy as np


def SSRG(L,fbtaps):
    """
    seq = SSRG(L,fbtaps) returns the binary code produced by the simple shift register generator (SSRG) fbtaps

    input:
    \tL -- number of registers. Total sequence length is 2**L-1
    \tfbtaps -- the indices feedback taps, where the first tap is indexed 0

    returns:
    \tseq -- the SSRG sequence generated
    """
    
    fbtaps = [t-1 for t in fbtaps] # this allows us to use the "proper" indexing as the normal definition in the SSRG filters
    taps = np.ones(L,dtype=np.int) 
    seq = np.empty(int(2**L)-1,dtype=np.int) 

    for i in range(int(2**L)-1):
        tmp = taps[0]
        taps[0] = np.sum(taps[fbtaps]) % 2
        taps[2:] = taps[1:-1]
        taps[1] = tmp
        seq[i] = taps[-1]

    return seq


def barkerCode(L,codeIdx = 0):
    """
    seq = barkerCode(L,codeIdx) 
    Returns the length L Barker code. If more than 1 code exists for the length, codeIdx selects the code

    input:
    \tL -- length of sequence
    \tcodeIdx -- index of code if more than one exists (only for L=[2,4]) [default 0]

    output:
    \tseq -- length L Barker code sequence


    Available code lengths:
    [2, 3, 4, 5, 7, 11, 13]
    
    """

    availableCodeLengths = [2, 3, 4, 5, 7, 11, 13]
    
    if L == 2:
        if codeIdx == 2:
            return np.array([1,-1])
        return np.array([1,1])
    elif L == 3:
        return np.array([1,1,-1])
    elif L == 4:
        if codeIdx == 1:
            return np.array([1,1,1,-1])
        return np.array([1,1,-1,1])
    elif L == 5:
        return np.array([1,1,1,-1, 1])
    elif L == 7:
        return np.array([1,1,1,-1,-1,1,-1])
    elif L == 11:
        return np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1])
    elif L == 13:
        return np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1])
    else:
        raise IndexError(f'Barker code of length {L} not found. Barker codes can be of lengths {availableCodeLengths}')


    

def PN9(num_codes=300,initial_value = np.ones(9)):
    """
    PN9 shift register with polynomial x^9 + x^5 + 1
    The register is clocked for every bit, but XORed with the byte every 8 clocks, thus every 8th sequence is used. 
    Initialized with initial_value
    This routine creates a LUT of num_codes PN9 codes
    """
    pp = np.empty((num_codes*8,9),dtype=np.uint8)
    pp[0,:] = initial_value.astype(np.uint8)

    for i in range(1,pp.shape[0]): 
        pp[i,0:8] = pp[i-1,1:9] 
        pp[i,8] = int(np.logical_xor(pp[i-1,0],pp[i-1,5])) 

    PP9 = np.dot(pp,np.r_[2**np.arange(8),0])
    return PP9[::8]
