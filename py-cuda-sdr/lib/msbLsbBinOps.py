# Copyright: (c) 2021, Edwin G. W. Peters

import numpy as np


def hexToBitStr(hexStr):
    """
    provide the hex string in format 'ab ab ab....
    Returns the bits in string format, as list of bytes
    """

    b = [int(hexStr[i*3:i*3+2],16) for i in range(int((len(hexStr)+1)/3))]

    bytesequence = ['{0:08b}'.format(g) for g in b]
    return bytesequence

def hexToBytes(hexStr):
    """
    Provide hex sting in format 'ab ab ab...'
    Returns the byte values
    """
    
    bInt = [int(hexStr[i*3:i*3+2],16) for i in range(int((len(hexStr)+1)/3))]
    return bInt

def hexToBits(hexStr):
    bInt = [int(hexStr[i*3:i*3+2],16) for i in range(int((len(hexStr)+1)/3))]
    byteSequence = np.array([[int(x) for x in bin(b)[2:].zfill(8)] for b in bInt])
    byteSequence = np.fliplr(byteSequence)
    return byteSequence.reshape(np.prod(byteSequence.shape))


def bitsToBytes(bitArray):
    if len(bitArray) % 8 != 0:
        raise IndexError('Input array length must be a multiple of 8')
    numBytes = int(len(bitArray)/8)

    byteData = np.dot(bitArray.reshape(numBytes,8),2**np.arange(0,8,1)).astype(np.uint8)
    return byteData

def bitsToHex(bits):
        if len(bits) % 8 != 0:
            raise IndexError('Dimension mismatch','Dimension needs to be multiple of 8')

        Decvals = np.dot(bits.reshape(int(len(bits)/8),8),2**np.arange(0,8,1)).astype(np.uint8)
        hexVals = ' '.join(['{:02X}'.format(i) for i in Decvals])
        return hexVals

def bytesToHex(byteArr):

        hexVals = ' '.join(['{:02X}'.format(i) for i in byteArr])
        return hexVals

