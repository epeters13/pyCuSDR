# Copyright: (c) 2021, Edwin G. W. Peters

from protocol.protocolBase import *
from protocol.benchmark.bench_base import *
from lib.filters import rrcosfilter

import numpy as np
from scipy import signal



MASKLEN = 16*8
FLAGLEN = 8*2
PACKETLEN = 1000

def decodeNRZS(NRZdata):
    """
    NRZ-S decoder
    """
    bitData = np.zeros(len(NRZdata),dtype=np.uint8)

    bitData[0]=NRZdata[0]
    bitPrev = NRZdata[0]
    for i, bit in enumerate(NRZdata[1:]):
        bitData[i+1] = 1 if bit == bitPrev else 0
        bitPrev = bit

    return bitData
    

class Bench_BPSK(Bench_base):
    """
    A class to benchmark BPSK performance
    
    Is supposed to receive a known fixed length signal with known preamble. Compares the signals bit for bit and logs the BER for performance analysis
    """
    name = 'bench_BPSK'

    packetEndDetectMode = PacketEndDetect.FIXED
    packetLen = PACKETLEN

    numBitsOverlap = MASKLEN*2 # definitely has to be longer than the mask

    SUM_ALL_MASKS_PYTHON = True


    def get_filter(self,Nfft,spSym,maskSize):
        """
        Get the GPU filters for BPSK modulation
        """

        self.num_masks = int(2**(maskSize-1)) # DIRTY HACK: to load the kernels, this must be known
        masks = self._get_xcorrMasks(maskSize).astype(np.float)*2-1

        # filt = rrcosfilter(0.25,2,spSym)
        filt = rrcosfilter(0.5,6,spSym)
        filt = filt/np.sum(filt)
        f_len = len(filt)
        
        filt_template = []
        for m in masks:
            tmp = np.convolve(np.repeat(m,spSym),filt)
            filt_template.append(tmp[f_len//2:-f_len//2+1])

        log.info(f'filter template len {filt_template[0].shape}')
        filtersPadded = np.empty((len(filt_template),Nfft),dtype=np.complex64)

        for i,f in enumerate(filt_template):
            filtersPadded[i] = np.conj(np.fft.fft(f,Nfft)).astype(np.complex64)



        return filtersPadded.shape[0], filtersPadded
        
            
    def _get_xcorrMasks(self,maskLen):

        rawMasks = np.zeros((2**(maskLen),maskLen))
        for i in range(2**(maskLen)):
            rawMasks[i] = np.array([np.float(i) for i in list(np.binary_repr(i,width=maskLen))])

        return rawMasks

    def get_symbolLUT_new(self,maskLen):
        # Allows lookup on more symbols
        # These masks do the NRZ decoding directly
        if maskLen == 5:
            # return np.concatenate((symLUT,np.flipud(symLUT)),axis=0)
            # use the old one
            return np.array([ # double check the masklen 5 table
                [
                    [0,1,2,3],
                    [4,5,6,7]
                ],   # 0
                [
                    [0,1,2,3],
                    [4,5,6,7]
                ],   # 1
                [
                    [0,1,2,3],
                    [4,5,6,7]
                ],   # 2
                [
                    [0,1,2,3],
                    [4,5,6,7]
                ],   # 3
                [
                    [12,13,14,15],
                    [8,9,10,11]
                ],   # 4
                [
                    [12,13,14,15],
                    [8,9,10,11]
                ],   # 5
                [
                    [12,13,14,15],
                    [8,9,10,11]
                ],   # 6
                [
                    [12,13,14,15],
                    [8,9,10,11]
                ],   # 7
                [
                    [12,13,14,15],
                    [8,9,10,11]
                ],   # 8
                [
                    [12,13,14,15],
                    [8,9,10,11]
                ],   # 9
                [
                    [12,13,14,15],
                    [8,9,10,11]
                ],   # 10
                [
                    [12,13,14,15],
                    [8,9,10,11]
                ],   # 11
                  [
                    [0,1,2,3],
                    [4,5,6,7]
                ],   # 12
                [
                    [0,1,2,3],
                    [4,5,6,7]
                ],   # 13
                [
                    [0,1,2,3],
                    [4,5,6,7]
                ],   # 14
                [
                    [0,1,2,3],
                    [4,5,6,7]
                ],   # 15
              
            ], 
            dtype=np.int)
        if maskLen == 4:
            # return np.concatenate((symLUT,np.flipud(symLUT)),axis=0)
            # use the old one
            return np.array([
                [
                    [0,1],
                    [2,3]
                ],   # 0
                [
                    [0,1],
                    [2,3]
                ],   # 1
                [
                    [6,7],
                    [4,5]
                ],   # 2
                [
                    [6,7],
                    [4,5]
                ],   # 3
                [
                    [6,7],
                    [4,5]
                ],   # 4
                [
                    [6,7],
                    [4,5]
                ],   # 5
                [
                    [0,1],
                    [2,3]
                ],   # 6
                [
                    [0,1],
                    [2,3]
                ]   # 7
            ], 
            dtype=np.int)
        else:
            raise Exception(f"bench_BPSK: Invalid mask length ({maskLen})")

        
    def get_symbolLUT(self,maskLen):

        if maskLen == 5:
            # return np.concatenate((symLUT,np.flipud(symLUT)),axis=0)
            # use the old one
            return np.array([[0,1],   # 0
                             [3,2],   # 1
                             [4,5],   # 2
                             [7,6],   # 3
                             [8,9],   # 4
                             [11,10], # 5
                             [12,13], # 6
                             [15,14], # 7
                             [15,14], # 8
                             [12,13], # 9
                             [11,10], # 10
                             [8,9],   # 11
                             [7,6],   # 12
                             [4,5],   # 13
                             [3,2],   # 14
                             [0,1]],  # 15
                            dtype=np.int)

        elif maskLen == 4:
            return np.array([[0,1],   # 0
                             [3,2],   # 1
                             [4,5],   # 2
                             [7,6],   # 3
                             [7,6],   # 4
                             [4,5],   # 5
                             [3,2],   # 6
                             [0,1]],   # 7
                            dtype=np.int)

        elif maskLen == 3:
            return np.array([[0,1],   # 0
                             [3,2],   # 1
                             [3,2],   # 2
                             [0,1]],  # 3
                            dtype=np.int)
        else:
            raise Exception(f"bench_BPSK: Invalid mask length ({maskLen})")

    ''

    def get_symbolLUT2(self,maskLen):
        """
        This symbol lookup table looks at the centre bit instead of the last one
        It will return:
            a mapping from symbols to bits 
            The old symbol LUT which can be used for trust weighting

        """
        masks =  self._get_xcorrMasks(maskLen)

        sampleIdx = int(maskLen/2)

        bitLUT = masks[:,sampleIdx]

        symbolLUT = self.get_symbolLUT_new(maskLen)
        
        return None, symbolLUT

        
        
        return None, self.get_symbolLUT(maskLen)
        
 

    # def decoderPreprocessor(self,signal,**args):
    #     ## Need to do nrz decoding -- currently done with the masks and special routine in demodulator. TODO: make it in line with the rest
    #     return decodeNRZS(signal)
