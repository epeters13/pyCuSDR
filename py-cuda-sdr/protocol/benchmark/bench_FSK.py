# Copyright: (c) 2021, Edwin G. W. Peters

from protocol.protocolBase import *
from protocol.benchmark.bench_base import *

import numpy as np
from scipy import signal



MASKLEN = 16*8
FLAGLEN = 8*2
PACKETLEN = 1000

class Bench_FSK(Bench_base):
    """
    A class to benchmark FSK performance
    
    Is supposed to receive a known fixed length signal with known preamble. Compares the signals bit for bit and logs the BER for performance analysis
    """
    name = 'bench_FSK'

    packetEndDetectMode = PacketEndDetect.FIXED
    packetLen = PACKETLEN

    numBitsOverlap = MASKLEN*2 # definitely has to be longer than the mask

    # for FSK we want to sum all masks before doing the Doppler Search
    SUM_ALL_MASKS_PYTHON = True


    def get_filter(self,Nfft,spSym,maskSize):
        """
        Get the GPU filters

        """

        """
        Create and fft the masks used for the cross correlations
        FSK filter.
        +pi radians/symbol for 1
        -pi radians/symbol for 0
        """

        wavePhase = np.linspace(1/spSym,1,spSym)*np.pi

        symbols = self._get_xcorrMasks(maskSize)

        filtersPh = np.empty((len(symbols),len(symbols[0])*spSym))
        for i,p in enumerate(symbols):
            p = p*2-1
            filtersPh[i,0:spSym] = p[0] * wavePhase + -1*p[0]*np.pi/2 
            for j in range(1,len(p)):
                filtersPh[i,j*spSym:(j+1)*spSym] = filtersPh[i,j*spSym-1] + p[j]*wavePhase
                
        filters = [np.exp(1j*f) for f in filtersPh]
        

        filtersPadded = np.empty((len(filters), Nfft),dtype=np.complex64)
        for k in range(len(filters)):
            filtersPadded[k] = np.conj(np.fft.fft(filters[k], Nfft)).astype(np.complex64)

        return filtersPadded.shape[0], filtersPadded
        
            

    def get_symbolLUT2(self,maskLen):
        """
        This symbol lookup table looks at the centre bit instead of the last one
        It will return:
            a mapping from symbols to bits 
            The old symbol LUT which can be used for trust weighting
        """

        masks = self._get_xcorrMasks(maskLen)

        sampleIdx = int(maskLen/2)

        bitLUT = masks[:,sampleIdx]

        # symbolLUT = self.get_symbolLUT(maskLen)
        
        return bitLUT, []
 

