# Copyright: (c) 2021, Edwin G. W. Peters

from protocol.protocolBase import *
from protocol.benchmark.bench_base import *
from lib.gmskmod import gmskMod


import numpy as np
from scipy import signal



MASKLEN = 16*8
FLAGLEN = 8*2
PACKETLEN = 1000

class Bench_GMSK(Bench_base):
    """
    A class to benchmark GMSK performance
    
    Is supposed to receive a known fixed length signal with known preamble. Compares the signals bit for bit and logs the BER for performance analysis
    """
    name = 'bench_GMSK'

    packetEndDetectMode = PacketEndDetect.FIXED
    packetLen = PACKETLEN

    numBitsOverlap = MASKLEN*2 # definitely has to be longer than the mask
    
    SUM_ALL_MASKS_PYTHON = True

    def _weight_filters(self,filters):
        
        weight = signal.get_window('hamming',len(filters[0]))
        # weight /= np.sum(weight)
        for i in range(len(filters)):
            filters[i] *= weight


    def get_filter(self,Nfft,spSym,maskSize):
        """
        Get the GPU filters
        """

        masks = self._get_xcorrMasks(maskSize)

        filt_template = []
        for m in masks:
            tmp, phase, f_len = gmskMod(m,spSym)
            filt_template.append(tmp[f_len//2:-f_len//2+1])

        self._weight_filters(filt_template)
                
        filtersPadded = np.empty((len(filt_template),Nfft),dtype=np.complex64)

        for i,f in enumerate(filt_template):
            filtersPadded[i] = np.conj(np.fft.fft(f,Nfft)).astype(np.complex64)



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

        return bitLUT, []
 

