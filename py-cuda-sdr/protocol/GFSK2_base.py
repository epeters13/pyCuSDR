# Copyright: (c) 2021, Edwin G. W. Peters

from protocol.protocolBase import *
from lib.filters import gaussianFilter
from scipy import signal

import numpy as np


BT = 1. # bandwith time product

class GFSK2(ProtocolBase):

    name = 'GFSK2 Base'
    
    """
    This subclass overrides the filter of ProtocolBase with a GFSK2 filter
    """



    def get_filter(self,Nfft,spSym,maskSize):

        masks = self._get_xcorrMasks(maskSize)


        filt = gaussianFilter(1,BT,spSym,4*spSym)*np.pi/spSym # half a period/symbol

        
        f_len = len(filt)
        

        filt_template = []
        for m in masks:
            phase = np.convolve(np.repeat(m*2-1,spSym),filt)
            tmp = np.exp(1j*np.cumsum(phase))

            filt_template.append(tmp[f_len//2:-f_len//2+1])

        self._weight_filters(filt_template)
        
        filtersPadded = np.empty((len(filt_template),Nfft),dtype=np.complex64)

        for i,f in enumerate(filt_template):
            filtersPadded[i] = np.conj(np.fft.fft(f,Nfft)).astype(np.complex64)



        return filtersPadded.shape[0], filtersPadded
    
    

    
    def _weight_filters(self,filters):
        
        
        weight = signal.get_window('hamming',len(filters[0]))
        # weight /= np.sum(weight)
        for i in range(len(filters)):
            filters[i] *= weight


