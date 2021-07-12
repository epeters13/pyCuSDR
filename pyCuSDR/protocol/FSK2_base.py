# Copyright: (c) 2021, Edwin G. W. Peters# Copyright: (c) 2021, Edwin G. W. Peters

from protocol.protocolBase import *
from scipy import signal

import numpy as np

class FSK2(ProtocolBase):

    name = 'FSK2 Basee'
    
    """
    This subclass overrides the filter of ProtocolBase with a FSK2 filter
    """


    def get_filter(self,Nfft,spSym,maskSize,nCycles=0.5):
        """
        Create and fft the masks used for the cross correlations
        FSK filter.
        +2*pi*nCycles radians/symbol for 1
        -2*pi*nCycles radians/symbol for 0

        nCycles = 0.5 corresponds to baud/2 in spacing
        nCycles = 0.25 corresponds to msk spacing
        """

        wavePhase = np.linspace(1/spSym,1,spSym)*np.pi*2*nCycles

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
    

