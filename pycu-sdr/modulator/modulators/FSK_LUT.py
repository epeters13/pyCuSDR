# Copyright: (c) 2021, Edwin G. W. Peters

from modulator.modulators.baseLUT import *

class FSKmod(BaseLUT):

    name = 'FSK'

    def __init__(self,protocol,confRadio):
        """
        Create a LUT for FSK modulation.

        FSK changes the phase by +f for a 1 and -f for a zero. There is no ISI between the symbols
        The waveform is restored by integrating the LUT and taking the complex exponential of this
        """

        self.spSym = spSym = confRadio['samplesPerSym'] # samples per symbol

        wavePhase = np.ones(spSym)/spSym*2*np.pi*0.5 # 0.5 for baud/2 spacing TODO: get this from protocol or config

        self.LUT = np.array([-wavePhase,wavePhase])

        

    def modulate(self,bitData,symTransLUTDopp):
        """
        FSK modulation
        The LUT is scaled in the modulator with the appropriate Doppler and frequency offsets
        """

        outPhase = np.cumsum(symTransLUTDopp[bitData]) - (bitData[0]*2-1)*np.pi/2
        outPhaseWrap = np.mod(outPhase,2*np.pi)

        log.debug('length output data: {}'.format(len(outPhaseWrap)))


        num_bytes = outPhase.nbytes

        sig = np.exp(1j*outPhaseWrap).astype(np.complex64)
           
        log.debug(f'signal length {len(sig)} samples. number of bytes sent {len(bitData)/8}')
        return sig

