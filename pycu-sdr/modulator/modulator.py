# Copyright: (c) 2021, Edwin G. W. Peters

import sys
sys.path.append('../')
from __global__ import *
import numpy as np
import logging
import time
#import cProfile
import modulator.encoders.encoder_base
import modulator.modulators.baseLUT
#from profilehooks import profile

"""
For some reason the GRC and the B210 behave unreliably when the signal consists of less than 16384 samples. Therefore any signal that is shorter gets zero padded before the packet.
The 16384 seems to be independent of the USRP send_buffer_size, which currently is set to 1024
TODO: maybe burst tags help here
"""

log	= logging.getLogger(LOG_NAME+'.'+__name__)
log.level = logging.WARNING

STORE_BITS_IN_FILE = False
BITS_FNAME = '/dev/null'


SIG_MIN_LENGTH = 16384 # 1024 * 16

NOISE_LEN = 4096
# NOISE_LEN = 4096*10
NOISE_VAR = 0.00000001 # very small


class Modulator():
    """
    Flexible modulator class that does:
         framing/encoding    (OSI lvl 2)
         modulation          (OSI lvl 1)
    
    The framing is specified by the protocol, and the framer is located in framers/
    
    The modulation is done using a LUT, which is located in LUTs. The LUT is chosen in the protocol
    """


    def __init__(self,conf ,confRadio, protocol):
        self.conf = conf
        self.confRadio = confRadio

        self.protocol = protocol

        # setup the framer
        modulator.encoders.encoder_base.log.level = log.level
        modulator.modulators.baseLUT.log.level = log.level
        encoderFun = protocol.getFramer(confRadio) # the protocol provides the framer function
        self.encoder = encoderFun(protocol,confRadio)


        modulatorFun = protocol.getModulator(confRadio) # the protocol provides the modulator
        self.modulatorCLS = modulatorFun(protocol,confRadio)

        log.info(f'Using modulator {self.modulatorCLS.name} with encoder {self.encoder.name}')
        
        # local needed variables
        self._spSym = confRadio['samplesPerSym']
        log.debug('Tx configured [{0}] samples/symbol'.format(self._spSym))
        self.Fc = confRadio['frequency_Hz']
        log.debug('Tx configured Fc [{0}] Hz'.format(self.Fc))
        self._TxFreqOffset = confRadio['frequencyOffset_Hz']
        log.debug('Tx configured TxFreqOffset [{0}] Hz'.format(self._TxFreqOffset))
        self._TxCentreFreqOffset = confRadio.get('centreFrequencyOffset',0.) # This can be configured externally over RPC
        log.debug('Tx configured TxCentreFreqOffset [{0}] Hz'.format(self._TxCentreFreqOffset))
        self.baudRate = confRadio['baud']
        log.debug('Tx configured baudRate [{0}]'.format(self.baudRate))
    

        self.noise = NOISE_VAR*(np.random.randn(SIG_MIN_LENGTH)+1j*np.random.randn(SIG_MIN_LENGTH)).astype(np.complex64)
        # local hidden variables (set by setters)
        self._rangeRate = 0

    #@profile(immediate=True)
    def encodeAndModulate(self,byteMessage):
        """
        Does the encoding and modulation
        """
        t = time.time()
        framedData = self.encoder.encodeAndFrame(byteMessage)
        print(f'Frame time {1000*(time.time()-t):.3f} ms')
        modSignal = self.modulate(framedData)

        return modSignal


    def encodeAndFrame(self,byteMessage):
        return self.encoder.encodeAndFrame(byteMessage)

    def modulate(self,bitData):
        """
        Apply the Doppler and frequency offsets to the LUT and let the modulator modulate the signal using this scaled LUT
        """

        # get the doppler based on the current rangerate
        dopplerCoef = self.getDoppler()/self.baudRate/self.spSym # Doppler phase increment

        # compensate for frequency offset in Tx and in the usrp
        freqOffset = self.TxFreqOffsetRads/self.baudRate/self.spSym # Fs = baud*spSym
        centreFreqOffset = self.TxCentreFreqOffsetRads/self.baudRate/self.spSym
        offsetCoef = freqOffset+centreFreqOffset # fixed frequency offset

        # doppler compensate the LUT
        symTransLUTDopp = self.modulatorCLS.LUT + dopplerCoef + offsetCoef
        # modulate
        t= time.time()
        txSig = self.modulatorCLS.modulate(bitData,symTransLUTDopp)
        print(f'Modulation time {1000*(time.time()-t):.3f} ms ')
        # The USRP requires around 4096 samples initially to stabilize the signal
        txSig = np.concatenate((self.noise[:NOISE_LEN],txSig))
        txSig = np.concatenate((txSig,self.noise[:NOISE_LEN]))
        
        # when a packet is shorter than 16384 samples (1024 * 16), the behaviour is unpredictable, and not necessarily all of the signal gets transmitted. Pre-pad shorter signals
        if len(txSig) < SIG_MIN_LENGTH:
            Lsig = len(txSig)
            txSig = np.concatenate((self.noise[:SIG_MIN_LENGTH-Lsig],txSig))

        
        if STORE_BITS_IN_FILE is True:
            np.save(BITS_FNAME,txSig)

        return txSig.astype(MODULATORDTYPE)





    ######
    #
    # Setters and getters to configure the modulator
    #
    ######

    def get_rangeRate(self):
        """RPC interface"""
        return self._rangeRate

    def set_rangeRate(self,rangeRate):
        """RPC interface"""
        self._rangeRate = rangeRate

    def getDoppler(self):
        # in rad/second
        # to remove doppler multiply with np.exp(-1j*self.getDoppler/baud/spSym)
        return self._rangeRate/3e8*self.Fc*2*np.pi


    # for RPC interface
    def get_samp_rate(self):
        """RPC interface"""
        return self.baudRate * self._spSym

    def set_samp_rate(self,samp_rate):
        """RPC interface"""
        log.warning('Setting sample rate should be done through the config')

    def get_Tx_Fc(self):
        """RPC interface"""
        return self._Fc
        
    def set_Tx_Fc(self,Fc):
        """RPC interface"""
        self._Fc = Fc
        
    
        
    @property
    def spSym(self):
        return self._spSym

    @spSym.setter
    def spSym(self,spSym):
        self._spSym = spSym
        self.TxFreqOffset = spSym*self.baudRate/4 ##  TODO: Should this be configurable?
    

    @property
    def TxTotalFreqOffset(self):
        """Returns centre frequency offset + doppler"""
        return self._TxFreqOffset + self._TxCentreFreqOffset + self._rangeRate/3e8*self.Fc
        
    @property
    def TxFreqOffset(self):
        return self._TxFreqOffset

    @TxFreqOffset.setter
    def TxFreqOffset(self,fo):
        self._TxFreqOffset = fo

    @property
    def TxFreqOffsetRads(self):
        return self._TxFreqOffset*2*np.pi

    @property
    def TxCentreFreqOffset(self):
        return self._TxCentreFreqOffset

    @TxCentreFreqOffset.setter
    def TxCentreFreqOffset(self,offset):
        self._TxCentreFreqOffset = offset


    @property
    def TxCentreFreqOffsetRads(self):
        return self._TxCentreFreqOffset*2*np.pi
    
