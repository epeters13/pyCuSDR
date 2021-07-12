# Copyright: (c) 2021, Edwin G. W. Peters

from modulator.modulators.baseLUT import *

WRAPLEN = 1000 # symbols on each side
VAL = 0.0000001
SEND_BUFFER_SIZE = 1024 # USRP send buffer N bytes
class GFSK2mod(BaseLUT):

    name = 'GFSK2'

    def __init__(self,protocol,confRadio):
        """
        Create a LUT for GFSK2 modulation.

        GFSK2 changes the phase by +f for a 1 and -f for a zero and passes this through a Gaussian filter.
        The waveform is restored by integrating the LUT and taking the complex exponential of this
        """

        self.spSym = spSym = confRadio['samplesPerSym'] # samples per symbol

        wavePhase = np.ones(spSym)/spSym*np.pi

        ## prepare the LUT
        bw = 1 # for gaussian filter -- default
        gain = 1 # filter gain, unity
        nTaps =  4*spSym # span 4 symbols
            
        gausFilt = gaussianFilter(gain,bw,spSym,nTaps)
        sqFilt = np.ones(spSym)
        filt = np.convolve(gausFilt,sqFilt)
        grpT = np.int(len(filt)/2) # compute filter group delay
        bitSequences = [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]
        interpVect = np.r_[1,np.zeros(spSym-1)]
        bitsInt = [np.kron(np.array(bits)*2-1,interpVect) for bits in bitSequences] # interpolate the NRZ of the bit sequences

        pulseShapes = np.zeros((len(bitSequences),spSym))
        # generate pulse shapes
        for i,bits in enumerate(bitsInt):
            tmpFilt = np.convolve(filt,bits)
            pulse = tmpFilt[grpT+int(spSym/2):grpT+int(1.5*spSym)] # the pulse is the middle bit
            pulseShapes[i] = pulse*np.pi/spSym # normalize pulses. LUT contains the gradient of the phase
        
        self.LUT = pulseShapes.astype(np.float)

        # for modulation
        self.BToD = np.array([4,2,1]) # to convert binary to decimal for indexing
        self.LUTidx = np.array([-1, 0 ,1])[:,np.newaxis] # Indexing, to make a new axis, such that adding is allowed

        

    def modulate(self,bitData,symTransLUTDopp):
        """
        GFSK2 modulation
        The LUT is scaled in the modulator with the appropriate Doppler and frequency offsets
        """
        outPhase = np.cumsum(symTransLUTDopp[bitData]) - (bitData[0]*2-1)*np.pi/2
        outPhaseWrap = np.mod(outPhase,2*np.pi)

        log.debug('length output data: {}'.format(len(outPhaseWrap)))

        outPhaseWrapPad = np.concatenate((VAL*np.ones(WRAPLEN*self.spSym),outPhaseWrap,VAL*np.ones(WRAPLEN*self.spSym)))

        num_bytes = outPhase.nbytes

        rem = np.remainder(num_bytes,SEND_BUFFER_SIZE)

        if rem > 0:
            rem = SEND_BUFFER_SIZE - rem
            outPhaseWrapPad = np.concatenate((outPhaseWrap,np.zeros(rem,dtype = np.complex64)))

        log.info(f'transmit number of blocks {num_bytes/SEND_BUFFER_SIZE}')
        
        
        return np.exp(1j*outPhaseWrapPad).astype(np.complex64)

