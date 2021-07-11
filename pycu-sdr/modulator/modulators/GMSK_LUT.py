# Copyright: (c) 2021, Edwin G. W. Peters

from modulator.modulators.baseLUT import *
from lib.filters import gaussianFilter

class GMSKmod(BaseLUT):

    name = 'GMSK'
    
    def __init__(self,protocol,confRadio):
        """
        Create a LUT for GMSK modulation based on a Gaussian filter.

        First the bits are interpolated and pulse-coded. This corresponds to inserting zeros between the bits and convolve it with a square filter.
        Then, the bits are run through the Gaussian filter.
        The Gaussian filter covers bits: 1 before and 1 after the center bit. Therefore, the templates are made of bit sequences of length 3
        The pulse that we are interested in is the middle one. This gets scaled to pi/2 
        The waveform is restored by integrating the LUT and taking the complex exponential of this
        """

        self.spSym = spSym = confRadio['samplesPerSym'] # samples per symbol

        ## prepare the LUT
        bw = 0.5 # for gaussian filter -- default
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
            pulseShapes[i] = pulse*np.pi/2/spSym # normalize pulses. LUT contains the gradient of the phase
        
        self.LUT = pulseShapes.astype(np.float)

        # for modulation
        self.BToD = np.array([4,2,1]) # to convert binary to decimal for indexing
        self.LUTidx = np.array([-1, 0 ,1])[:,np.newaxis] # Indexing, to make a new axis, such that adding is allowed

        

    def modulate(self,bitData,symTransLUTDopp):
        """
        GMSK modulation
        The LUT is scaled in the modulater with the appropriate Doppler and frequency offsets
        """
        
        # Special care needs to be taken when doing the first and last symbol
        idxStart = np.dot(np.array([2,1]), bitData[:2])
        idxEnd =  np.dot(np.array([4,2]), bitData[-2:])

        idxTab = np.arange(1,len(bitData)-1) + self.LUTidx # makes a 3 X len(bitData) matrix with indices
        idx = np.r_[idxStart,np.dot(self.BToD,bitData[idxTab]),idxEnd] # get LUT index
        outPhase = symTransLUTDopp[idx] # gradient of the phase
        
        log.debug("bitData: {}".format(bitData))

        outPhase = np.cumsum(np.reshape(outPhase,np.prod(outPhase.shape))) # reshape and integrate the phase
        outPhaseWrap = np.mod(outPhase,2*np.pi)

        log.debug('length output data: {}'.format(len(outPhaseWrap)))

        return np.exp(1j*outPhaseWrap)
        
