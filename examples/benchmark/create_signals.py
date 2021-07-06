# Copyright: (c) 2021, Edwin G. W. Peters

import numpy as np

import sys
sys.path.append('../../py-cuda-sdr')
from lib import *


packetData = lambda:  createBitSequence(10000,seed=123) # gives us a default 10000 bit length packet
zeropad = lambda sig,n: np.concatenate((np.zeros(n),sig,np.zeros(n))) # pre and post pad signal with zeros

def createBitSequence(n_bits, seed=None):
    """
    Returns a random sequence of bits. Can be provided with a seed, which returns a deterministic random sequence of bits. The random number generator state is preserved and restored after generation of the bit sequence
    """
    if seed:
        cur_state = np.random.get_state() # store the random gen state
        np.random.seed(seed)

    bitData = np.random.randint(0,2,n_bits)

    if seed:
        np.random.set_state(cur_state) # restore random gen state

    return bitData


def encodeNRZS(bitData):
    """
    NRZ-S encode the binary data
    """
    outData = np.zeros(len(bitData),dtype=np.uint8)

    outData[0]=bitData[0]
    for i, bit in enumerate(bitData[1:],1):
        if bit == 1:
            outData[i] = outData[i-1]
        else:
            outData[i] = ~outData[i-1] & 1

    return outData

def modulateBPSK(raw_bits,samples_pr_sym):
    """
    BPSK modulate the raw bits with NRZ-S encoding to avoid phase ambiguity
    The NRZ-s coding leaves the first bit ambiguous. Therefore, a few extra bits are inserted in front of the sequence 
    """

    
    bits_nrzs = encodeNRZS(np.concatenate(([1,0,1],raw_bits))).astype(float)*2-1

    # filt = rrcosfilter(0.25,2,samples_pr_sym) # root raised cosine filter spanning 2 symbols
    filt = rrcosfilter(0.5,6,samples_pr_sym) # root raised cosine filter spanning 2 symbols
    filt = filt/np.sum(filt)
    
    sig = np.convolve(filt,np.repeat(bits_nrzs,samples_pr_sym)).astype(np.complex64)

    return sig



def modulateFSK(raw_bits,samples_pr_sym):
    """
    FSK modulate the raw bits with NRZ-S encoding to avoid phase ambiguity
    """
    
    wavePhase = np.ones(samples_pr_sym)/samples_pr_sym*np.pi
    
    LUT = np.array([-wavePhase,wavePhase])

    outPhase = np.cumsum(LUT[raw_bits]) - (raw_bits[0]*2-1)*np.pi/2
    outPhaseWrap = np.mod(outPhase,2*np.pi)


    # outPhaseWrapPad = np.r_[VAL*np.ones(WRAPLEN*self.spSym),outPhaseWrap,VAL*np.ones(WRAPLEN*self.spSym)]
        
    return np.exp(1j*outPhaseWrap).astype(np.complex64)


    return sig

def modulateGFSK2(raw_bits,samples_pr_sym):
    """
    GFSK2 modulation of raw bits. without NRZ-S encoding, since there is no phase ambiguity
    
    """

    gausFilt = gaussianFilter(1,1,samples_pr_sym,4*samples_pr_sym)


    phase = np.convolve(gausFilt,np.repeat(raw_bits*2-1,samples_pr_sym))

    sig = np.exp(1j*np.cumsum(phase)/samples_pr_sym*np.pi).astype(np.complex64)

    return sig


def modulateGMSK(raw_bits,samples_pr_sym):
    """
    GMSK modulation of raw bits. without NRZ-S encoding, since there is no phase ambiguity
    
    """

    gausFilt = gaussianFilter(1,0.5,samples_pr_sym,4*samples_pr_sym)


    phase = np.convolve(gausFilt,np.repeat(raw_bits*2-1,samples_pr_sym))

    sig = np.exp(1j*np.cumsum(phase)/samples_pr_sym*np.pi/2).astype(np.complex64)

    return sig


def awgn(sig,snr,measured = True):
    """
    Sends signal sig through AWGN channel

    Inputs:
       sig -- modulated signal
       noisePwr -- noise power in dB
       measured -- True if noise power is scaled to signal power. Scaled to unity otherwise (default True)

    Returns:
       sigN -- signal plus noise
       SNR -- actual snr
    """

    if measured:
        # Normalized SNR
        sigp = 10*np.log10(np.linalg.norm(np.abs(sig),2)**2/len(sig))
        snr = snr - sigp
        noiseP = 10**(-snr/10)
    else:
        # Assuming unity signal power
        noiseP = 10**(-snr/10)

    if np.iscomplexobj(sig):
        return sig + np.sqrt(noiseP/2) * (np.random.randn(len(sig)) + 1j*np.random.randn(len(sig)))
    else:
        return sig + np.sqrt(noiseP) * np.random.randn(len(sig))
        


def get_GMSK_packet(spSym = 16, fs = 9600*16, offset_freq = None):
    """
    returns standard GMSK packet
    
    """
    if offset_freq is None:
        offset_freq = fs/4
        
    raw_bits = packetData()
    sig_GMSK = modulateGMSK(raw_bits,spSym)
    sig_GMSK_full = zeropad(sig_GMSK,10000)
    sig_GMSK_full*= np.exp(1j*2*np.pi*offset_freq/fs*np.arange(len(sig_GMSK_full))) 


    return sig_GMSK_full, raw_bits


def get_BPSK_packet(spSym = 16, fs = 9600*16, offset_freq = None):
    """
    returns standard BPSK packet
    
    """
    if offset_freq is None:
        offset_freq = fs/4
        
    raw_bits = packetData()
    sig = modulateBPSK(raw_bits,spSym)
    sig_full = zeropad(sig,10000)
    sig_full*= np.exp(1j*2*np.pi*offset_freq/fs*np.arange(len(sig_full))) 


    return sig_full, raw_bits


def get_padded_packet(modulation, spSym = 16, fs = 9600*16, offset_freq = None, raw_bits = []):

    if offset_freq is None:
        offset_freq = fs/4

    if len(raw_bits) == 0:
        raw_bits = packetData()
    if modulation == 'BPSK':
        sig = modulateBPSK(raw_bits,spSym)
    elif modulation == 'GMSK':
        sig = modulateGMSK(raw_bits,spSym)
    elif modulation == 'FSK':
        sig = modulateFSK(raw_bits,spSym)
    elif modulation == 'GFSK':
        sig = modulateGFSK2(raw_bits,spSym)
    else:
        raise TypeError('Only supports GMSK, FSK and BPSK')
        
    sig_full = zeropad(sig,10000)
    sig_full*= np.exp(1j*2*np.pi*offset_freq/fs*np.arange(len(sig_full))) 


    return sig_full, raw_bits

    
if __name__ == "__main__":


    spSym = 16
    fs = 9600*16
    
    raw_bits = packetData()

    sig_BPSK = modulateBPSK(raw_bits,spSym)

    sig_GMSK = modulateGMSK(raw_bits,spSym)

    SNR = 10
    bw_gmsk = 9.6e3/0.7
    bw_bpsk = 9.6e3*2
    
    SNR_r = SNR + 10*np.log10(bw_gmsk/fs) # for generating AWGN, the bandwidth and oversampling rate need to be taken into account

    # zero pad each signal to make it behave as a packet

    sig_BPSK_full = zeropad(sig_BPSK,10000)
    sig_GMSK_full = zeropad(sig_GMSK,10000)
    
    offset_freq = fs/4
    var_noise = 0.01
    
    sig_BPSK_full *= np.exp(1j*2*np.pi*offset_freq/fs*np.arange(len(sig_BPSK_full))) 
    sig_GMSK_full *= np.exp(1j*2*np.pi*offset_freq/fs*np.arange(len(sig_GMSK_full))) 

    sig_BPSK_full_n = awgn(sig_BPSK_full,SNR_r)
    sig_GMSK_full_n =  awgn(sig_GMSK_full,SNR_r)
    sig_GMSK_short_shift = awgn(sig_GMSK * np.exp(1j*2*np.pi*offset_freq/fs*np.arange(len(sig_GMSK))) ,SNR_r)
    
    
    # longer GMSK signel
    # sig_GMSK_short_shift = sig_GMSK * np.exp(1j*2*np.pi*offset_freq/fs*np.arange(len(sig_GMSK))) 
    sig_GMSK_long = np.concatenate((np.tile(sig_GMSK_full_n,10), sig_GMSK_short_shift, sig_GMSK_short_shift, sig_GMSK_short_shift, np.tile(sig_GMSK_full_n,10))) # contains 23 packets (3 in row and 20 with a delay)

    sig_GMSK_very_long = np.tile(sig_GMSK_long,10) # contains 2300 packets


    sig_BPSK_short_shift = sig_BPSK * np.exp(1j*2*np.pi*offset_freq/fs*np.arange(len(sig_BPSK))) 
    sig_BPSK_long = np.concatenate((np.tile(sig_BPSK_full_n,10), sig_BPSK_short_shift, sig_BPSK_short_shift, sig_BPSK_short_shift, np.tile(sig_BPSK_full_n,10))) # contains 23 packets (3 in row and 20 with a delay)

    sig_BPSK_very_long = np.tile(sig_BPSK_long,10) # contains 2300 packets


    # add enough zeros at the end to ensure that adfags modem empties the buffer
    np.save('sig_BPSK.bin',np.concatenate((sig_BPSK_very_long,np.zeros(int(2**17)))).astype(np.complex64))
    np.save('sig_GMSK.bin',np.concatenate((sig_GMSK_very_long,np.zeros(int(2**17)))).astype(np.complex64))
