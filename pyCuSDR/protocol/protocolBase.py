# Copyright: (c) 2021, Edwin G. W. Peters

import sys
sys.path.append("../")
from __global__ import *
from modulator import encoders, modulators
from enum import Enum

import logging

LOG_NAME = "gpu-modem"
log	= logging.getLogger(LOG_NAME+'.'+__name__)
log.setLevel(logging.INFO)

DEFAULT_NO_SYNC_FLAGS = 2

class PacketEndDetect(Enum):
    FLAGS = 0
    FIXED = 1
    IN_DATA = 2

class PacketLenEndianness(Enum):
    LITTLE = True
    BIG = False


class ProtocolBase():

    name = 'ProtocolBase'

    
    numBitsOverlap = 2*513  # in Decoder the number of bits we let blocks overlap 
    packetEndDetectMode = PacketEndDetect.FLAGS

    # FIXED length variables
    packetLen = None # NA for FLAGS
    
    # IN_DATA length variables
    packetEndLenField = None # NA for FLAGS
    packetEndLenFieldNumBytes = None # NA for FLAGS
    
    packet_sizes = [] # list of allowable packet sizes. 
    
    def __init__(self,**args):
        pass
        


    def get_filter(self,Nfft,spSym=None,maskSize=0):
        print("""This method returns the filter for the Doppler search
        It accepts the inputs 
        Nfft -- FFT length
        spSym -- samples per symbol
        maskSize -- number of bits in the mask. This results in a table of 2**(maskSize-1) entries

        This method returns the number of masks and the filters in frequency domain (fft length Nfft), complex conjucated and datatype np.complex64
        """)
        raise NotImplementedError('Sub class needs to implement this method')


    def get_symbolLUT2(self,maskLen):
        print("""
        This symbol lookup table looks at the centre bit instead of the last one
        It will return:
            a mapping from symbols to bits 
            The old symbol LUT which can be used for trust weighting

        TODO: for Buccy, it doesn't return the bits yet. returns None instead, which in
        cudaFindUHF_BRMM indicates that the old symbol lookup is used

        """)
        raise NotImplementedError('Sub class needs to implement this method')

    def get_symbolLUT(self,maskLen):
        # returns the lookup table to get from the masks to the symbols
        raise NotImplementedError('returns the symbol LUT')
        

    ##########
    #
    # Decoder stuff 
    #
    ##########
    def get_mask(self):
        print('This method returns the mask for the decoding')

        
    def get_syncFlag(self):
        print('This method returns the syncflag bitsequence')

        
    def decoderPreprocessor(self,signal,**args):
        return signal


    def decoderPostprocessor(self,packet,**args):
        return packet


    def packetDataProcessor(self,packet):
        """
        This is used in IN_DATA mode in case the length byte is whitened or interleaved
        """
    
    def packetEndLenDecoder(self,bits,**args):
        """
        Gets the sequence of bits that indicate the packet length and decodes this to an integer
        """
        return 0


    def Packet(self,*args,**kwargs):
        return Packet(self,*args,**kwargs)
    
    ##########
    #
    # Modulator stuff 
    #
    ##########

    def getFramer(self,confRadio):
        """ 
        select the encoder
        """

        return None


    def getModulator(self,confRadio):
        """3
        select the LUT
        """

        return None


    ################
    #
    # Tx framing setup
    #
    ###############

    def initTxHeader(self,noFlags = DEFAULT_NO_SYNC_FLAGS):
        """
        Initializes the header for the modulation
        Returns the sync sequence and packet header
        """
        print('This method returns the header for the Tx frame')
        
     
    
    def initTxTail(self,noFlags = DEFAULT_NO_SYNC_FLAGS):
        """
        Initializes the tail for the modulation
        Returns sync sequence of the tail and the tail marker
        """
        print('This method returns the tail for the Tx frame')


    def __str__(self):
        return "Protocol base class"
        

    def __repr__(self):
        return "Protocol base class"
    
class Packet():
    """This class defines the packet structure"""

    def __init__(self,protocol,bits,*args,**kwargs):

        self.protocol=protocol
        self.bits = bits

    @property
    def bitsRaw(self):
        return self.bits

    def getBinaryData(self):
        lenPacket = len(self.bits)
        lenBytes = lenPacket//8

        data = np.dot(self.bits[:lenBytes*8].reshape(lenBytes,8),2**np.arange(0,8,1)).astype(np.uint8)
        noError = 0 
        correctBytes = self.bits

        return data, noError, correctBytes


    def printPacket(self, pre_str = "", pos_str = "",verbosity=0,**kwargs):
        """
        This method returns a string with relevant packet info for logging and printing
        The verbosity level can be adjusted
        pre_str and pos_str allow formatted content to be added to the printing
        """


        log.info(f'{pre_str}\tlen: {len(self.bits)}\t Data:\n{printBytesAsHex(self.getBinaryData()[0])}{pos_str}')

    def getBinaryRawData(self):
        return self.bits


    def getAsciiAddress(self):
        return "ASCII_address"

    

    def __str__(self):
        return printPacket()

    def __repr__(self):
        return printPacket()



BITS_PR_LINE = 32
def printPacket(pkt):
    """
    Pretty printing of packet data with line numbering
    Does not convert any data
    """
    s = ''

    nRows = len(pkt)//BITS_PR_LINE

    for r in range(nRows):
        idxB, idxE = r*BITS_PR_LINE,(r+1)*BITS_PR_LINE
        s += f'{r*BITS_PR_LINE:>6.0f}:\t{pkt[idxB:idxE]}\n'

    s += f'{nRows*BITS_PR_LINE:>6.0f}:\t{pkt[(nRows)*BITS_PR_LINE:]}'

    return s
    
