# Copyright: (c) 2021, Edwin G. W. Peters

from protocol.protocolBase import *
from lib.filters import rrcosfilter

import numpy as np
from scipy import signal



MASKLEN = 16*8
FLAGLEN = 8*2
PACKETLEN = 1000
RAND_SEED = 123



class Bench_base(ProtocolBase):
    """
    A base class to benchmark demodulation performance
    
    Is supposed to receive a known fixed length signal with known preamble. Compares the signals bit for bit and logs the BER for performance analysis
    """
    name = 'bench_base_class'

    packetEndDetectMode = PacketEndDetect.FIXED
    packetLen = PACKETLEN

    numBitsOverlap = MASKLEN*2 # definitely has to be longer than the mask
    
    def __init__(self,**args):

        try:
            self.conf = args['conf']
        except KeyError as e:
            log.warning(f'No config file specified. Using default fixed packet length of {PACKETLEN}')
            self.packetLen = PACKETLEN

        self.packetLen = attr_in_config(self.conf['Main'],'PacketLen',PACKETLEN)
        self.randSeed = attr_in_config(self.conf['Main'],'RandSeed',RAND_SEED)

            
        log.info(f'Expects packets of length {self.packetLen} bits. Using seed {self.randSeed}')

        
            
    def _get_xcorrMasks(self,maskLen):

        rawMasks = np.zeros((2**(maskLen),maskLen))
        for i in range(2**(maskLen)):
            rawMasks[i] = np.array([np.float(i) for i in list(np.binary_repr(i,width=maskLen))])

        return rawMasks


    ##########
    #
    # Decoder stuff 
    #
    ##########
    numOnesSyncSig = 0 # set in get_syncFlag
    numOnesHeader = 0 # set in get_mask
    syncSigTol = 1 # number of bit errors tolerance accepted in syncFlag
    headerTol = 27 # bit error tolerance on the packet header

    
    def get_mask(self):
        np.random.seed(123) # use the seed for now

        
        mask = np.random.randint(0,2,MASKLEN)

        self.numOnesHeader = np.sum(mask)
        log.info(mask[:min((100,len(mask)))])
        return np.flipud(mask*2-1)

        
    def get_syncFlag(self):
        """ No syncflags needed, this protocol just uses the header"""
        # self.numOnesSyncSig = 11 # we don't want this to match
        # return  np.flipud(np.ones(10))

        np.random.seed(123) # use the seed for now
   
        mask = np.random.randint(0,2,FLAGLEN)

        self.numOnesSyncSig = np.sum(mask)
        log.info(f'num ones {self.numOnesHeader}')
        log.info(mask[:100])
        return np.flipud(mask*2-1)

    
    def Packet(self,*args,**kwargs):
        return Packet_bench(self,*args,**kwargs,packetLen = self.packetLen, randSeed = self.randSeed)



    
   ##########
    #
    # Modulator stuff 
    #
    ##########

    # just placeholders in this setting

    def getFramer(self,confRadio):
        """ 
        select the encoder
        """

        return encoders.BRMM_AX25


    def getModulator(self,confRadio):
        """
        select the LUT
        """

        return modulators.GMSKmod
 
    TX_FLAG = np.array([0,1,1,1,1,1,1,0],dtype=np.uint8) #  0x7e

    def initTxHeader(self,noFlags = DEFAULT_NO_SYNC_FLAGS):
        """
        Initializes the header for the modulation
        Returns the sync sequence and packet header
        """
        HeaderDst = [0x86,0xA2,0x40,0x40,0x40,0x40,0xE1]
        HeaderSrc = [0x9C,0x9E,0x86,0x82,0x98,0x98,0x60]
        HeaderCtrl = 0x03
        HeaderPID = 0xF0

        flags = np.tile(self.TX_FLAG,noFlags)
        Header = np.r_[HeaderDst,HeaderSrc,HeaderCtrl,HeaderPID]

        return flags, Header

    def initTxTail(self,noFlags = DEFAULT_NO_SYNC_FLAGS):
        """
        Initializes the tail for the modulation
        Returns sync sequence of the tail and the tail marker
        """
        return np.tile(self.TX_FLAG,noFlags), np.array([],dtype=np.uint8)





class Packet_bench(Packet):
    """
    Packet structure for benchmark packets. Fixed length known data based of a random seed

    """
    
    def __init__(self,protocol,bits,frameStartIdx,maskBitErrors,frameSplitIdx=0,packetLen = PACKETLEN,randSeed = RAND_SEED):

        self.protocol= protocol
        self.frameStartIdx = frameStartIdx
        self.maskBitErrors = maskBitErrors
        self.bits = bits.astype(np.int8)
        self.frameSplitIdx = frameSplitIdx

        # specific for benchmark packet
        self.packetLen = packetLen
        self.randSeed = randSeed # seed used to generate length packetLen packet
        
    def checkPacketData(self):
        """
        Checks the data of the packet against the known random seed

        """

        if len(self.bits) < self.packetLen:
            log.warning(f'Length of received bits too short ({len(self.bits)}), expected {self.packetLen}')
            return -0.1

        state = np.random.get_state()
        np.random.seed(self.randSeed)
        compareSequence = np.random.randint(0,2,self.packetLen)
        np.random.set_state(state) # restore random state


        bit_error_loc = np.where(self.bits[:self.packetLen] != compareSequence)[0]

        log.debug(f'Bit errors at {bit_error_loc}')

        return len(bit_error_loc)


    def printPacket(self, pre_str = "", pos_str = "", verbosity=0, workerId= ''):

        bit_errors = self.checkPacketData()
        log.info(pre_str + f'\tbit errors {bit_errors}\t BER (this packet) {bit_errors/self.packetLen}'+ pos_str)

        # log.info(f'worker {workerId}\tbits raw {self.bits}')
        # pStr = printPacket(self.bits)
        # log.info(f'worker {workerId}\tbits raw\n{pStr}')



    def getBinaryData(self):
        # bit_errors = -self.checkPacketData() # negative number marks it on the stats plot # takes time
        bit_errors = 0
        return self.bits, bit_errors, self.bits

        

