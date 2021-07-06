# Copyright: (c) 2021, Edwin G. W. Peters

from lib.shift_registers import PN9
from protocol.protocolBase import *

from protocol.FSK2_base import FSK2
from protocol.GFSK2_base import GFSK2
from scipy import signal
import crcmod

import numpy as np

DEFAULT_SYNC = [0xAB,0x35,0xAB,0x35]
DEFAULT_PREAMBLE = [0xAA]
DEFAULT_NUM_PREAMBLE = 4 # the number of repeats of the preamble
NUM_PREAMBLE = DEFAULT_NUM_PREAMBLE # used in Packet. updated in __init__
DEFAULT_NO_SYNC_FLAGS = 500


# CC11xx Direct responses
RESP_LOOPBACK = 0x27
RESP_GET_VERSION = 0x1D # command to get version (no data)
RESP_MSG_VERSION = 0x1E
RESP_MSG_PING = 0x20 # get a ping with 0x24
RESP_MSG_CONFIG = 0x23 # get config with 0x22
RESP_SL = 0xc1
RESP_WARNINGS = [RESP_LOOPBACK,RESP_MSG_VERSION,RESP_MSG_PING,RESP_MSG_VERSION, RESP_SL]


modIDX = 0  # 0 FSK, 1 GFSK
modulationNames = ['FSK-2', 'GFSK-2']
modProtocols = [FSK2,GFSK2]

# LST ReedSolomon settings
CC_HEADER_N_BYTES = 5
CC_FOOTER_N_BYTES = 2
CC_PACKET_TYPE_BYTE_NUM = 4
CC_SPACELINK_TYPE_VALUE = 0x1C



class CC11xx(modProtocols[modIDX]):

    name = f'CC11xx {modulationNames[modIDX]}'
    
    packetEndDetectMode = PacketEndDetect.FIXED
    packetLen = (256 + 9 + 2) * 8 # max data + header and flags + optional CRC
    packetEndLenField = 9
    packetEndLenFieldNumBytes = 1
    packetEndLenEndianness = PacketLenEndianness.LITTLE
    deWhiten = True # for downlink
    whiten = True # for uplink

    # for FSK we want to sum all masks before doing the Doppler Search
    SUM_ALL_MASKS_PYTHON = True

    numBitsOverlap = 2048
    def __init__(self,**args):
        self.PN9seq = PN9() # store the PN9 LUT to be used in the decoder

        cfg = args.get('conf',None)
        cfg_prot = cfg['Radios'].get('Protocol',None)
        if cfg_prot:
            self.rx_preamble      = json_str_list_to_int_list(cfg_prot['rx_preamble'])
            self.rx_sync_seq      = json_str_list_to_int_list(cfg_prot['rx_sync_seq'])
            self.tx_preamble      = json_str_list_to_int_list(cfg_prot['tx_preamble'])
            self.tx_num_preambles = cfg_prot['tx_num_preambles']
            self.tx_sync_seq      = json_str_list_to_int_list(cfg_prot['tx_sync_seq'])
            global NUM_PREAMBLE
            NUM_PREAMBLE = len(self.tx_preamble * self.tx_num_preambles)
        else:
            print(f'CC11xx: No config provided. Using default preamble')
            self.tx_preamble      = DEFAULT_PREAMBLE
            self.rx_sync_seq      = DEFAULT_SYNC*4
            self.tx_preamble      = DEFAULT_PREAMBLE
            self.tx_num_preambles = DEFAULT_NUM_PREAMBLE
            self.tx_sync_seq      = DEFAULT_SYNC

            
    def _get_xcorrMasks(self,maskLen):

        rawMasks = np.zeros((2**(maskLen),maskLen))
        for i in range(2**(maskLen)):
            rawMasks[i] = np.array([np.float(i) for i in list(np.binary_repr(i,width=maskLen))])

        return rawMasks

    def get_symbolLUT2(self,maskLen):
        """
        LUT for CC11xx. uses center symbol
        """

        masks = self._get_xcorrMasks(maskLen)

        sampleIdx = int(maskLen/2)

        bitLUT = masks[:,sampleIdx]

        symLUT = np.zeros((2**(maskLen-1),2),dtype=np.int)
        for i in range(2**(maskLen-1)):
            symLUT[i] = np.array([
                i*2+1,
                i*2
            ])
        return bitLUT, np.concatenate((symLUT,symLUT),axis=0)

    

    ##########
    #
    # Decoder stuff 
    #
    ##########
    numOnesSyncSig = 0 # set in get_syncFlag
    numOnesHeader = 0 # set in get_mask
    syncSigTol = 2 # number of bit errors tolerance accepted in syncFlag
    headerTol = 5 # bit error tolerance on the packet header


    def get_mask(self):
        '''This method returns the mask for the decoding'''
        # add preamble to sync
        # preambleA = [0xAA for i in range(NUM_PREAMBLE)] + SYNC
        preambleA = self.rx_preamble + self.rx_sync_seq
        # print(f'Looking for preamble {preambleA}')
        preambleB = [list(bin(a)[2:].zfill(8)) for a in preambleA]
        preambleM = np.array(preambleB).astype(np.float)
        mask = np.reshape(preambleM,np.prod(preambleM.shape))

       
        self.numOnesHeader = np.sum(mask)
        mask = mask*2 - 1;
        return np.flip(mask,axis=0)


    def get_syncFlag(self):
        """
        4 sync flags
        """
        # preambleA = [0xAA for i in range(NUM_PREAMBLE)]
        preambleA = self.rx_preamble
        preambleB = [list(bin(a)[2:].zfill(8)) for a in preambleA]
        preambleM = np.array(preambleB).astype(np.float)
        mask = np.reshape(preambleM,np.prod(preambleM.shape))
        self.numOnesSyncSig = np.sum(mask > 0)
        return mask*2-1

    
    def decoderPreprocessor(self,signal):
        return signal


    def decoderPostprocessor(self,packet):
        return packet


        
    ##########
    #
    # Modulator stuff 
    #
    ##########

    def getFramer(self,confRadio):
        """ 
        select the encoder
        """

        return encoders.CC11xx


    def getModulator(self,confRadio):
        """
        select the LUT
        """

        return modulators.FSKmod
        
    ################
    #
    # Tx framing setup
    #
    ###############


    # CC11xx_TX_FLAG = np.array([1,0,1,0,1,0,1,0],dtype=np.uint8) #  0xAA

    def initTxHeader(self):
        """
        Initializes the header for the modulation
        Returns the sync sequence and packet header
        """
        # HeaderSync = SYNC[:2]
        headerSync = np.array([self.tx_sync_seq],dtype=np.uint8)

        preamble = np.unpackbits(np.array([self.tx_preamble]*self.tx_num_preambles,dtype=np.uint8))
        header = np.unpackbits(headerSync)

        return preamble, header

    def initTxTail(self):
        """
        Initializes the tail for the modulation
        Returns sync sequence of the tail and the tail marker
        """
        # no tail needed
        return np.array([],dtype=np.uint8),np.array([],dtype=np.uint8)


    def Packet(self,*args,**kwargs):
        return PacketCC11xx(self,*args,**kwargs)




class PacketCC11xx(Packet):
    """
    CC11xx packet. This receives the max amount of data and parses the packet length field
    
    Packet format
    -------------------------------------------------------------------------------------------------
    |              |             |            |            |                 |                      |
    | flags (4 b)  |  mask (4 b) | pLen (1 b) | Addr (1 b) | data (pLen-1 b) | CRC (2 b optional)   |
    |              |             |            |            |                 |                      |
    -------------------------------------------------------------------------------------------------

    """

    packetLenFieldIndex = 8
    packetLenDecVector = 2**np.arange(7,-1,-1)

    # TODO: These numbers need to be set based on config
    flagLen = NUM_PREAMBLE # number of flag bits
    maskLen = 4 # number of bits in sync mask
    pLen = 1 # number of bits for length bit
    CRClen = 2 # number of bits for CRC
    packetPreOverHead = flagLen +  maskLen + pLen
    packetPosOverHead = CRClen
    packetLenOverHead = packetPreOverHead + packetPosOverHead 



    
    def __init__(self,protocol,bits,*args,**kwargs):

        self.protocol=protocol
        
        if self.protocol.deWhiten:
            self.PN9 = protocol.PN9seq
            self.packetLen = packetLen = np.bitwise_xor(self.getPacketLen(bits),self.PN9[0])
        else:
            self.packetLen = packetLen = self.getPacketLen(bits)

        self.bits = np.array(bits)[:int(packetLen+self.packetLenOverHead)*8]
        self.crc16 = crcmod.mkCrcFun(0x18005,rev = False, initCrc = 0xFFFF, xorOut=0x0000)  
       

    def getPacketLen(self,bits):

        pLenBits = bits[self.packetLenFieldIndex*8:self.packetLenFieldIndex*8+8]
        return np.uint16(np.sum(pLenBits * self.packetLenDecVector))
        
    def deWhiten(self,byteData):

        log.debug(f'packet length {self.packetLen} len bytes {len(byteData)}')
        pOffset = 0
        byteData[pOffset:pOffset + self.packetLen] = np.bitwise_xor(byteData[pOffset:pOffset + self.packetLen],self.PN9[1:self.packetLen + 1])
        
        
    @property
    def bitsRaw(self):
        return self.bits

    def getBinaryData(self):
        lenBytes = np.uint8(self.packetLen)
        # lenPacket = lenBytes * 8 
        
        data = np.dot(self.bits[self.packetPreOverHead*8:(self.packetPreOverHead + lenBytes)*8].reshape(lenBytes,8),2**np.arange(7,-1,-1)).astype(np.uint8)

        if self.protocol.deWhiten:
            self.deWhiten(data)

        
        #CRCt = np.dot(self.bits[-self.CRClen*8:].reshape(self.CRClen,8),2**np.arange(7,-1,-1))
        CRCt = np.dot(self.bits[-self.CRClen*8:].reshape(self.CRClen,8),2**np.arange(7,-1,-1))
        CRC = np.sum(CRCt*np.array([1,2**8]))
        dCheck = np.r_[lenBytes,data].copy().astype(np.uint8)
        log.debug(f'dataCRC {printBytesAsHex(dCheck.tobytes())}\t {data}')
        CRC_check = self.crc16(dCheck.tobytes())
        
        CRCL = CRC_check//256
        CRCH = np.uint8(CRC_check)
        CRCPacked = np.r_[CRCH,CRCL].astype(np.uint8)

        
        noError = CRC!=CRC_check
        correctBytes = data

        return data, noError, correctBytes


    def printPacket(self, pre_str = "", pos_str = "",verbosity=0,**kwargs):
        """
        This method returns a string with relevant packet info for logging and printing
        The verbosity level can be adjusted
        pre_str and pos_str allow formatted content to be added to the printing

        Additionally, packets with certain response codes can be printed using warning level to always appear in the log
        """

        data = self.getBinaryData()[0]

        try:
            resp_code = data[4]
            if resp_code == RESP_LOOPBACK:
                RSSI = data[-5]
                if RSSI >= 128:
                    RSSI = (RSSI-256)/2 - 75
                else:
                    RSSI = (RSSI)/2 - 75

                freq_est = data[-3]
                if freq_est >= 128:
                    freq_est -= 256
                freq_est = float(freq_est)*26e6/2**14

                log.warning(f'{pre_str}\tLoopback response len: {self.packetLen} RSSI {RSSI} dBm, LQI {data[-4]}, freq_est {freq_est} Hz, HW_ID {data[-2:]}\nloopback data: {printBytesAsHex(data[5:-5])}\n{pos_str}')

            elif resp_code in RESP_WARNINGS:
                log.warning(f'{pre_str}\tlen: {self.packetLen} bytes\t Data:\n{printBytesAsHex(self.getBinaryData()[0])}{pos_str}')
            else:
                log.warning(f'{pre_str}\tlen: {self.packetLen} bytes\t Data:\n{printBytesAsHex(self.getBinaryData()[0])}{pos_str}')
        except Exception as e:
            log.warning(f'{pre_str}\tlen: {self.packetLen} bytes\t Data:\n{printBytesAsHex(self.getBinaryData()[0])}{pos_str}')
            log.exception(e)
            
                
    def getBinaryRawData(self):
        return self.bits


    def getAsciiAddress(self):

        headerBits = np.reshape(self.bits[:8*8],(8,8))*self.packetLenDecVector
        return np.vectorize(hex)(np.sum(headerBits.astype(np.int),axis=1))
        # return "ASCII_address"

    

    def __str__(self):
        return printPacket()

    def __repr__(self):
        return printPacket()
