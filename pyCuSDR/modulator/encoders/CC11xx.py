# Copyright: (c) 2021, Edwin G. W. Peters
import modulator
from modulator.encoders.encoder_base import *

import sys
sys.path.insert(1,'../../')

from lib.shift_registers import PN9

import crcmod

MAX_TX_DATA_LEN = 256

# log.setLevel(logging.DEBUG)

class CC11xx(Encoder):
    """
    CC11xx compatible encoder
    
    preframe: None -- Perhaps whiten
    frame:
       1) add header
       2) add CRC
       3) whiten

    postframe:
       Nothing
    """
    name = 'CC11xx'
    
    def __init__(self,protocol,confRadio):

        self.protocol = protocol
        self.whiten = self.protocol.whiten
        
        self.Flags, self.Header = protocol.initTxHeader()
        self.TailFlags, self.Tail = protocol.initTxTail()

        self.whiten = protocol.whiten

        # CRC 16 encoder
        self.crc16 = crcmod.mkCrcFun(0x18005,rev = False, initCrc = 0xFFFF, xorOut=0x0000)  

        # PN9 whitener
        if self.whiten:
            self.PN9seq = PN9()

    def preframingProcess(self,byteData):
        """
        Input:
            bits

        performs:
            whitening and interleaving if enabled

        Output:
            bits
        """
        if self.whiten:
            byteData[:] = np.bitwise_xor(byteData,self.PN9seq[:len(byteData)])
        return byteData
    
    def encodeAndFrame(self,data):
        """
        Input: 
            byte data

        Performs:
            Count length
            Convert to bits
            Pre-encode
            Frame
            Post-encode

        Returns:
            bit data to get modulated
        """
        
        if type(data) == list: # ensure that we have a numpy array
            data = np.array(data) 

        dataLen = len(data) + 2 # 2 extra for CRC
        if dataLen > MAX_TX_DATA_LEN:
            raise modulator.DataLengthError(f'TX maximum allowed data length {MAX_TX_DATA_LEN} bytes. Got {dataLen} bytes')

        data = np.r_[dataLen,data] # data length is first byte of message

        CRC = self.crc16(data.astype(np.uint8))

        CRCL = CRC//256
        CRCH = np.uint8(CRC)
        CRCPacked = np.array([CRCH,CRCL]).astype(np.uint8)
        # CRCPacked = np.array([0xFF,0xff]).astype(np.uint8)
        
        log.debug(f'CRC data {np.vectorize(hex)(data)}')
        data = np.r_[data,CRCPacked].astype(np.uint8)

        log.error(f'with CRC {np.vectorize(hex)(data.astype(np.uint8))}, CRC: {np.vectorize(hex)(CRCPacked)}')
        data = self.preframingProcess(data) # do whitening if necessary
        log.debug(f'whitened {np.vectorize(hex)(data.astype(np.uint8))}, CRC: {np.vectorize(hex)(CRCPacked)}')
        
        bitData = np.unpackbits(data.astype(np.uint8)) # possible interleaving happens on the bits

        # log.info(f'Bit data {bitData}')

        dataFramed = self.frame(bitData) # add headers and optional CRCs
        
        dataEncoded = self.postframingProcess(dataFramed) # don't do anything for OpenLST

        # log.info(f'Encoded bits {dataEncoded[:1000]}')
        return dataEncoded

        
    def frame(self,data):
        """
        Input:
            Data bits
        
        Performs:
            1) add preamble
            2) add SYNC

        Returns:
            framed bits
        """

        preamble = np.r_[self.Flags,self.Header].astype(np.uint8)
        
        
        dataFramed = np.r_[preamble,data] # just put some flags behind for test


        return dataFramed
        

    def postframingProcess(self,data):
        """
        Input:
            data bits
        
        Performs:
            nothing

        Returns:
            encoded bits
        """
        
        return data
        
