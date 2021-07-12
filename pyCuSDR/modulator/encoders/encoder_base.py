# Copyright: (c) 2021, Edwin G. W. Petersimport numpy as np

import sys
sys.path.append('../../')
from __global__ import *
import logging

log	= logging.getLogger(LOG_NAME+'.'+__name__)

class Encoder():
    """
    base class for encoder and framer modules
    """

    def __init__(self,protocol,confRadio):
        """
        The init function gets necessary attributes from the protocol and config files.
        """

    def encodeAndFrame(self,data):
        """
        Input: 
            byte data

        Performs:
            Pre-encode
            Frame
            Post-encode

        Returns:
            bit data to get modulated
        """

        return data

    def frame(self,data):
        """
        Input: raw bytes to be framed.

        Output: raw bits ready for post framing
        
        Input:
            Data bytes
        
        Performs (example):
            1) add header
            2) add CRC
            3) LSB encode
            4) stuffing
            5) add flags

        Returns:
            framed bits

        
        """
        return data

    def preframingProcess(self,data):
        """
        Methods that are applied to the data before framing can be wrapped in here
        Examples: endianness encoding, encrypting (although higher level protocol should really do this) 

        Input:
            bytes

        performs (example):
            change endianness

        Output:
            bytes
        """
        return data
        
    def postframingProcess(self,data):
        """
        Methods that are applied to the data after framing can be wrapped in here
        Examples: stuffing, scrambling, encoding

        Input:
            data bits
        
        Performs (example):
            1) scramble
            2) NRZ-S
        
        Returns:
            encoded bits
        """
