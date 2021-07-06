# Copyright: (c) 2021, Edwin G. W. Peters

import sys
sys.path.append('../../')
from __global__ import *
import logging
import numpy as np

log	= logging.getLogger(LOG_NAME+'.'+__name__)

class BaseLUT():
    """
    Base class for LUT
    """

    def __init__(self,protocol,confRadio):
        """
        Creates the LUT, uses attributes from protocol and confRadio
        """
        self.LUT = None


    def getLUT(self):
        """
        returns the LUT
        """
        return self.LUT



    def modulateData(self,data,LUT):
        """
        Modulates the data using the LUT, which has been scaled with Doppler and frequency offsets in the modulator
        """

        return data

        
