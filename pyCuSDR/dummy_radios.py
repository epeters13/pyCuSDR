# Original Author    : Edwin G. W. Peters @ sdr-Surface-Book-2
#   Creation date    : Sat Jul 10 15:12:03 2021 (+1000)
#   Email            : edwin.peters@unsw.edu.au
# ------------------------------------------------------------------------------
# Last-Updated       : Mon Jul 12 16:27:11 2021 (+1000)
#           By       : Edwin G. W. Peters @ sdr-Surface-Book-2
# ------------------------------------------------------------------------------
# File Name          : dummy_radios.py
# Description        : 
# ------------------------------------------------------------------------------
# Copyright          : Insert license
# ------------------------------------------------------------------------------

import scipy.constants
from lib.freq_from_rangerate import *


class DummyRadio():

    """
    Just a dummy class implementing the methods adressed by rig_server
    """


    def __init__(self):

        self._Fc = 186e6 # set from config
        self._rangerate = 0
        self._doppler = 0 


    @property
    def freq_hl(self):
        # return frequency for hamlib
        return self.Fc + self.doppler

    @freq_hl.setter
    def freq_hl(self,val):
        # extract the rangerate from the frequency provided by hamlib
        self.rangerate = rangerate_from_freq(val,self.Fc)

    @property
    def Fc(self):
        return self._Fc

    @Fc.setter
    def Fc(self,val):
        self._Fc = val

    @property
    def rangerate(self):
        return self._rangerate

    @rangerate.setter
    def rangerate(self,val):
        self.doppler = val*self.Fc/scipy.constants.speed_of_light
        self._rangerate = val

    @property
    def doppler(self):
        return self._doppler

    @doppler.setter
    def doppler(self,val):
        self._doppler = val


    


        
