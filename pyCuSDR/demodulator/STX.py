# Copyright: (c) 2021, Edwin G. W. Peters

from demodulator.demodulator_base import Demodulator as Demodulator_base
import time

class Demodulator(Demodulator_base):

    def uploadAndFindCarrier(self,samples):
        """
        For STX we don't do Doppler search
        """
        # t = time.time()
        self.__thresholdInput(samples)
        # print('theshold time {}'.format(time.time()-t))

        # t = time.time()
        self.uploadToGPU(samples)
        # print('upload time {}'.format(time.time()-t))

        # need to be compatible with the return format TODO SNR estimate for STX
        return 0, 0, self.clippedPeakIPure, 0

    def demodulate(self):
        return self.demodulateSTX()
    
