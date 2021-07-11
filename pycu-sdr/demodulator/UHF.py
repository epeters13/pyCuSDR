# Copyright: (c) 2021, Edwin G. W. Peters

from demodulator.demodulator_base import Demodulator as Demodulator_base

class Demodulator(Demodulator_base):

    def uploadAndFindCarrier(self,samples):
        """
        Performs:
            thresholding -- remove large spikes of interference
            uploadToGPU  -- store data on GPU and perform FFT
            findUHF      -- find the UHF modulated carrier
        """
        # self.__thresholdInput(samples)
        self.uploadToGPU(samples)
        return self.__findUHF(samples)


    def demodulate(self):
        return self.demodulateUHF()
