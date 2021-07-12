# Copyright: (c) 2021, Edwin G. W. Peters

from __global__ import *

from multiprocessing import Process, Event, Value, Lock
import numpy as np
import logging
import zmq
import time
import sys
import signal
import time 
import scipy
import demodulator
import sigFIFO

log	= logging.getLogger(LOG_NAME+'.'+__name__)

TOLVAL = 0.5 # tolerance on spSym for stats (SNR, TxOffset, RxOffset)

def radioBackendVoteGroupIDX(radioBackend):
    """
    This is only used when trustprocessor is enabled. 

    Assign different radio channels to different groups to avoid cross voting
    """
    if radioBackend == "UHF":
        return demodulator.UHF, 0
    elif radioBackend == "STX":
        return demodulator.STX, 1
    elif radioBackend == "STX1":
        return demodulator.STX, 2
    elif radioBackend == "STX2":
        return demodulator.STX, 3
    else:
        raise Exception('radioBackend {} not defined in voteGroup'.format(radioBackend))
    


class Demodulator_process(Process):

    def __init__(self,conf,protocol,radio):
        Process.__init__(self)

        # set all sub module log levels similar to this modules
        demodulator.log.setLevel(log.level)
        sigFIFO.log.setLevel(log.level)
        self.logLevel = log.level # set this in the thread
        # for averaging time
        self.timeMA = 0 # moving average time
        self.iterCount = 0 # iteration for moving average

        try:
            worker_radio_name = conf['Radios']['Rx'][radio]['name']
        except KeyError:
            log.warning(f'Protocol name not specified in config. Using default radio name: {radio}')
            worker_radio_name = radio
            
        self.radioName = radio
        self.conf = conf
        self.protocol = protocol
        self.confRadio = confRadio = conf['Radios']['Rx'][radio] # Config for the specific radio
        self.confGPU = confGPU = conf['GPU'][confRadio['CUDA_settings']]
        
        self.overlap = 2**confGPU['overlap']
        self.blockSize = 2**confGPU['blockSize']
        self.samplesPerSlice = self.blockSize - self.overlap
        log.info('[{}]: Block size {} samples\tOverlap {} samples'.format(self.radioName,self.blockSize,self.overlap))

        try: # Option to print all demodulator info
            self.PRINT_ALWAYS = conf['LogInfo']['demodulator_print_always']
            self.PRINT_NTH_BLOCK = conf['LogInfo']['demodulator_print_interval']
            self.PRINT_THRESHOLD_ENABLED = conf['LogInfo']['demodulator_print_threshold_enabled']
            self.PRINT_THRESHOLD_LVL = conf['LogInfo']['demodulator_print_threshold_lvl']
        except:
            # default values in case the config fails
            log.warning('Failed reading LogInfo from config. Setting default print values')
            self.PRINT_ALWAYS = False
            self.PRINT_NTH_BLOCK = 5
            self.PRINT_THRESHOLD_ENABLED = False
            self.PRINT_THRESHOLD_LVL = 4


        self.baudRate = confRadio['baud']
        self.spSym = confRadio['samplesPerSym']

        self.sigFIFOTimeout = conf['Demodulator']['timeoutSeconds']

        self.name = 'demod-%s' % (radio)
        try:
            if 'Interfaces' in self.confRadio.keys() and 'RxInPort' in self.confRadio['Interfaces'].keys():
                self.RxInAddr = confRadio['Interfaces']['RxInPort']
            else:
                log.warning(f'[{self.radioName}]: [deprecation] Failed to find "RxInPort" in "Interfaces" for current radio. Resolving to old "RxInPort"')
                self.RxInAddr = confRadio['RxInPort']
        except KeyError as e:
            log.error('[{}]: No Rx input channel defined\tMessage:'.format(radio))
            log.exception(e)
            raise e


        # initialize the worker ID and radio backend
        self.workerId = conf['Main']['workerId'] + '-' + worker_radio_name
        self.radioBackend = confRadio['radioBackend']

        # to assign groups of channels to compare in the softCombiner
        if 'voteGroup' in confRadio:
            self.demodulator = radioBackendVoteGroupIDX(self.radioBackend)[0]
            self.voteGroup = radioBackendVoteGroupIDX(confRadio['voteGroup'])[1]
        else:
            log.warning(f'[{self.radioName}]: Did not find "voteGroup" in radio config. Setting based on radio Backend')
            self.demodulator, self.voteGroup = radioBackendVoteGroupIDX(self.radioBackend)

        if 'Protocol' in self.confRadio.keys():
            # tells the decoder which protocol to use
            self.decoderProtocol = self.confRadio['Protocol']
        else:
            self.decoderProtocol = 'None'
            
        # local decoder
        if 'Interfaces' in self.confRadio.keys() and 'demodOut' in self.confRadio['Interfaces'].keys():
            self.demodOutAddr = self.confRadio['Interfaces']['demodOut']
        else:
            log.error(f'[{self.radioName}]: [deprecation] Failed to find "demodOut" port in "Interfaces" for current radio. Resolving to old "demodOut" from default interfaces')
            self.demodOutAddr = self.conf['Interfaces']['Internal']['demodOut']

        
        try:
            self.demodOutAddrClient = self.confRadio['Interfaces']['demodOutExternal']
            self.client = True

        except KeyError:
            log.error('[{}]: No external decoder found in config file. '.format(self.radioName))
            self.demoutOutAddrClient = None
            self.client = False
            

        ## Variables that can be read and/or set from outside the process for monitoring and online configuration
        self.__rangerate = Value('f',1)
        self.__Fc = Value('d',int(self.confRadio['frequency_Hz'] - self.confRadio['frequencyOffset_Hz']))
        if 'Tx' in self.conf['Radios'].keys() and 'frequency_Hz' in self.conf['Radios']['Tx'].keys(): 
            self.TxFc = self.conf['Radios']['Tx']['frequency_Hz'] # used for offset computation. TODO shouldn't be done here
        else:
            # log.warning(f'[{self.radioName}]: TODO: this should not be here --- Tx not found in config file. Setting Tx Fc equal to Rx Fc')
            self.TxFc = self.__Fc.value
            
        self.__Fs = Value('d',int(self.baudRate*self.spSym))
        self.__TxRangeRate = Value('f',0) # used for offset computation
        self.__RxIFFreqOffset = Value('f',0) # computed IF Rx frequency offse
        self.__TxIFFreqOffset = Value('f',0) # estimated -- Includes the setting in the config
        self.__SNR = Value('f',0)
        self.__RxFreqOffset = Value('f',0) # raw frequency offset from self.Fc  (doppler + offset)
        self.__baudRateEst = Value('f',0) # estimate of the baudrate based on the samples per symbol
        
        self.daemon = True
        self.runStatus = Event()
        self.runStatus.set() # set running == True

        self.GRCTimeoutFlag = Event() # This flag is set if the zmq port for GRC imput timed out
        log.error('[{}]: Demodulator initialized'.format(self.radioName))

    def sigTermHandler(self,signum,frame):
        # We make SIGTERM raise an exception to exit the main thread if desired
        #raise SystemExit('SIGTERM received')
        pass

    def sigTermHandler_atexit(self,signum,frame):
        # We make SIGTERM raise an exception to exit the main thread if desired
        raise SystemExit('SIGTERM received')
        # pass

    def stop(self):
        """
        Stops the process in a clean way
        """
        log.info('PID {} -- Received request to stop'.format(self.pid))
        self.runStatus.clear()
        #self.terminate()

    def GRCTimeout(self):
        """
        Can be used to poll whether the process is running
        """
        return self.GRCTimeoutFlag.is_set()

    def computeMATime(self,t):
        self.iterCount += 1 # iteration for moving average
        self.timeMA = self.timeMA + (t-self.timeMA)/ self.iterCount # moving average time

        return self.timeMA
    
    def run(self):
        """
        wait for data on RxPipeIn
        find doppler
        demodulate signal
        send to trustprocessor or decoder
        """

        time.sleep(.5) 

        log	= logging.getLogger(LOG_NAME+'.'+ self.name)
        log.setLevel(self.logLevel)


        self.GRCTimeoutFlag.clear() # is set when the ZMQ inport port times out

        ctx = zmq.Context()

        log.warning('[{}]: Configuring demodulator output socket: {}'.format(self.radioName,self.demodOutAddr))
        try:
            demodOut = ctx.socket(zmq.PUSH)
            demodOut.connect(self.demodOutAddr)
            
        except Exception as e:
            log.error('[{}]: Error while configuring demodulator output socket'.format(self.radioName))
            log.exception(e)
            
            raise e

        if self.client == True:
            # Also send the data out on the client socket
            log.warning('[{}]: Configuring demodulator remote output socket: {}'.format(self.radioName, self.demodOutAddrClient))
            demodOutClient = ctx.socket(zmq.PUSH)
            demodOutClient.setsockopt(zmq.LINGER,1000)
            demodOutClient.connect(self.demodOutAddrClient)
        
        try:
            log.warning('[{}]: Initializing input buffer on {}'.format(self.radioName,self.RxInAddr))
            sigIn = sigFIFO.SigFIFO(self.RxInAddr, self.samplesPerSlice,
                                    dtype = np.complex64, enableTimeoutException=True,
                                    timeoutRetries = self.sigFIFOTimeout,
                                    runStatus = self.runStatus)
        except Exception as e:
            log.error('[{}]: Error while initializing input FIFO'.format(self.radioName))
            log.exception(e)
            demodOut.close()
            raise e

        try:
            log.info('[{}]: Initializing doppler finder and demodulator'.format(self.radioName))
            demod = self.demodulator.Demodulator(self.conf,self.protocol,self.radioName) # must be initialized in process
        except Exception as e:
            log.error('[{}]: Error while initializing demodulator'.format(self.radioName))
            log.exception(e)
            demodOut.close()
            del sigIn
            raise e


        count = 0
        raw = np.zeros(self.blockSize,dtype=np.complex64)

        
        # raw = np.zeros(self.blockSize,dtype=np.complex64) # old way remove
        raw = demod.get_signalBufferHostPointer() #GPU_bufSignalTime_cpu_handle


        data = {'workerId' : self.workerId,
                'count' : 0,
                'timestamp': 0,
                'voteGroup': self.voteGroup, # for voting in the decoder and trustProcessor
                'doppler': 0, 
                'doppler_std': 0,
                'data': np.array([]),
                'trust': np.array([]),
                'spSymEst': 0,
                'SNR' : float(0),
                # 'RxFreqOffset' : float(RxFreqOffset),
                # 'TxFreqOffsetEst' : float(TxFreqOffset),
                # 'TxRangeRate': float(self.TxRangeRate),
                'rangerateEst' : 0,
                'baudRate': self.baudRate,
                'baudRate_est' : 0,
                'sample_rate' : self.Fs,
                'protocol': self.decoderProtocol}

        
        log.info('[{}]: Demodulator process initialized and running'.format(self.radioName))
        try:
            orig_sigterm_handler = signal.getsignal(signal.SIGTERM) # store this for when shutting down. Else the process may hang
            signal.signal(signal.SIGTERM, self.sigTermHandler)
            # get the pinned memory array to store the uploaded signal in
            while self.runStatus.is_set():
                try:
                    ts = time.time()
                    raw[self.overlap:] = sigIn.getBlock()
                    # log.info(f'Time sigFIFO {time.time()-ts} s')
                    
                    data['timestamp'] = timeStamp = time.time()
                    data['count'] = count
                    # dopp,sdev,thresHoldIdx,SNR = demod.uploadAndFindCarrier(raw)
                    data['doppler'],data['doppler_std'],thresHoldIdx,data['SNR'] = demod.uploadAndFindCarrier(raw)
                    # log.info('[{}]: doppFind time {} s'.format(self.radioName,time.time()-timeStamp))
                    ts = time.time()
                    # dataBits,centres,trust,spSymEst = demod.demodulate()
                    data['data'],centres,data['trust'],data['spSymEst'] = demod.demodulate()
                    data['baudrate_est'] = self.Fs/data['spSymEst']
                    # log.info('[{}]: demodulate time {} s'.format(self.radioName,time.time()-ts))

                    # compute frequency offsets and rangerate
                    # TxFreqOffset, RxFreqOffset, rangerate = self.computeTxFreqOffset(dopp,spSymEst)
                    TxFreqOffset, RxFreqOffset, data['rangerate'] = self.computeTxFreqOffset(data['doppler'],data['spSymEst'])
                    self.SNRStats(data['SNR'],data['spSymEst'])

                    try:
                        ts = time.time()
                        #demodOut.send_json(data,zmq.NOBLOCK)
                        demodOut.send_pyobj(data,0.001)
                        log.debug(f'Time send data {time.time()-ts} s')

                    except zmq.error.Again as e:
                        log.error('[{}]: failed to send data to decoder. Message:  [{}]'.format(self.radioName,e))

                    if self.client == True: # send to remote decoder
                        try:
                            demodOutClient.send_pyobj(data,zmq.NOBLOCK)
                        except zmq.error.Again as e:
                            log.warning('[{}]: failed to send data to external decoder. Message:  [{}]'.format(self.radioName,e))

                        
                    # symErrorRate = np.sum(trust<0)/len(trust)*100

                    timeSpend = time.time()-timeStamp # time spent this iteration
                    self.computeMATime(timeSpend) # moving average time
                    if self.PRINT_THRESHOLD_ENABLED is True:
                        printStats = float(data['SNR']) > self.PRINT_THRESHOLD_LVL
                    else:
                        printStats = False
                    if log.level < logging.INFO or count % self.PRINT_NTH_BLOCK == 0 or self.PRINT_ALWAYS == True or printStats:
                        # log.info('[{}]: rangerate {: 6.0f} m/s, sd {: 5.5f} Hz, TxFreqOffset {:4.0f} Hz, SNR {: 2.1f} dB, symErrorRate = {: 3.2f} %, est spsym {: 3.2f}, time {: 3.2f} ms (avg {: 3.2f} ms)'.format(self.radioName,
                        log.info('[{}]: freq offset {: 6.0f} Hz, sd {: 5.5f} Hz, TxFreqOffset {:4.0f} Hz, SNR {: 2.1f} dB, est spsym {: 3.2f}, time {: 3.2f} ms (avg {: 3.2f} ms), rate {:5.0f} ksamples/s (avg {:5.0f})'.format(self.radioName,
                                                                                                                                                                                                                                                            data['doppler'], data['doppler_std'], TxFreqOffset,data['SNR'], data['spSymEst'], timeSpend*1000, self.timeMA*1000,(self.blockSize-self.overlap)/timeSpend/1000,(self.blockSize-self.overlap)/self.timeMA/1000))


                    
                    raw[:self.overlap] = raw[-self.overlap:]
                    count += 1
                except (TimeoutError, ConnectionRefusedError) as e:
                    log.info('[{}]: ZMQ from GRC timed out'.format(self.radioName))
                    count = 0
                    if not self.GRCTimeoutFlag.is_set():
                        self.GRCTimeoutFlag.set()
                    
                    
        except Exception as e:
            log.exception(e)
        finally:
            # np.save(f'Doppler_{self.radioName}',np.array(Dopplers))
            # clean up
            demodOut.close()
            del sigIn
            del demod
            log.info('[{}]: Finished'.format(self.radioName))
            signal.signal(signal.SIGTERM, orig_sigterm_handler)  # restore signal, such that process.terminate() can kill this
            sys.stdout.flush()


            
    def computeTxFreqOffset(self,Doppler_Hz,spSym):
        ## Note: what is called doppler is actually the frequency offset from Fc. It needs to be compensated for the DC IF frequency offset to get the actual doppler
        Rx_rangerate = -Doppler_Hz/self.Fc*scipy.constants.speed_of_light
        dRangeRate = self.TxRangeRate - Rx_rangerate
        log.debug('TxRangeRate: {} Hz'.format(self.TxRangeRate))

        # set values for external access
        rangerate = Rx_rangerate
        RxIFFreqOffset = dRangeRate*self.Fc/scipy.constants.speed_of_light
        TxFreqOffset = dRangeRate*self.TxFc/scipy.constants.speed_of_light

        
        self.freqOffsetEstStats(TxFreqOffset,RxIFFreqOffset,rangerate,spSym,Doppler_Hz)

        return TxFreqOffset, RxIFFreqOffset, rangerate
    # setters and getters
    @property
    def Fs(self):
        return self.__Fs.value

    @Fs.setter
    def Fs(self,Fs):
        with self.__FS.get_lock():
            self.__Fs.value = Fs
            
    @property
    def rangerate(self):
        with self.__rangerate.get_lock():
            val = self.__rangerate.value
            self.__rangerate.value = 0
        return val

    @rangerate.setter
    def rangerate(self,rangerate):
        with self.__rangerate.get_lock():
            self.__rangerate.value = np.double(rangerate)

    
    @property
    def Fc(self):
        return self.__Fc.value

    @Fc.setter
    def Fc(self,Fc):
        with self.__Fc.get_lock():
            self.__Fc.value = np.double(Fc)

    @property
    def TxRangeRate(self):
        return self.__TxRangeRate.value

    @TxRangeRate.setter
    def TxRangeRate(self,val):
        """Sets the Tx rangerate which is used to compute the IF offsets"""
        with self.__TxRangeRate.get_lock():
            self.__TxRangeRate.value = np.double(val)

    @property
    def RxIFFreqOffset(self):
        #  averaged over last samples where packet was
        with self.__RxIFFreqOffset.get_lock():
            val = self.__RxIFFreqOffset.value
            self.__RxIFFreqOffset.value = 0
        return val

    @RxIFFreqOffset.setter
    def RxIFFreqOffset(self,val):
        with self.__RxIFFreqOffset.get_lock():
            self.__RxIFFreqOffset.value = np.double(val)

    @property
    def TxIFFreqOffset(self):
        # this value is estimated based on the Rx offset -- averaged over last samples where packet was
        with self.__TxIFFreqOffset.get_lock():
            val = self.__TxIFFreqOffset.value
            self.__TxIFFreqOffset.value = 0
        return val

    @TxIFFreqOffset.setter
    def TxIFFreqOffset(self,val):
        # this value is estimated based on the Rx offset
        with self.__TxIFFreqOffset.get_lock():
            self.__TxIFFreqOffset.value = np.double(val)

    @property
    def SNR(self):
        # averaged over last samples where packet was
        with self.__SNR.get_lock():
            val = self.__SNR.value
            self.__SNR.value = 0
        return val

    @SNR.setter
    def SNR(self,val):
        with self.__SNR.get_lock():
            self.__SNR.value = val

    @property
    def RxFreqOffset(self):
        # returns the raw frequency offset from self.Fc
        return self.__RxFreqOffset.value

    @RxFreqOffset.setter
    def RxFreqOffset(self,val):
        # sets the raw frequency offset from self.Fc
        with self.__RxFreqOffset.get_lock():
            self.__RxFreqOffset.value = val

    def SNRStats(self,snr,spSym):
        """
        This method is called in the process to update the SNR
        If a transmission is present, the SNR will only be based upon measurements from data
        Whether a transmission is data is based on the spSym estimate 
        """
        if self.__SNR.value == 0: # has been cleared by getter
            try:
                log.debug('SNR: {}'.format(self.__SNRArray))
            except:
                pass    
            self.__SNRArray = [snr]
            self.__SNRLastSpSym = spSym
        elif np.abs(spSym - 16) < TOLVAL:
            if np.abs(self.__SNRLastSpSym - 16) > TOLVAL and len(self.__SNRArray) == 1:
                self.__SNRArray[0] = snr
                self.__SNRLastSpSym = spSym
            else:
                self.__SNRArray.append(snr)
                
        self.SNR = sum(self.__SNRArray)/len(self.__SNRArray)

    @property
    def baudRateEst(self):
        return self.__baudRateEst.value

    @baudRateEst.setter
    def baudRateEst(self,val):
        with self.__baudRateEst.get_lock():
            self.__baudRateEst.value = val

        
    def freqOffsetEstStats(self,txIFOffset,rxIFOffset,rangerate,spSym,doppler_Hz):
        """
        This method is called in the process to update the frequency offsets (TxFreqOffset and RxFreqOffset)
        If a transmission is present, the frequency offsets will only be based upon measurements from data
        Whether a transmission is data is based on the spSym estimate 
        """
        baudRateEst = self.Fs/spSym
        if self.__TxIFFreqOffset.value == 0: # has been cleared by getter
            try:
                log.debug('TxIFFOffset: {}'.format(self.__TxIFFreqOffsetArray))
            except:
                pass    
            self.__TxIFFreqOffsetArray = [txIFOffset]
            self.__RxIFFreqOffsetArray = [rxIFOffset]
            self.__rangerateArray = [rangerate]
            self.__TxIFFreqOffsetLastSpSym = spSym
            self.__RxFreqOffsetArray = [doppler_Hz]
            self.__baudRateEstArray = [baudRateEst]
        elif np.abs(spSym - self.spSym) < TOLVAL:
            if np.abs(self.__TxIFFreqOffsetLastSpSym - self.spSym) > TOLVAL and len(self.__TxIFFreqOffsetArray) == 1:
                self.__TxIFFreqOffsetArray[0] = txIFOffset
                self.__RxIFFreqOffsetArray[0] = rxIFOffset
                self.__rangerateArray[0] = rangerate
                self.__TxIFFreqOffsetLastSpSym = spSym
                self.__RxFreqOffsetArray[0] = doppler_Hz
                self.__baudRateEstArray[0] = baudRateEst
            else:
                self.__TxIFFreqOffsetArray.append(txIFOffset)
                self.__RxIFFreqOffsetArray.append(rxIFOffset)
                self.__rangerateArray.append(rangerate)
                self.__RxFreqOffsetArray.append(doppler_Hz)
                self.__baudRateEstArray.append(baudRateEst)
                
        self.TxIFFOffset = sum(self.__TxIFFreqOffsetArray)/len(self.__TxIFFreqOffsetArray)
        self.RxIFFreqOffset = sum(self.__RxIFFreqOffsetArray)/len(self.__RxIFFreqOffsetArray)
        self.rangerate = sum(self.__rangerateArray)/len(self.__rangerateArray)
        self.RxFreqOffset = sum(self.__RxFreqOffsetArray)/len(self.__RxFreqOffsetArray) + self.baudRate*self.spSym/4
        self.baudRateEst = sum(self.__baudRateEstArray)/len(self.__baudRateEstArray)

    
        
            
if __name__ == "__main__":
    print("run /test/Rx/test_demodulator.py to test this class")
