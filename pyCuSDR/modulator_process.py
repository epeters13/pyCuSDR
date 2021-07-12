# Copyright: (c) 2021, Edwin G. W. Peters

from multiprocessing import Process, Event, Value, Lock
import numpy as np
import logging
import zmq
import time
import sys
import signal

from __global__ import *

from lib import *
from lib.freq_from_rangerate import *

import modulator

log	= logging.getLogger(LOG_NAME+'.'+__name__)

# log.setLevel(logging.DEBUG)

class Modulator_process(Process):

    timeOut_ms = 100

    def __init__(self,conf,protocol,radioName=''):
        Process.__init__(self)

        # sub-module log levels
        # modulator.log.setLevel(log.level) # set loglevel of modulator to same as class
        
        self.conf = conf
        self.protocol = protocol
        if radioName != '':
            self.name = radioName
            self.confRadio = conf['Radios']['Tx'][radioName]
        else:
            # backward compatibility
            self.name = protocol.name
            self.confRadio = conf['Radios']['Tx']
            
        self.__rangerate = Value('f',0)
        self.__Fc = Value('d',self.confRadio.get('frequency_Hz',1))
        # this is used to extract the Doppler from gpredict if gpredict has the center frequency set to something different. Is used to extract the Doppler component of the frequency provided
        self.__Fc_hl = self.confRadio.get('frequency_hamlib_Hz',self.__Fc.value)
        self.__Fs = Value('d',self.confRadio.get('samplesPerSym')*self.confRadio.get('baud'))
        self.__baudRate = Value('d',self.confRadio['baud'])

        self.__centreFreqOffset = Value('d',self.confRadio.get('centreFrequencyOffset',0.)) # receiver radio offset can be configured over RPC
            
        self.__freqOffset = Value('d',self.confRadio['frequencyOffset_Hz']) # GRC offset
        self.__totalFreqOffset = Value('f',0) # To read out over RPC. set by the modulator. freqOffset + centreFreqOffset + doppler 

        self.daemon = True
        self.runStatus = Event()
        self.runStatus.set() # set running == True

        log.info(f'[{self.name}]: Modulator initialized')

    def sigTermHandler(self,signum,frame):
        # We make SIGTERM raise an exception to exit the main thread if desired
        #raise SystemExit('SIGTERM received')
        pass
    
    def run(self):
        """
        runs the modulator
        """

        modulation_time = np.zeros(1000)
        modulation_time_index = 0
        N_modulation_time = len(modulation_time)
        
        time.sleep(.5) 



        log	= logging.getLogger(LOG_NAME+'.'+__name__)
                
        try:
            log.info(f'[{self.name}]: Starting modulator')
            if 'Interfaces' in self.confRadio and 'TxModToUSRPPort' in self.confRadio['Interfaces']:                
                TX_sock = self.confRadio['Interfaces']['TxModToUSRPPort']
            else:
                log.info(f'[{self.name}]: No specific TxModToUSRPPort found for this modulator -- Using default TxModToUSRPPort interface')
                TX_sock = self.conf['Interfaces']['Internal']['TxModToUSRPPort']
            ctx = zmq.Context()
            TXsock = ctx.socket(zmq.PUSH)
            log.warning(f'[{self.name}]: Opening GRC TX socket {TX_sock}')
            try:
                TXsock.bind(TX_sock)
            except Exception as e:
                log.error(f'[{self.name}]: Failed to open TX socket [{e}]')
                TXsock.close()
                raise

            byteDataIn = self.confRadio['Interfaces']['TxDataIn']
            byteDataIn_ZMQ = ctx.socket(zmq.PULL)
            byteDataIn_ZMQ.setsockopt(zmq.LINGER, 0)
            log.warning(f'[{self.name}]: Opening byte data in In socket {byteDataIn}')
            try:
                byteDataIn_ZMQ.bind(byteDataIn)
            except Exception as e:
                log.error(f'[{self.name}]: Failed to open byte data in socket [{e}]')
                TXsock.close()
                byteDataIn_ZMQ.close()
                raise

            try:
                TxManualIn = self.confRadio['Interfaces']['TxManualIn']
            except KeyError:
                log.warning(f'[{self.name}]: TxManualIn socket not found in config -- disabling manual transmissions')
                TxManualIn = None
                TxManualInSock = None

            if TxManualIn: # allows a second input socket. Easy for testing in parallel with the normal link manager
                TxManualInSock = ctx.socket(zmq.PULL)
                TxManualInSock.setsockopt(zmq.LINGER, 0)
                log.info(f'[{self.name}]: Opening manual transmit In socket {TxManualIn}')
                try:
                    TxManualInSock.bind(TxManualIn)
                except Exception as e:
                    log.error(f'[{self.name}]: Failed to open manual transmit socket [{e}], continuing execution')
                    TxManualInSock.close()
                    TxManualInSock = None
                    
            inDataPoller = zmq.Poller()
            inDataPoller.register(byteDataIn_ZMQ,zmq.POLLIN)
            if TxManualInSock:
                inDataPoller.register(TxManualInSock,zmq.POLLIN)
            
            modul = modulator.Modulator(self.conf, self.confRadio, self.protocol)
            orig_sigterm_handler = signal.getsignal(signal.SIGTERM)                
            signal.signal(signal.SIGTERM, self.sigTermHandler) # handle SIGTERM
            while self.runStatus.is_set():
                try:
                    tm = time.time()
                    evts = dict(inDataPoller.poll(self.timeOut_ms))
                    with self.__rangerate.get_lock():
                        rr = self.__rangerate.value
                        modul.set_rangerate(self.__rangerate.value)
                    modul.TxCentreFreqOffset = self.__centreFreqOffset.value
                    # modul.noFlags = self.__numSyncFlags # change the number of sync flags
                    if len(evts) > 0:
                        if log.level <= logging.DEBUG:
                            log.debug(f"[{self.name}]: evts {str(evts)}")
                            log.debug(f"[{self.name}]: link {str(byteDataIn_ZMQ)}")
                            log.debug(f"[{self.name}]: manual {str(TxManualInSock)}")
                        t1 = time.time()
                        if byteDataIn_ZMQ in evts and evts[byteDataIn_ZMQ] == zmq.POLLIN:
                            rawTxData = byteDataIn_ZMQ.recv()
                        elif TxManualInSock and TxManualInSock in evts and evts[TxManualInSock] == zmq.POLLIN:
                            rawTxData = TxManualInSock.recv()
                        else:
                            log.error(f'[{self.name}]: Error: polling socket not found')
                            rawTxData = None
                        t2 = time.time()
                        if rawTxData:
                            TxData = np.frombuffer(rawTxData,dtype=np.uint8)

                            if log.level >= logging.WARNING:
                                log.warning(f'[{self.name}]: RangeRate {rr} m/s (Doppler {rr/3e8*self.Fc} Hz) frequency offset {self.centreFreqOffset} Hz\tTransmitting {len(TxData)} bytes')
                            else:
                                log.warning(f'[{self.name}]: RangeRate {rr} m/s (Doppler {rr/3e8*self.Fc} Hz) frequency offset {self.centreFreqOffset} Hz\tTransmitting {len(TxData)} bytes {bytesToHex(TxData)}')
                            
                            t3 = time.time()
                            sigMod = modul.encodeAndModulate(TxData)
                            t4 = time.time()
                            # log.warning(f'Time to make waveform {(time.time()-tm)*1000} ms')

                            if SAVETX_DATA:
                                fName = f'{self.name}_TxPacket'
                                log.warning(f'saving Tx Packet to {fName}')
                                np.save(fName,sigMod.astype(MODULATORDTYPE))
                            try:                            
                                TXsock.send(sigMod.astype(MODULATORDTYPE),zmq.NOBLOCK)
                            except zmq.error.Again:
                                log.warning(f"[{self.name}]: timeout while sending transmission to GNU Radio")
                            t5 = time.time()
                        # log.warning(f'Time to make waveform and transmit {(time.time()-tm)*1000} ms')
                        modulation_time[modulation_time_index] = time.time()-t1
                        log.warning(f'time modulation {modulation_time[modulation_time_index]:.6f} s avg {N_modulation_time} transmissions: {np.mean(modulation_time):.6f} s')
                        modulation_time_index +=1
                        if modulation_time_index >= N_modulation_time:
                            modulation_time_index = 0

                    # Get some data from the modulator for monitoring -- these can be fetched on RPC
                    with self.__Fc.get_lock():
                        self.__Fc.value = modul.Fc - modul.TxFreqOffset
                    with self.__Fs.get_lock():
                        self.__Fs.value = modul.get_samp_rate()
                    with self.__freqOffset.get_lock():
                        self.__freqOffset.value = modul.TxFreqOffset
                    with self.__totalFreqOffset.get_lock():
                        self.__totalFreqOffset.value = modul.TxTotalFreqOffset
                    with  self.__baudRate.get_lock():
                        self.__baudRate.value = modul.baudRate

                except modulator.DataLengthError as e:
                    log.error(f'ERROR: Data length does not satisfy requirements. Message:\n{e}')
                    
                
        except Exception as e:
            log.exception(e)
        finally:
            time.sleep(2)
            TXsock.close()
            byteDataIn_ZMQ.close()
        log.info('Process finished -- Bye')
        signal.signal(signal.SIGTERM, orig_sigterm_handler)  # restore signal, such that process.terminate() can kill this

            
    # Setters and getters
    def stop(self):
        log.info('Received request to stop')
        self.runStatus.clear()
        
    @property
    def Fs(self):
        return self.__Fs.value

    @Fs.setter
    def Fs(self,Fs):
        # raise Exception('Setting Fs not supported at this stage')
        with self.__FS.get_lock():
           self.__Fs.value = Fs

    @property
    def baudRate(self):
        return self.__baudRate.value
        
    @property
    def rangerate(self):
        return self.__rangerate.value

    @rangerate.setter
    def rangerate(self,rangerate):
        # Set by orbit tracker to precompensate the uplink Doppler
        log.debug('setting rangerate {} m/s'.format(rangerate))
        with self.__rangerate.get_lock():
            self.__rangerate.value = np.double(rangerate)

    @property
    def Fc(self):
        return self.__Fc.value

    @Fc.setter
    def Fc(self,Fc):
        raise NotImplementedError('Setting Fc not supported at this stage')
        #with self.__Fc.get_lock():
        #    self.__Fc.value = np.double(Fc)

    @property
    def centreFreqOffset(self):
        """Read the fixed frequency offset for the radio. Used to adjust offsets compared to the receiver radio"""
        return self.__centreFreqOffset.value

    @centreFreqOffset.setter
    def centreFreqOffset(self,fo):
        """Set a fixed frequency offset for the radio. Used to adjust offsets compared to the receiver radio"""
        with self.__centreFreqOffset.get_lock():
            self.__centreFreqOffset.value = int(fo)
    
    @property
    def freqOffset(self):
        """The frequency offset that is used to offset the DC level in GNU radio. This is added here and subtracted in GNU radio"""
        return self.__freqOffset.value 

    @freqOffset.setter
    def freqOffset(self,val):
        """The frequency offset that is used to offset the DC level in GNU radio. This is added here and subtracted in GNU radio"""
        with self.__freqOffset.get_lock():
            self.__freqOffset.value = int(val)
            
    @property
    def totalFreqOffset(self):
        """Returns the freqOffset + centreFreqOffset + doppler"""
        return self.__totalFreqOffset.value

    @property
    def doppler(self):
        return freq_from_rangerate(self.__rangerate.value,self.__Fc_hl)

    @property
    def freq_hl(self):
        """
        Hamlib frequency setter
        """
        return self.__Fc_hl + self.doppler

    @freq_hl.setter
    def freq_hl(self,val):
        self.rangerate = rangerate_from_freq(val,self.__Fc_hl)


