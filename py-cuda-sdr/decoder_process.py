# Copyright: (c) 2021, Edwin G. W. Peters

"""
Contains "Decoder" module and "VisualizerData"

"VisualizerData" stores statistics that can be retrieved for plotting using the decoder.getVisualData() method

Refer to "keys" in "VisualizerData" to specify which variables are safed
"""


__author__ = "Edwin Peters"

from __global__ import *

from multiprocessing import Process, Event, Queue
from collections import deque
import hashlib
import numpy as np
import logging
import zmq
import time
import sys
import signal
try:
    from telegraf.client import TelegrafClient
    TELEGRAPH_AVAILABLE = True
except ModuleNotFoundError:
    TELEGRAPH_AVAILABLE = False

import decoder

log = logging.getLogger(LOG_NAME+'.'+__name__)

PACKETFILTER_TIMEOUT = 5

class Decoder(Process):

    pollTimeout = 1000 # ms
    workerData = {}
    testCnt = 0
        
    def __init__(self,conf,protocol):
        Process.__init__(self)
        # set sub module loglevel similar to this
        decoder.log.setLevel(log.level)
        self.name = 'Decoder'
        self.conf = conf
        self.protocol = protocol # if multiple protocols are used, this is a dictionary

        # log.info(f"----------decoder protocol== dict? {isinstance(self.protocol,dict)}, {'decodeBytesOut_ZMQ' in conf['Interfaces']['External'].keys()}")
        if 'decodeBytesOut_ZMQ' in conf['Interfaces']['External'].keys(): # do specific output adresses exist in this config?
            # log.info('-------------------In interfaces')
            self.decodeBytesOutAddr_ZMQ = dict()
            for k in self.protocol.keys():
                if k in conf['Interfaces']['External']['decodeBytesOut_ZMQ']:
                    self.decodeBytesOutAddr_ZMQ[k] = conf['Interfaces']['External']['decodeBytesOut_ZMQ'][k]
                else:
                    raise ValueError(f"All 'decodeBytesOut_ZMQ' adresses have to be defined in 'interfaces'/'decodeBytesOut_ZMQ' in config")
                
            # log.info(f'----------------------Decode out addresses are {self.decodeBytesOutAddr_ZMQ}')
            self.decodeBytesOut_ZMQ = None # for the initialisation later on
        else:
            raise ValueError("'decodeBytesOut_ZMQ' not specified")    
            
        self.decodeInAddr = conf['Interfaces']['Internal']['decodeIn']  # everything comes in to the same socket
        self.visualLogBufferSize = conf['Main']['plotBufferSize'] 

        
        self.daemon = True
        self.runStatus = Event()
        self.runStatus.set() # running

        if log.level < 20:
            # show warnings for Visualizer data when an attribute is not found
            self.showWarnings = True
        else:
            self.showWarnings = False
            
        log.debug('Decoder configured')

        # this socket is used when the getVisualData method is used to retrieve data
        ctx = zmq.Context()
        self.workerSock = ctx.socket(zmq.PULL)
        self.workerSock.connect('tcp://localhost:11001')

   
    def sigTermHandler(self,signum,frame):
        # We make SIGTERM raise an exception to exit the main thread if desired
        #raise SystemExit('SIGTERM received')
        pass
        
    def stop(self):
        """
        Stops the process in a clean way
        """
        log.info('PID {} -- Received request to stop'.format(self.pid))
        self.runStatus.clear()


    def inspectPacket(self,byteData):

        if self.protocol.name == 'buccaneer':
            if byteData[0] == 0x31: # beacon
                # TAP-B (beacon)
                lastRSSI = byteData[173]
                log.info('last RSSI: -%.2f dB',lastRSSI)

        elif self.protocol.name == 'magpie':
            if byteData[0] == 0x31: # beacon
                lastRSSI = byteData[169]
                cntRxPackets = np.sum(byteData[175:171:-1]*16**np.array([3,2,1,0]))
                cntTxPackets = np.sum(byteData[179:175:-1]*16**np.array([3,2,1,0]))
                uhfBytesReceived = np.sum(byteData[85:81:-1]*16**np.array([3,2,1,0]))
                uhfBytesTransmitted = np.sum(byteData[89:85:-1]*16**np.array([3,2,1,0]))

                log.info('last RSSI -{} dB\t Rx packets {}\tTx packets {}\tRx bytes {}\tTx bytes {}'.format(lastRSSI,cntRxPackets,cntTxPackets,uhfBytesReceived,uhfBytesTransmitted))
                

        return lastRSSI
            
    def run(self):
        """
        Wait for bitstreams from trustprocessor or demodulator
        find packets and push out on socket. We're finished at that point
        """
        log	= logging.getLogger(LOG_NAME+'.'+__name__)

        ctx = zmq.Context()
        log.warning('Configuring decoder input socket: ' + self.decodeInAddr)
        try:
            decodeIn = ctx.socket(zmq.PULL)
            decodeIn.bind(self.decodeInAddr)
        except Exception as e:
            log.error('Error while configuring decoder input socket')
            log.exception(e)
            return
        decodePol = zmq.Poller()
        decodePol.register(decodeIn,zmq.POLLIN)


        decodeOutZMQPorts = dict()
        for k in self.protocol.keys():
            try:
                log.warning(f'Configuring ZMQ PUSH output socket for {k} on {self.decodeBytesOutAddr_ZMQ[k]}')
                decodeOutZMQPorts[k] = ctx.socket(zmq.PUSH)
                decodeOutZMQPorts[k].setsockopt(zmq.LINGER, 0) # set the retry amount
                decodeOutZMQPorts[k].bind(self.decodeBytesOutAddr_ZMQ[k])
            except Exception as e:
                log.error('Error while configuring output sockets')
                log.exception(e)
                decodeIn.close()
                del decodePol
                return

            
                    
        decoders = {}

        # List of hashes of previously received packets. Useful to prevent duplicate versions of the same packet to be received when using mulitple antennas
        global PACKETFILTER_TIMEOUT 
        PACKETFILTER_TIMEOUT = self.conf['decoder'].get('packetCheckHistTimeout',0) # default 0 -- disabled
        if PACKETFILTER_TIMEOUT == 0:
            queueLen = 0
        else:
            queueLen = self.conf['decoder']['packetCheckHist']
        hashHist = PacketHist(queueLen)

        # setup telegraph monitoring
        if TELEGRAPH_AVAILABLE:
            try:
                telegraf_host = self.conf['Main']['telegraf_ip']
                telegraf_port = int(self.conf['Main']['telegraf_port'])
                log.info(f"Logging telegraf on {telegraf_host}:{telegraf_port}")
                telegraf = TelegrafClient(host=telegraf_host,port=telegraf_port)
            except KeyError:
                log.error("Not configuring telegraf since 'telegraf_ip' not found in config")
                telegraf = None
            except Exception:
                log.exception("Unable to start telegraf:")
                telegraf = None
        else:
            telegraf = None
                
        # just a dummy initialization such that errors are thrown as soon as the software is started
        try:
            if isinstance(self.protocol,dict):
                for prot in self.protocol:
                    log.info(f'protocol {prot} name {self.protocol[prot].name}')
                    dec = decoder.Decoder(self.conf['decoder'],self.protocol[prot])
            else:
                dec = decoder.Decoder(self.conf['decoder'],self.protocol)
        except Exception as e:
            log.error('Error while initializing decoder')
            log.exception(e)
            return
        del dec # clean it up again -- these are initialized on demand

        orig_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self.sigTermHandler)
        log.info('Decoder process initialized and running')
        try:
            while self.runStatus.is_set():
                try:
                    log.debug('polling')
                    socks = decodePol.poll(self.pollTimeout) #  wait for input data
                    if len(socks) > 0 and socks[0][1] == zmq.POLLIN:
                        log.debug('receiving')
                        dataCont = decodeIn.recv_pyobj(zmq.DONTWAIT)

                        self.testCnt += 1
                        workerId = dataCont['workerId']
                        if not workerId in decoders.keys():
                            try:
                                if isinstance(self.protocol,dict):
                                    log.info(f"Adding new worker {workerId}. DataCont: {dataCont['protocol']}")
                                    decoders[workerId] = decoder.Decoder(self.conf['decoder'],self.protocol[dataCont['protocol']])
                                else:
                                    decoders[workerId] = decoder.Decoder(self.conf['decoder'],self.protocol)
                                self.workerData[workerId] = VisualizerData(workerId,self.visualLogBufferSize,showWarnings = self.showWarnings)

                            except Exception as e:
                                log.error('Error while initializing decoder for worker %s' %(workerId))
                                log.exception(e)

                            
                        # rawBits = np.array(dataCont['data'],dtype=DATATYPE)
                        rawBits = dataCont['data']
                        # rawTrust = np.array(dataCont['trust'],dtype=TRUSTTYPE)
                        # numClippedPeaks = np.sum(rawTrust < -.8)
                        t = time.time()                        
                        packets, bits_DS, numSyncSig = decoders[workerId].findFrames(rawBits, 0)
                        # packets, bits_DS, numSyncSig = dec.findFrames(rawBits, 0)
                        pktsErr, pktsSuc = 0,0
                        # log.debug('Found %2d packets' % len(packets))
                        if len(packets) > 0:
                            log.warning('worker {:10}\tbits processed {:6}\tsyncSigs {}\tpackets {}\ttime {:.2f} ms'.format(workerId,len(rawBits),numSyncSig,len(packets),(time.time()-t)*1000))
                           
                            for packet in packets:
                                tb = time.time()
                                try:
                                    byteData, noError, correctBytes = packet.getBinaryData()
                                except Exception as e:
                                    log.exception(e)
                                    log.error(f'length raw bits: {len(packet.bitsRaw)}')
                                    raise e
                                if noError < 0:
                                    pktsErr += 1
                                else:
                                    pktsSuc += 1

                                # log.warning(f'decoding took {(time.time()-tb)}ms')
                                # tb = time.time()
                                rawBytes = packet.getBinaryRawData() # non destuffed
                                if log.level <= logging.INFO:
                                    if len(decoders) > 1:
                                        log.warning('worker %s\tfound packet %s\tBit errors: %d\tnum voters: %d\tchannels used: %s' %(workerId,
                                                                                                                                      packet.getAsciiAddress(),
                                                                                                                                      noError,
                                                                                                                                      safe_get_from_list(dataCont,'numSlaves',0),
                                                                                                                                      str(safe_get_from_list(dataCont,'slaveNames',[]))))
                                    else:
                                        log.warning('worker %s\tfound packet %s\tBit errors: %d' %(workerId,
                                                                                                   packet.getAsciiAddress(),
                                                                                                   noError))

                                if log.level <= logging.INFO:
                                    packet.printPacket(pre_str=f'worker {workerId}, SNR {dataCont["SNR"]} dB, freq offset {dataCont["doppler"]} Hz.',workerId=workerId,verbosity = log.level)
                               
                                # log.warning(f'printing and decoding took {(time.time()-tb)*1000}ms')
                                if noError > -1:
                                    try:
                                        if len(correctBytes) > 0:
                                            newPacket = hashHist.checkHash(correctBytes,len(decoders))
                                        else:
                                            log.info('worker {}\tPacket length = 0.. skipping'.format(workerId))
                                            newPacket = False # ignore length 0 packets
                                    except Exception as e:
                                        log.exception(e)
                                        log.info('bytes' + str(correctBytes))
                                        raise e
                                else:
                                    log.info('worker {}\tpacket Error, sending anyway'.format(workerId))
                                    newPacket = True # just send failed packets anyway.
                                try:
                                    lastRSSI = self.inspectPacket(correctBytes)
                                    
                                except Exception:
                                    lastRSSI = 0

                                if (newPacket and not BENCHMARK_MODE) or (BENCHMARK_MODE and 'UHF-V' in workerId): # In benchmark mode
                                    try:
                                        decodeOutZMQPorts[dataCont['protocol']].send(byteData,zmq.NOBLOCK)
                                        log.info(f"worker {workerId}\tsent to ZMQ on protocol {dataCont['protocol']}")
                                    except zmq.error.Again as e:
                                        log.error('worker {}\tfailed to send data to ZMQ. Message:  [{}]'.format(workerId,e))
                                else:
                                    log.info('worker {}\tNot sending to ZMQ -- Previous copy of the packet has already been sent'.format(workerId))
                        dataCont['packetFail'] = pktsErr
                        dataCont['packetSuc'] = pktsSuc
                        dataCont['numSyncSig'] = numSyncSig/len(rawBits)*dataCont['baudRate']
                        dataCont['numBits'] = len(rawBits)
                        # if workerId not in self.workerData.keys():
                        #     self.workerData[workerId] = VisualizerData(workerId,dataCont.keys(),self.visualLogBufferSize)

                        self.appendLogData(dataCont)

                        ## telegraf logging
                        if telegraf:
                            #log.warning('telegraf logging')
                            t = time.time()
                            vals = dataCont.copy() # everything is in here                 
                            time_stamp = vals.pop('timestamp')
                            vals.pop('data')
                            try:
                                vals.pop('trust')
                            except:
                                pass
                            try:
                                vals.pop('slaveNames')
                            except:
                                pass
                            vals['packets_decoded'] = vals['packetFail'] + vals['packetSuc']
                            
                            tags = {
                                'workerId' : vals.pop('workerId'),
                                'voteGroup' : vals.pop('voteGroup'),
                                'protocol' : vals.pop('protocol')
                            }
                            
                            try:

                                telegraf.metric("mon.modem", vals, tags=tags, timestamp=int(time_stamp * 1e9))
                                
                                # log.warning(f'----------Printed to telegraf tags {tags} vals {vals} -- took {time.time()-t} s')    
                            except Exception:
                                log.exception("Unable to log to telegraf!")
                                raise
                        
                    else:
                        pass
                        # timeout
                        #log.debug('TimeOut waiting for data')
                except Exception as e:
                    log.error('Exception occured for worker %s', dataCont['workerId'])
                    log.error('len data %d', len(dataCont['data']))
                    log.error('number of decoders %d', len(decoders))
                    log.exception(e)
                    raise
        except SystemExit:
            pass
        except Exception as e:
            log.exception(e)

        finally:
            if self.decodeBytesOut_ZMQ is not None:
                 decodeOutZMQPorts.close()
            else: # all sockets need to be closed in this case
                [ decodeOutZMQPorts[k].close() for k in  decodeOutZMQPorts.keys()] 
            decodeIn.close()
            ctx = zmq.Context()
            workerRet = ctx.socket(zmq.PUSH)
            workerRet.bind('tcp://*:11001')

            log.info('Preparing data')
            for w in self.workerData.values():
                workerRet.send_pyobj(w.getData())
            log.info('Finished sending data')
                        
            workerRet.close()

            del decodePol
            log.info('Finished')
            for d in decoders:
                del d
            signal.signal(signal.SIGTERM, orig_sigterm_handler) # restore signal, such that process.terminate() can kill this
            sys.stdout.flush()

                
    def checkAppendLogData(self,dataCont):
        if self.visualLogBufferSize > 0:
            self.appendLogData(dataCont)

    def appendLogData(self,dataCont):
        self.workerData[dataCont['workerId']].addData(dataCont)


    def getVisualData(self):
        """
        Get statistical data to do analysis
        """
        data = []
        poller = zmq.Poller()
        poller.register(self.workerSock,zmq.POLLIN)
        log.info('Waiting for data')
        while True:
           
            evts = poller.poll(2000)
            if len(evts)>0 and evts[0][1] == zmq.POLLIN:
                data.append(self.workerSock.recv_pyobj())
                log.info('Received data')
            else:
                log.info('Timed out waiting for data')
                break
        log.info('finished loop')
        self.workerSock.close()
        log.info('socket closed')
        return data
    
    
class VisualizerData():
    """
    This class allows storage of all kind of visualizer data. 
    One object can be created for each worker.
    
    The class is fail-safe such that if a field does not exists in the input data,
    it is not written. An optional warning can be printed
    """

    keys = ['timestamp','count','doppler','doppler_std','spSymEst','SNR','numSyncSig','packetSuc','numBits','packetFail','baudRate','numSlaves']
    
    def __init__(self, workerId, bufferSize, showWarnings = True):
        self.data = {}
        self.data['workerId'] = workerId
        for f in self.keys:
            self.data[f] = np.zeros(bufferSize)

        self.idx = 0
        self.bufferSize = bufferSize
        self.workerId = workerId
        self.showWarnings = showWarnings


    def safeAddArray(self,key,dataCont,nValues = 1):
        """
        Add lists if key exists
        """
        if key in dataCont.keys():
            # log.debug('key {}: {}'.format(key,dataCont[key]))
            try:
                self.data[key][self.idx:self.idx + nValues] = dataCont[key][self.idx:self.idx + nValues]
            except Exception as e:
                log.error('Key %s',key)
                log.exception(e)
                raise e
        else:
            if self.showWarnings:
                log.warning('Key %s not found for worker %s' %(key,dataCont['workerId']))

    def safeAddValue(self,key,dataCont,nValues = 1):
        """
        Add single scalar values if key exists
        """
        if key in dataCont.keys():
            # log.debug('key {}: {}'.format(key,dataCont[key]))
            try:
                self.data[key][self.idx:self.idx + nValues] = dataCont[key]
            except Exception as e:
                log.error('Key %s',key)
                log.exception(e)
                raise e
        else:
            if self.showWarnings:
                log.warning('Key %s not found for worker %s' %(key,dataCont['workerId']))

    def safeAdd(self,key,dataCont,nValues = 1):
        """
        Calls the corresponding add method depending on whether the data is in a list or scalar
        """
        if key in dataCont.keys():
            # log.debug('key {}: {}'.format(key,dataCont[key]))
            if isinstance(dataCont[key],list):
                self.safeAddArray(key,dataCont,nValues = nValues)
            else:
                self.safeAddValue(key,dataCont,nValues = nValues)
        
        else:
            if self.showWarnings:
                log.warning('Key %s not found for worker %s' %(key,dataCont['workerId']))
            

    def addData(self,dataCont):
        """
        Add data
        if timestamp is a scalar, everything is
        if timestamp is a list, all imported values are a list of the same length. We need to append all data. The values produced in trustprocessor and decoder are scalar. These have to be duplicated 
        """
        if isinstance(dataCont['timestamp'],float):
            nValues = 1
        else:
            nValues = len(dataCont['timestamp'])
            if self.idx + nValues > self.bufferSize:
                nValues = self.bufferSize
            
        # log.info(dataCont.keys())
        if self.idx < self.bufferSize:
            for k in self.keys:
                self.safeAdd(k,dataCont,nValues = nValues)
        self.idx += nValues

        
    def getData(self):
        out = {}
        out['workerId'] = self.workerId
        for k in self.keys:
            out[k] = self.data[k][:self.idx].tolist()
        return out

        
class PacketHist():
    """
    Stores a fixed length list of hashes of recently received packets. 
    init:
       Needs max queue length
    checkHash:
       Input: data (after RS decoding) and number of workers
                If number of workers == 1 we save time by not searching the queue
       Output: True if the packet is not in the queue
               False if the packet is in the queue
    """

    def __init__(self,queueLen):
        self.dq = deque([],queueLen) # packets
        self.tq = deque([],queueLen) # time
        self.hasher = hashlib.md5
        
    def checkHash(self,data,numWorkers = 1):
        if PACKETFILTER_TIMEOUT == 0:
            # duplicate checking disabled
            return True
        dataHash = self.hasher(data).hexdigest()

        if numWorkers == 1:
            self.dq.appendleft(dataHash)
            self.tq.appendleft(time.time())
            return True
        else:
            if dataHash in self.dq:
                idx = self.dq.index(dataHash)
                if time.time()-self.tq[idx] > PACKETFILTER_TIMEOUT:
                    # put a copy of the hash in the front of the queue 
                    self.dq.appendleft(dataHash)
                    self.tq.appendleft(time.time()) # update time
                    log.info('packet already received, but too long ago ({:.3f} s)'.format(time.time()-self.tq[idx]))
                    return True
                # data already received
                log.debug('Packet already received -- discarding')
                return False
            else:
                self.dq.appendleft(dataHash)
                self.tq.appendleft(time.time())
                return True
            

        
