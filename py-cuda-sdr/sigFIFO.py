# Copyright: (c) 2021, Edwin G. W. Peters

from __global__ import *
import sys
import zmq
import numpy as np
import json
import logging

log	= logging.getLogger(LOG_NAME+'.'+__name__)
# log.setLevel(logging.INFO)

class RingBuffer():
    """
    Implements a ringbuffer to store data until needed 
    """

    headIdx = 0
    tailIdx = 0
    currentBufSize = 0
    bufLen = 2**15
    outLen = 2**14-2**10
    dtype = None
    buf = []

    def __init__(self,outLen,bufLen=None,dtype=np.complex64):
        
        self.outLen = outLen
        if bufLen == None:
            self.bufLen = 10*outLen
        else:
            if bufLen < outLen:
                raise IndexError('bufLen < outLen','Buffer size too small for expected output size')
            self.bufLen = bufLen
            
        self.dtype = dtype
        self.buf = np.empty(self.bufLen,dtype=self.dtype)

    
        
    def insert(self,data):
        """
        Insert a block of data at the end of the buffer.
        We flush the buffer in case of a overflow
        """
        
        if data.dtype != self.dtype:
            log.error('wrong datatype. Expected %s' %(str(self.dtype)))
            data = data.astype(self.dtype)
        N = len(data)
        bufEnd = N + self.headIdx

        if self.currentBufSize + N > self.bufLen:
            # buffer full
            log.error('buffer full: Flush')
            self.flush()
        
        if bufEnd > self.bufLen:
            Nmid = self.bufLen-self.headIdx
            self.buf[self.headIdx:] = data[:Nmid]
            self.headIdx = N-Nmid
            self.buf[:self.headIdx] = data[Nmid:]
        else:
            self.buf[self.headIdx:bufEnd] = data
            self.headIdx = bufEnd

        self.currentBufSize += N
        return self.currentBufSize
            
    def popBlock(self,noSamples):
        """
        Pops and returns noSamples from the buffer
        """
        if self.currentBufSize < noSamples:
            return []           # We only want to pop if there is enough data

        popEnd = self.tailIdx + noSamples
        data = np.empty(noSamples,dtype=self.dtype)
        
        if popEnd > self.bufLen:
            Nmid = self.bufLen - self.tailIdx
            data[:Nmid] = self.buf[-Nmid:]
            self.tailIdx = noSamples - Nmid
            data[Nmid:] = self.buf[:self.tailIdx]
        else:
            tailIdxNew = self.tailIdx + noSamples
            data = self.buf[self.tailIdx:tailIdxNew]
            if tailIdxNew == self.bufLen:
                self.tailIdx = 0
            else:
                self.tailIdx = tailIdxNew

        self.currentBufSize -= noSamples
        return data


    def flush(self):
        """
        Clear the buffer
        """
        self.headIdx = 0
        self.tailIdx = 0
        self.currentBufSize = 0




class SigFIFO():
    blockSize = None
    dtype = None
    
    # GRCtimePerBlock = 0.026 # 32760/8/9600/16 from gnu radio
    # timeoutSec = 10         # timeout in seconds
    # timeoutLim =  int(np.ceil(timeoutSec/GRCtimePerBlock))

    def __init__(self, socket, reqDataSize, dtype = np.complex64, timeOut_ms = 1000, exitOnTimeout = False, enableTimeoutException = False,timeoutRetries = 120,runStatus = None):

        self.blockSize = reqDataSize
        self.dtype = dtype
        self.timeoutRetries = timeoutRetries # 120 sec in case both USRPs need flashing
        self.runStatus = runStatus
        
        log.debug('Configuring RX socket:' + socket)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

        try:
            self.socket.connect(socket)
            self.socket.setsockopt_string(zmq.SUBSCRIBE,'') # subscribe to all data in the socket
            self.poller = zmq.Poller()
            self.poller.register(self.socket,zmq.POLLIN)
        except Exception as e:
            log.error('Exception occurred when connecting to RX socket. Message: ')
            log.exception(e)
            
        self.buf = RingBuffer(self.blockSize, bufLen = self.blockSize*2, dtype = dtype)

        self.timeOut_ms = timeOut_ms
        self.exitOnTimeout = exitOnTimeout
        self.raiseExceptionOnTimeout = enableTimeoutException
        
    def __del__(self):
        self.socket.close()

        
    def getBlock(self):
        """
        get a block of data
        stopCondition can be provided, which can be used to stop the loop
        """
            
        data = []
        # timeoutCount = 0
        timeoutCount = 0
        while len(data) == 0:
            # Gnu radio provides data in blocks of around 4095-4096 samples.
            evts = self.poller.poll(self.timeOut_ms)
            if len(evts)>0:
                timeoutCount = 0
                rawBytes = self.socket.recv()
                # Fill in the ringbuffer until we have enough to continue processing
                self.buf.insert(np.frombuffer(rawBytes, dtype=self.dtype))
            else:
                log.debug('ZMQ poll timed out')
                timeoutCount += 1

                if self.runStatus:
                    if not self.runStatus.is_set():
                        raise TimeoutError('Terminated')
                if timeoutCount > self.timeoutRetries:
                    if self.raiseExceptionOnTimeout == True:
                        raise TimeoutError('ZMQ poll timed out')
                    
                    if self.exitOnTimeout == True:
                        log.info('Exiting....')
                        sys.exit()
            # Try to get a block of data. If not possible, we try again
            data = self.buf.popBlock(self.blockSize)
            
        return data

    
