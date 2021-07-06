# Copyright: (c) 2021, Edwin G. W. Peters

"""
the softCombiner contains the modules 
Worker -- Stores demodulated data from workers and keeps track of indexing
softCombiner -- Collects data from multiple workers and votes if data align

Test scripts located in test/test_softCombiner

This module does not utilize precise timing to align the data. This makes it useful across dispersed ground sites and unsynchronized and disciplined clocks.

This moule is a basic working proof of concept.
Currently, all channels are compared against each other and all combinations are returned. This creates N copies of each bit stream, which then are sorted and culled in the decoder. The advantage is that when the vote for a bit is equal, the weight is set towards the 'master' channel, thus we get both options through the decoder. The FEC should then take care of the rest.

Another option is to only use one channel as reference and the other 2 as voting. However, this foregoes some of the robustness for channels dropping out and sample loss. Especially when only two channels remain and the vote is equal (when not using confidence)

An option should be added to tage the data such that the decoder immediately can see all three versions, and based on FEC can verify the integrity
"""

from __global__ import *

from multiprocessing import Process, Event, Queue
import numpy as np
import logging
import zmq
import time
import sys
import signal
import itertools
import queue # just for queue.Empty exception


from lib.customXCorr import customXCorr

log	= logging.getLogger(LOG_NAME+'.'+__name__)

MAX_DATA_LEN_BEFORE_TRANSMIT  = 6000

# values to indicate votes when bits disagree 
BOTH_TRUST_ERR = 0.1 # both trust but disagree
MASTER_TRUST = 0.7
SLAVE_TRUST = 0.3
BOTH_DISTRUST = -1




class WorkerIdError(AssertionError):
    def __init__(self,*args,**kwargs):
        AssertionError.__init__(self,*args,**kwargs)


SLAVE_TIMEOUT = 5 # seconds
class Slave():
    """
    Keeps track of the indices for the slave workers when one on one compare is done in pairs
    Currently not used
    """

    def __init__(self,slaveId,head,tail):
        self.workerId = slaveId
        self.count = 0
        self.head = head
        self.tail = tail
        self.timestamp = time.time()

    def updateIdx(self,newHead):
        if time.time()-self.timestamp > SLAVE_TIMEOUT:
            # this slave is too old. Remove it
            return -1
        # update the indices after old data has been removed
        if self.head < newHead:
            log.warning('slave %s -- removing more data than has been processed',self.workerId)
            self.head = 0
            self.tail -= newHead
        else:
            self.head -= newHead
            self.tail -= newHead

        return 0

    def updateTail(self,idx):
        self.tail = idx
             
    def resetHeadTail(self):
        # to flush the indices
        self.head = 0
        self.tail = 0

        
        
class Worker():
    """
    This class holds the data for one worker. It takes care of the flushing and aligning of data
    """
    
    keyNames = ['count','timestamp','voteGroup','doppler','doppler_std','spSymEst','SNR','baudRate','protocol']
    keyDataTypes = {'count' : int,
                    'timestamp' : float,
                    'voteGroup' : int,
                    'doppler' : float,
                    'doppler_std' : float,
                    'spSymEst' : float,
                    'SNR' : float,
                    'TxRangeRate': float,
                    'baudRate': int,
                    'protocol': str}
    
    arrayKeyNames = ['data','trust']
    arrayDataTypes = {'data' : DATATYPE,
                      'trust' : TRUSTTYPE}
    
    
    def __init__(self,workerData,timestampTimeOut = .5, showWarnings = False):
        """
        Have to make some queue with the latest containers/JSON data received such that we can ensure that everything is processed
        When worker is initialized, we have to store the latest container and
        workerId,
        count, 
        timestamp,
        data (uint8),
        trust (float)
        timestampTimeout (default 0.2)
        """

        self.showWarnings = showWarnings
        self.slaves = []

        # Counters
        self.getCount = 0 # increments when data is sent (by calling clear requestcounter)
        self.totalRequestCount = 0 # total times data is requested
        self._dataRequestCounter = 0 # increments every time data is requested. Gets reset by softCombiner every time data is sent for decoding. read by getDataRequestCounter and clear with clearDataRequestCounter

        self.arrivalTimes = []
        self.arrivalTimes.append({'time' : time.time(),
                                  'idx' : 0})
        self.data = data = {}
        
        self.workerId = str(workerData['workerId'])
        self.timestamp = time.time() # for comparing objects
        
        for key in self.keyNames:
            self.safeAdd(key,workerData)

        for key in self.arrayKeyNames:
            self.data[key] = np.array([],dtype = self.arrayDataTypes[key])
            self.safeAppend(key,workerData)

        try:
            self.voteGroup = self.data['voteGroup'] # determines which slaves will be compared together
        except KeyError:
            log.warning("{}: 'voteGroup' not defined -- assigning 0".format(self.workerId))
            self.voteGroup = 0

            
        assert len(self.data['data']) == len(self.data['trust']), "Data and trust vectors have different lengths. The demodulator should have taken care of this"

        # keep track of which data of this worker has been sent to the decoder
        self.head = 0
        self.tail = len(self.data['data']) 

        self.timestampTimeOut = timestampTimeOut
        self.showWarnings = showWarnings
        self._arrivalTimestamp = time.time()
        log.debug('worker %s len data %d, len trust  %d', self.workerId, len(self.data['data']),len(self.data['trust']))


        self.slaves = [] # workers to compare with
        self.activeSlave = None # the current active one

    def clearDataRequestCounter(self):
        """
        Clear the counter on how many times data has been requested but not sent
        """
        self._dataRequestCounter = 0

    def getDataRequestCounter(self):
        return self._dataRequestCounter
        
    def __del__(self):
        for s in self.slaves:
            del s
        
    def avgValueSetter(self,attr,val):
        ## TODO: CHECK if this is still used
        return np.r_[attr,val]

    def removeOldData(self):
        """
        Remove old data based on the timestamp of arrival
        """
        while self.arrivalTimes[0]['time'] < time.time() - self.timestampTimeOut:
            if len(self.arrivalTimes) > 1:
                newHead = self.arrivalTimes[1]['idx']
                # print('newHead %d' %(newHead))
                self.data['data'] = self.data['data'][newHead:]
                self.data['trust'] = self.data['trust'][newHead:]

                # update slave indices

                for s in self.slaves:
                    try:
                        status = s.updateIdx(newHead)
                        if status == -1:
                            log.info('slave %s timed out -- removing',s.workerId)
                            self.slaves.remove(s)
                            
                    except Exception as e:
                        log.error('error while updating slave %s for master %s',s.workerId,self.workerId)
                        log.exception(e)
                        self.slaves.remove(s)  # remove the slave

                if len(self.slaves) > 0:
                    self.head = min([s.head for s in self.slaves])
                    self.tail = min([s.tail for s in self.slaves])
                else:
                    if self.head < newHead:
                        log.warning('worker {}: Removing more data than has been processed'.format(self.workerId))
                        self.head = 0
                        self.tail -= newHead
                    else:
                        self.head -= newHead
                        self.tail -= newHead

                log.debug('%s removeOldData(): head %d tail %d',self.workerId,self.head,self.tail)
                # remove the old head from list of time indices
                for at in self.arrivalTimes[1:]:
                    at['idx'] -= newHead
                    
                self.arrivalTimes.pop(0)
            else:
                break

    # @classmethod
    # def toList(self,data):
    #     """ Returns the data as lists"""
    #     for key in self.arrayKeyNames:
    #         data[key] = data[key].tolist()
        
    
    def insertData(self,workerData):
        """
        Assuming that packets arrive in sequence and no timeouts occurred, the bits should already have been aligned by the demodulator. This should be easily obtainable since a packet is generated ~ every 200 ms (and pinging google takes  7 ms (uon around 10))
        
        This function checks the age of the last data based on time stamp. The buffers are flushed if the timestamp is too old
        
        """

        if not self.workerId == workerData['workerId']:
            raise WorkerIdError('Data workerId %s does not match worker workerId %s' %(
                            workerData['workerId'],self.workerId))
        
        self.arrivalTimes.append({'time': time.time(),
                                  'idx' : self.tail})

        if workerData['count']-1 > self.data['count']:
            log.warning('Missing {} packets. Last index {}, current {}'.format(workerData['count']-self.data['count']-1, self.data['count'], workerData['count']))
        
        for key in self.keyNames:
            # overwrite current stat data -- only used for visualization anyway
            self.safeAdd(key,workerData)

        for key in self.arrayKeyNames:
            # append bits and trust
            self.safeAppend(key,workerData)

            Ndata = len(self.data['data'])
            [s.updateTail(Ndata) for s in self.slaves] # update worker tails
            self.tail = Ndata

        # check if the length did not mess up
        assert len(self.data['data']) == len(self.data['trust']), "Data and trust vectors have different lengths. The demodulator should have taken care of this"



    def getData(self,idx=None):
        """
        Returns the bit data and trust up to idx or all if idx = None
        """
        if idx is None:
            return self.data['data'], self.data['trust']
        
        
        if idx >= len(self.data['data']):
            raise IndexError('Index out of range')

        return self.data['data'][:idx], self.data['trust'][:idx]
        

    def updateIdx(self,idx,dataUsed=True):
        """
        update the indices for active slave or self
        """

        if self.activeSlave:
            self.activeSlave.head -= idx
        else:
            self.head -= idx

        if not dataUsed:
            self.getCount -=1
    
    def getSelf(self,slaveId = None):
        """
        Returns the statistics collected to be passed on to the decoder
        """
        log.debug('worker %s count %d' %(self.workerId,self.data['count']))
        out = {}
        out['workerId'] = self.workerId
        for key in self.keyNames:
            # overwrite current stat data -- only used for visualization anyway
            out[key] = self.safeGet(key)
            # log.info('key {}: {}'.format(key,out[key]))
        slaveFound = False
        if slaveId:
            for s in self.slaves:
                if slaveId == s.workerId:
                    slaveFound = True
                    self.activeSlave = s
                    for key in self.arrayKeyNames:
                        out[key] = self.data[key][s.head:s.tail]
                        
                    if len(out['data']) > 0:
                        s.count += 1
                        s.head = s.tail
                    break
            if slaveFound == False:
                slaveFound = True
                s = Slave(slaveId,self.head,self.tail)
                self.activeSlave = s
                for key in self.arrayKeyNames:
                    out[key] = self.data[key][s.head:s.tail]
                    
                if len(out['data']) > 0:
                    s.head = s.tail

                self.slaves.append(s)
        else:
            # no slave id provided. return own head
            self.activeSlave = None
            for key in self.arrayKeyNames:
                out[key] = self.data[key][self.head:self.tail]

            out['count'] = self.getCount # gets incremented upon dataRequestCounter
            if len(out['data']) > 0:
                self.totalRequestCount +=1
                self._dataRequestCounter += 1 # counts up every time data is requested. Cleared by softCombiner every time data is sent for decode
                self.getCount +=1 # this means that data is sent

            log.debug('getSelf(): head %d tail %d',self.head,self.tail)
            self.head = self.tail
            
        # Clean up old data
        # self.removeOldData()
        return out
        
    
    def removeData(self,idx = None):
        """
        Truncates the used data from the buffers
        """
        self._arrivalTimestamp = None
        if idx != None:
            log.debug('Worker %s -- %d of %d items removed', self.workerId, idx,len(self.data['data']))
            
        if idx == None or idx == len(self.data):
            for key in arrayKeyNames:
                self.data[key] = np.array([],dtype = arrayDataTypes[key])

            self.head = 0
            self.tail = 0
            [s.resetHeadTail() for s in self.slaves]
        else:
            if idx > len(self.data['data']):
                # clean the arrays and throw an error that can be handled outside of the loop
                Ndata = len(self.data['data'])
                for key in arrayKeyNames:
                    self.data[key] = np.array([],dtype = arrayDataTypes[key])
                    
                self.head = 0
                self.tail = 0
                [s.resetHeadTail() for s in self.slaves]


                raise IndexError('Index %d out of range in array length %d' %( idx, Ndata))

            # remove the old data
            for key in arrayKeyNames:
                self.data[key] = self.data[key][idx:]

            # update indices for workers    
            [s.updateIdx(idx) for s in self.slaves]

            # set the head and tail to the minimum
            self.head = min([s.head for s in self.slaves])
            self.tail = min([s.tail for s in self.slaves])
            
           
                
    def safeAdd(self,key,dataIn):
        """
        Safe method to add data to internal dict and ignore fields that do not exist
        """
        if key in dataIn.keys():
            try:
                self.data[key] = self.keyDataTypes[key](dataIn[key])
            except Exception as e:
                log.error('Error while adding key %s',key)
                log.exception(e)
        else:
            if self.showWarnings:
                log.warning('Key %s not found for worker %s' %(key,dataIn['workerId']))

    def safeAppend(self,key,dataIn):
        """
        Safe method to appen array data to internal dict and ignore fields that do not exist
        """
        if key in dataIn.keys():
            try:
                self.data[key] = np.r_[self.data[key], np.array(dataIn[key],dtype = self.arrayDataTypes[key])]
            except Exception as e:
                log.error('Error inserting key {}. Expected array datatype {}, got {}'.format(key,self.arrayDataTypes[key],type(dataIn[key])))
                log.exception(e)
                raise e
        else:
            if self.showWarnings:
                log.warning('Key %s not found for worker %s' %(key,dataIn['workerId']))
   

    def safeGet(self,key):
        """
        Safe method to fetch data from dict and ignore fields taht do not exist
        """
        if key in self.data.keys():
            return self.data[key]
        else:
            if self.showWarnings:
                log.warning('Key %s not found for worker %s' %(key,self.workerId))
            return []

        
    def __eq__(self,other):
        # to check object identity
        if isinstance(other,self.__class__):
            return self.workerId == other.workerId and self.timestamp == other.timestamp
        return False

    def __ne__(self,other):
        # to check not identity
        return self.workerId != other.workerId or self.timestamp != other.timestamp


                
        
class SoftCombiner(Process):


    def __init__(self,conf):
        """
        Get some parameters from the config file
        """
        Process.__init__(self)
        self.name = 'softCombiner'
        self.conf = conf

        self.dataRequestThreshold = conf['SoftCombiner']['workerDataRequestThreshold']
        self.demodInAddr = conf['Interfaces']['Internal']['demodIn']
        self.decoderOutAddr = conf['Interfaces']['Internal']['decodeOut']
        
        self.MIN_LENGTH = conf['SoftCombiner']['minProcessingLength'] # minimum number of bits before processing commences
        self.pollingTimeout = conf['SoftCombiner']['pollingTimeout'] # the timeout of each poll
        self.workerTimeout =  conf['SoftCombiner']['workerTimeout'] # the timeout before workers get deleted
        self.workerDataTimeout = conf['SoftCombiner']['workerDataTimeout'] # if the time difference between timestamps for one worker is greater than this, the old data gets deleted

        self.compareInterval = self.conf['SoftCombiner']['processingInterval']
        self.varMultiplier = self.conf['SoftCombiner']['varianceMultiplier'] # used to scale the threshold for the cross correlation
        self.masterVoteWeight = self.conf['SoftCombiner']['masterVoteWeight'] # putting this > 1 will make ties go to the master < 1 to the others


        # stuff to output stats on number of workers to RPC etc
        self.getWorkerStatsFlag = Event()
        self.getWorkerStatsFlag.clear()
        self.workersSubmittedToRPCQueue  = Queue()
        
        
        
        #
        self.daemon = True
        self.runStatus = Event()
        self.runStatus.set() # set running == True
        log.info('Starting SoftCombiner')

    def stop(self):
        """
        Stops the process in a clean way
        """
        log.info('Received request to stop')
        self.runStatus.clear()


    def getActiveWorkers(self,timeout=0.15):
        """
        Returns a list with the active workers that have submitted since the last check
        Can be called from other processes or RPC interfaces
        """
        while not self.workersSubmittedToRPCQueue.empty():
            try:
                self.workersSubmittedToRPCQueue.get(False) # don't block
            except queue.Empty:
                log.error('Expected elements to be in the queue')
        
        self.getWorkerStatsFlag.set() # is polled regularly in the process and makes it put the list of workers in the queue
        try:
            activeWorkers = self.workersSubmittedToRPCQueue.get(True,timeout=timeout) # 150 ms timeout
        except queue.Empty:
            log.warning('Quirying active workers timed out -- returning empty array instead')
            activeWorkers = []

        return activeWorkers
        
    

    def sigTermHandler(self,signum,frame):
        # We ignore SIGTERM and let the main process handle shutdown of the threads
        #raise SystemExit('SIGTERM received')
        pass
    
    def receiveData(self,demodIn):
        """
        Called from run
        This method receives data on the demodIn port and assigns it to a worker
        """
        
        timeoutCount = 0
        log.debug('getting data')
        data = demodIn.recv_pyobj(zmq.DONTWAIT)
        log.debug('data received')
        # Source of the data is identified by 'workerId'
        workerId = data.get('workerId') # string

        if workerId == None:
            log.error('Invalid data format: expected \'workerId\'')
        else:
            # we're good -- Append the data to the current workers data
            lastTimeStamp = time.time()
            workerFound = False

            # Check if the submitting worker is registered
            #log.debug('Current workers registered %d' %(len(self.workers)))
            for worker in self.workers:
                if worker.workerId == workerId:
                    #log.debug('Found worker %s -- appending block %d' % (workerId,data['count']))
                    # The worker exists, Add data
                    worker.insertData(data)
                    if not worker in self.workersSubmitted:
                        self.workersSubmitted.append(worker)
                    workerFound = True
                    #log.debug('worker %s len data %d' %(worker.workerId,len(worker.data['data'])))
                    break
            # if the worker was not found, register a new one
            if workerFound == False:
                log.debug('Could not find worker %s -- creating new one. current workers %s', workerId,
                          str([w.workerId for w in self.workers]))
                # Create new worker
                worker = Worker(data, timestampTimeOut = self.workerDataTimeout)
                self.workers.append(worker)
                self.workersSubmitted.append(worker)
            # log.debug('Worker %s appended', worker.workerId)
     
    def _doVoteN(self,bitsM,trustM,bitsS,trustS):
        """
        Does voting between master and slaves. bitsM, trustM are arrays. bitsS, trustS are lists
        
        vote between all
        First check bits that disqualify:
        Check based on trust whether a bit disqualifies (<0). set the bit value to 0
        Check consensus
        """

        log.debug('doVoteN -- trust size {}, trustS size {},  bits size {}, bitsS size {}'.format(trustM.shape,np.array(trustS).shape,bitsM.shape,np.array(bitsS).shape))

        try:
            bits = np.vstack([bitsM.astype(float)*self.masterVoteWeight,np.array(bitsS).astype(float)])  # The master gets weighted, such that ties go to the master side
        except Exception as e:
            log.error('can not stack bits arrays -- bits size {}, bitsS size {}'.format(bitsM.shape,np.array(bitsS).shape))
            log.exception(e)
            raise e
        try:
            trust = np.vstack([trustM,np.array(trustS)])
        except Exception as e:
            log.error('can not stack trust arrays -- trust size {}, trustS size {},  bits size {}, bitsS size {}'.format(trustM.shape,np.array(trustS).shape,bitsM.shape,np.array(bitsS).shape))
            log.exception(e)
            raise e


        bits[trust<0] = 0
        threshold = np.sum(trust>=0,axis=0).astype(float)/2 # threshold is based on bits that have not disqualified themselves
        threshold[trustM>=0] += self.masterVoteWeight/2 # weight the trust to the master side
        
        bVal = np.sum(bits,axis=0).astype(float)
        bitsT = (bVal.astype(float) > threshold).astype(DATATYPE)
        ## Trust has the format v.n
        # TODO: this has to be made compatible with the int8 format (use upper and lower nibble)
        #    - where v is the number of channels that agreed for the bit
        #    - n is the number of qualified voters for the bit
        trustT = bits.shape[0]/10-np.sum(trust==-1,axis=0)/10 # number of qualified voters for each bit
        # see the amount of trusted channels that agreed
        trustT[bVal==1] += np.sum(bits[:,bVal==1],axis=0).astype(DATATYPE) # to get all the ones
        trustT[bVal==0] += np.sum(bits[:,bVal==0]-(trustT[bVal==0]*10+bits.shape[0]),axis=0).astype(DATATYPE) # to get all the zeros

        log.debug('trust \n{}'.format(trust[:,100:120]))
        log.debug('doVoteN -- bVal shape {}, trustT shape {}'.format(bitsT.shape,trustT.shape))
        log.debug('all bits \n{}'.format(bits[:,100:120]))
        log.debug('threshold {}'.format(threshold[100:120]))
        log.debug('bVal {}'.format(bVal[100:120]))
        log.debug('bits {}'.format(bitsT[100:120]))
        log.debug('trustT {}'.format(trustT[100:120]))
        return bitsT.astype(DATATYPE), trustT.astype(TRUSTTYPE)


    

    def _doVote2(self,bitsM,trustM,bitsS,trustS):
        """
        Does voting for two same length arrays of bits and trust, returns combined bits and trust
        """

        bitsV = bitsM + bitsS # 0: both agree, 2: both agree, 1: disagree

        trustV = np.ones(len(bitsM),dtype=TRUSTTYPE)
        
        idx = np.where(bitsV == 1)[0] # mismatching bits

        bitsV = (bitsV/2).astype(DATATYPE) # we want 0 and 1s
        try:
            log.debug('trust M\n{}'.format(trustM[100:120]))
            log.debug('trust S\n{}'.format(trustS[100:120]))
        except:
            log.debug('trust M\n{}'.format(trustM))
            log.debug('trust S\n{}'.format(trustS))
        
        # check disagreements
        for i in idx:
            if trustS[i] < 0:
                # prefer master
                bitsV[i] = bitsM[i]
                if trustM[i] < 0:
                    # both disqualify themselves
                    trustV[i] = BOTH_DISTRUST # both distrust
                else:
                    trustV[i] = MASTER_TRUST # master trust

            elif trustM[i] < 0:
                if trustS[i] > 0:
                    bitsV[i] = bitsS[i] # use slave
                    trustV[i] = SLAVE_TRUST # slave trust
            else:
                # both trust their bit, choose master
                bitsV[i] = bitsM[i]
                trustV[i] = BOTH_TRUST_ERR # both trust
       
        return bitsV, -trustV
                    
  
    def correlate(self,master,slaves):
        """
        The master data gets compared to all the slaves. If the length of the match is too short, ignore the slave data
        If a slave does not fit, discard. 
        If there is more than one slave:
               we can do proper voting
        If there is one slave: 
               we vote with disagreements favored to the master
        If theres is no fit: 
               use the master
        """
        """
        Xcorr
        if no clear fit:
           ignore slave data and submit master
        Align slave data to master
        vote for the bits where they align, use master for the rest
        send to decoder
        
        """
        dataM = master.getSelf()
        
        if len(dataM['data']) == 0:
            # no new data
            return None

        bitsM, trustM = dataM['data'], dataM['trust'] # only get the data we want to submit

        bitsS, trustS, nameS = [], [], []

        start = time.time()
        log.debug('start finding slaves --------------------')
        for s in [s for s in slaves if s.voteGroup == master.voteGroup]:
            bitsT, trustT = s.getData() # get all the data

            t = time.time()
            # xCorrRes = np.abs(customXCorr(bitsT[-10000:],bitsM[:5000])) # max shows the index where bitsM is in bitsS
            n = len(bitsT)
            nAdd = int(2**(np.ceil(np.log2(n))))
            bitsX = np.r_[bitsT,np.zeros(nAdd-n)]
            log.debug('startXcorr  -- len bitsT {}, len zeros {} total length {} (log2 {})'.format(n,nAdd-n,len(bitsX),np.log2(len(bitsX))))
            xCorrRes = np.abs(customXCorr(bitsX,bitsM[:n])) # max shows the index where bitsM is in bitsS
            log.debug('xcorr Time {:6f} -- length {}'.format(time.time()-t,len(xCorrRes)))
            
            Nidx = 15
            idx = np.empty(Nidx,dtype=int)
            val = np.empty(Nidx)

            for i in range(Nidx):
                idx[i] = np.argmax(xCorrRes)
                val[i] = xCorrRes[idx[i]]
                xCorrRes[idx[i]] = 0

            
            #Xcorr conditions
            cond = np.mean(val[2:]) + self.varMultiplier*np.std(val[2:])
            condTrue = val[0] > cond

            log.debug('master {:8} slave {:8}: Xcorr tol {: 6.0f}\t values: {}'.format(master.workerId,s.workerId,cond, str(val[:5])))

        
            if condTrue: 
            # We're good, check whether length of vectors aligns
            
                log.debug('vote for %s',dataM['workerId'])
                bitsT = bitsT[idx[0]:idx[0]+len(bitsM)]
                trustT = trustT[idx[0]:idx[0]+len(trustM)]
                # check length
                if len(bitsT) < self.MIN_LENGTH:
                    log.debug('below min len')
                    # if below the minimum length, cancel the comparison and try again next time
                    master.updateIdx(len(bitsM),dataUsed=False)
                    return None
                elif len(bitsT) < len(bitsM):
                    log.debug('Truncating')
                    # else the matching part is longer than the minimum length, continue
                    master.updateIdx(len(bitsM) - len(bitsT))
                    bitsM = bitsM[:len(bitsT)]
                    trustM = trustM[:len(trustT)]
                    # update all the slaves
                    for i in range(len(bitsS)):
                        bitsS[i] = bitsS[i][:len(bitsT)]
                        trustS[i] = trustS[i][:len(bitsT)]


                bitsS.append(bitsT)
                trustS.append(trustT)
                nameS.append(s.workerId)
                
                log.debug('len matched bits %d',len(bitsS))

                
            # log.debug('bits Master: %s', str(bitsM[:30]))
            # log.debug('bits Slave: %s', str(bitsT[:30]))
        log.debug('correlate and fit took {:.6f} s to process'.format(time.time()-start))
    
        if len(bitsS) > 1:
            # Do proper vote
            log.debug('Master %s\tvote between %d workers -- bits length %d',master.workerId,len(bitsS)+1,len(bitsM))
            bitsM, trustM = self._doVoteN(bitsM,trustM,bitsS,trustS)
            log.debug('received bits length %d',len(bitsM))
            dataM['data'] = bitsM
            dataM['trust'] = trustM
            
        elif len(bitsS) == 1:
            # vote with master being right in disagreements
            log.debug('vote between 2 workers')
            bitsM, trustM = self._doVote2(bitsM,trustM,bitsS[0],trustS[0])
            dataM['data'] = bitsM
            dataM['trust'] = trustM
        else:
            # crap: just use the master values
            log.debug('no vote for %s',dataM['workerId'])

            if len(dataM['data']) > MAX_DATA_LEN_BEFORE_TRANSMIT:
                log.debug('Too many unprocessed bits -- sending data')
            elif master.getDataRequestCounter() < self.dataRequestThreshold:
                log.debug('Waiting another cycle before sending data')
                master.updateIdx(len(bitsM),dataUsed=False)
                return None
            else:
                log.debug('Waited too long --  sending data')

            
        master.clearDataRequestCounter()
        # dataM['workerId'] = '%s-%s' %(master.workerId,'-'.join([s.workerId for s in slaves]))

        # dataM['workerId'] = master.workerId
        dataM['numSlaves'] = len(bitsS)
        dataM['slaveNames'] = nameS
    
        log.debug('Submitting with workerId %s',dataM['workerId'])
        
        return dataM


    @classmethod
    def printData(self,dataM):
        for key in dataM.keys():
            log.info('d[\'%s\']\t dtype %s',key,type(dataM[key]))
    
    
    def compareWorkers(self):
        """
        Called by a timer after a fixed interval
        Compare all possibilities as master and slave
        send output to decoder
        """

        log.debug('compareworkers')
        start = time.time()
        for m in range(len(self.workers)):
            slaves = self.workers.copy()
            master = slaves.pop(m)

            # if len(slaves) > 0:
            data = self.correlate(master,slaves)
            # else:
            #     data = None
            if data:
                log.debug('Compare master: \'{}\' '.format(master.workerId))
                try:
                    log.debug('sending data {} bytes'.format(len(data['data'])))
                    self.decodeOut.send_pyobj(data,zmq.NOBLOCK)
                except zmq.error.Again as e:
                    log.error('Failed to send data. Message [{0}]'.format(e))
            else:
                log.debug('no new data')    
            

        log.debug('compareworkers took {:.4f} s to process'.format(time.time()-start))
        # clean up old data
        for m in self.workers:
            m.removeOldData()

        
    def timerProcessor(self,signum,frame):
        """
        Timer overflow sets a flag, which triggers comparing of the bits
        """
        # log.debug('%f timer overflow. Current startProcessing: %s',time.time()%60, self.startProcessing)
        log.debug('timer -- Current startProcessing {} -- time since last {}'.format(self.startProcessing,time.time()-self.t))
        self.t = time.time()
        if self.startProcessing == False:
            self.startProcessing = True
    
    
    def run(self):
        """
        This function does the job (will likely be run in a process)
        """
        time.sleep(.5)
        
        self.t = time.time()
        ctx = zmq.Context()

        log.info('Registering demodulator input socket on ' + str(self.demodInAddr))
        try:
            demodIn = ctx.socket(zmq.PULL)
            demodIn.setsockopt(zmq.LINGER,0)
            demodIn.bind(self.demodInAddr)
        except Exception as e:
            log.error('Error while initializing demodulator input socket')
            log.exception(e)
            return

        log.info('Registering decoder output socket on ' + str(self.decoderOutAddr))
        try:
            self.decodeOut = ctx.socket(zmq.PUSH)
            # # decodeOut.setsockopt(zmq.LINGER, 0) # set the retry amount
            self.decodeOut.connect(self.decoderOutAddr)
        except Exception as e:
            log.error('Error while initializing decoder output socket')
            log.exception(e)
            demodIn.close()
            return

        # register poller
        demodPol = zmq.Poller()
        demodPol.register(demodIn,zmq.POLLIN)
        
        self.workers = []
        
        self.workersSubmitted = []
        timestampQueue = []
        timeoutCount = 0

        log.info('polling timeout %f s\t worker timeout %f s\t processing interval %f s' %(
            self.pollingTimeout,self.workerTimeout,self.compareInterval))

        log.info('SoftCombiner process initialized and running')
        self.startProcessing = False # set by timer
        try:
            orig_sigterm_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self.sigTermHandler)
            signal.signal(signal.SIGALRM, self.timerProcessor)
            signal.setitimer(signal.ITIMER_REAL,self.compareInterval,self.compareInterval)

            while self.runStatus.is_set():

                log.debug('polling -- timeout {}'.format(self.pollingTimeout))
                socks = demodPol.poll(self.pollingTimeout)
                log.debug('polling -- done')
                if len(socks) > 0 and socks[0][1] == zmq.POLLIN:
                    # Get the data
                    log.debug('received data')
                    self.receiveData(demodIn)
                    timeoutCount = 0
                else:
                    # Polling timed out
                    timeoutCount += self.pollingTimeout/1000
                    log.debug('polling timed out')
                    if timeoutCount > self.workerTimeout:
                        log.info('Input socket timed out')
                        if len(self.workers) > 0:
                            log.info('Cleaning up workers')
                            for worker in self.workers:
                                self.workers.remove(worker)
                            workersSubmitted = []
                        timeoutCount = 0


                if self.startProcessing is True and self.runStatus.is_set():
                    self.startProcessing = False
                    log.debug('go to compare')
                    self.compareWorkers()


                if self.getWorkerStatsFlag.is_set():
                    # requested to return status on number of workers submitted
                    self.getWorkerStatsFlag.clear()
                    workerNames = [w.workerId for w in self.workersSubmitted]
                
                    log.info('Active workers: {}'.format(workerNames))
                    self.workersSubmittedToRPCQueue.put(workerNames)

                    self.workersSubmitted = []
                    
            log.info('finished')
            
        except Exception as e:
            log.exception(e)
            raise e
        finally:
            log.info('shutting down')
            self.workersSubmittedToRPCQueue.close()
            for worker in self.workers:
                del worker
            del demodPol
            self.decodeOut.close()
            demodIn.close()
            log.info('closed')
            signal.signal(signal.SIGTERM, orig_sigterm_handler)  # restore signal, such that process.terminate() can kill this



        
if __name__ == "__main__":
    print("""
    This class is not meant to be run on itself
    Unit tests are located in test/test_softCombiner/
    """)
