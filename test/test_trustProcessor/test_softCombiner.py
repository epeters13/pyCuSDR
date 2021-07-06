# Copyright: (c) 2021, Edwin G. W. Peters

import sys
sys.path.insert(1,'../../py-cuda-sdr/')
sys.path.append('../')

from __global__ import *
import importlib
import json,rjsmin
import numpy as np
import matplotlib.pyplot as plt
import logging as log
import zmq
import time
import itertools
import lib

import softCombiner

import loadConfig
import unittest


# count maximum number of consecutive ones (https://codereview.stackexchange.com/questions/138550/count-consecutive-ones-in-a-binary-list)
def lenIter(items):
    return sum(1 for _ in items)
def consecutiveOne(data):
    return max(lenIter(run) for val, run in itertools.groupby(data) if val)
#

# repeat the same test
def repeat(times):
    def repeatHelper(f):
        def callHelper(*args):
            for i in range(0, times):
                f(*args)

        return callHelper

    return repeatHelper

def generateRandomWorkerData(N=4000,workerId = 'testCase',dataType='random'):
    workerD = {'workerId': workerId,
               'doppler': np.random.randn(),
               'doppler_std': np.random.randn(),
               'count' : 0,
               'timestamp': time.time(),
               'spSymEst': 16,
               'data': np.random.randint(0,2,N,dtype=DATATYPE).tolist(),
               'trust': np.abs(np.random.randn(N).astype(TRUSTTYPE)).tolist(),
               'voteGroup' : 1
               }
    if dataType == 'arange':
        # data has to fit in uint8
        workerD['data'] = (np.arange(N)/N*128).astype(DATATYPE).tolist() 
        workerD['trust'] = np.arange(N).astype(TRUSTTYPE).tolist()
    return workerD


def initSockets(conf):
    ctx = zmq.Context()

    log.debug('Initializing sockets')
    demodOutAddr = conf['Interfaces']['Internal']['demodOut']
    log.debug('Registering PUSH output socket on ' + str(demodOutAddr))
    demodOut = ctx.socket(zmq.PUSH)
    demodOut.connect(demodOutAddr)

    decodInAddr = conf['Interfaces']['Internal']['decodeIn']
    log.debug('Registering PULL input socket on ' + str(decodInAddr))
    decodIn = ctx.socket(zmq.PULL)
    decodIn.bind(decodInAddr)

    return demodOut, decodIn

class TestSoftCombiner(unittest.TestCase):

    def setUp(self):
        global conf
        self.verificationErrors = []

        self.softComb = softCombiner.SoftCombiner(conf)
        self.softComb.compareInterval = 0.15
        self.softComb.start()

        self.demodOut,self.decodIn = initSockets(conf)
        self.decodPol = zmq.Poller()
        self.decodPol.register(self.decodIn,zmq.POLLIN)


    def tearDown(self):
        self.softComb.stop()
        self.softComb.join()
        self.demodOut.close()
        self.decodIn.close()
        del self.softComb
        if len(self.verificationErrors)>0:
            [print(e) for e in self.verificationErrors]
        self.assertEqual([], self.verificationErrors)

    def pollRecv_pyobj(self,timeout=200):
        socks = [self.decodIn.poll(timeout)]
        # print('socks ' + str(socks))
        if len(socks) > 0 and socks[0] == zmq.POLLIN:
            # Get the data
            return self.decodIn.recv_pyobj()

        else:
            return None

    def testSendWorkerSequentialData(self):
        """
        Sends sequential data to the worker 
        """
        print('testSendWorkerSequentialData')
        d1 = generateRandomWorkerData(workerId='first',dataType='arange')
        d2 = generateRandomWorkerData(workerId='second',dataType='arange')
        time.sleep(2)
        self.demodOut.send_pyobj(d1)
        self.demodOut.send_pyobj(d2)
        time.sleep(2)
        dR1 = self.pollRecv_pyobj()
        dR2 = self.pollRecv_pyobj()
        dR3 = self.pollRecv_pyobj()
        dR4 = self.pollRecv_pyobj()

        print('name dR1 %s' % dR1['workerId'])
        print('name dR2 %s' % dR2['workerId'])
        self.assertEqual(dR3,None,'Only two responses expected')
        self.assertEqual(dR4,None,'Only two responses expected')

        self.assertEqual(len(dR1['data']),len(d1['data']),'data from worker 1 and received should be equal length')
        self.assertEqual(len(dR2['data']),len(d2['data']),'data from worker 1 and received should be equal length')
        self.assertEqual(len(dR1['trust']),len(d1['trust']),'trust from worker 1 and received should be equal length')
        self.assertEqual(len(dR2['trust']),len(d2['trust']),'trust from worker 1 and received should be equal length')

        self.assertTrue(np.all(np.array(d1['data'])==np.array(dR1['data'])),'data from worker 1 expected to be equal')
        self.assertTrue(np.all(np.array(d1['trust'])==np.array(dR1['trust'])),'data from worker 1 expected to be equal')
        self.assertTrue(np.all(np.array(d2['data'])==np.array(dR2['data'])),'data from worker 2 expected to be equal')
        self.assertTrue(np.all(np.array(d2['trust'])==np.array(dR2['trust'])),'data from worker 2 expected to be equal')


    def testSendWorkerRandomData(self):

        print('testSendWorkerRandomData')

        d1 = generateRandomWorkerData(workerId='first')
        d2 = generateRandomWorkerData(workerId='second')
        time.sleep(.5)
        self.demodOut.send_pyobj(d1)
        self.demodOut.send_pyobj(d2)
        time.sleep(.5)

        dR1, dR2 = {}, {}
        dR1['data'] = []
        dR1['trust'] = []
        dR1['numSlaves'] = []
        dR2['data'] = []
        dR2['trust'] = []
        dR2['numSlaves'] = []
        for i in range(8):
            dd = self.pollRecv_pyobj()
            if dd:
                print('name packet {}'.format(dd['workerId']))
                if dd['workerId'] == 'first':
                    dR1['data'].extend(dd['data'])
                    dR1['trust'].extend(dd['trust'])
                    dR1['numSlaves'].append(dd['numSlaves'])
                elif dd['workerId'] == 'second':
                    dR2['data'].extend(dd['data'])
                    dR2['trust'].extend(dd['trust'])
                    dR2['numSlaves'].append(dd['numSlaves'])

                else:
                    raise Exception('Worker {} not found'.format(dd['workerId']))


        
        #self.assertEqual(dR3,None,'Only two responses expected')
        #self.assertEqual(dR4,None,'Only two responses expected')

        self.assertEqual(len(dR1['data']),len(d1['data']),'data from worker 1 and received should be equal length')
        self.assertEqual(len(dR2['data']),len(d2['data']),'data from worker 1 and received should be equal length')
        self.assertEqual(len(dR1['trust']),len(d1['trust']),'trust from worker 1 and received should be equal length')
        self.assertEqual(len(dR2['trust']),len(d2['trust']),'trust from worker 1 and received should be equal length')

        if all([s == 0 for s in dR1['numSlaves']]):
            self.assertTrue(np.all(np.array(d1['data'])==np.array(dR1['data'])),'data from worker 1 expected to be equal')
            self.assertTrue(np.all(np.array(d1['trust'])==np.array(dR1['trust'])),'data from worker 1 expected to be equal')
        else:
            print('Not comparing data for dR1 since vote occured -- data will not fit original')
        if all([s == 0 for s in dR2['numSlaves']]):
            self.assertTrue(np.all(np.array(d2['data'])==np.array(dR2['data'])),'data from worker 2 expected to be equal')
            self.assertTrue(np.all(np.array(d2['trust'])==np.array(dR2['trust'])),'data from worker 2 expected to be equal')
        else:
            print('Not comparing data for dR2 since vote occured -- data will not fit original')

        
    def testSendWorkerNoisy(self):
        """
        Worker 2 contains corrupted data from worker 1 with appropriate trust levels (-1 for clipping)
        The voted array for both workers must be equal to each other and worker 1
        The voted trust for both workers must be equal to each other and different to worker 1
        """
        print('testSendWorkerNoisy')

        d1 = generateRandomWorkerData(workerId='first')
        d2 = generateRandomWorkerData(workerId='second')
        time.sleep(.5)
        dd2 = np.array(d1['data'].copy(),dtype=DATATYPE)
        d2['data'] = (((dd2.astype(TRUSTTYPE)+np.sqrt(.9)*np.random.randn(len(dd2))) > 0.5).astype(DATATYPE)).tolist()

        # generate trust data
        d1['trust'] = np.ones(len(d1['data'])).tolist()
        d2['trust'] = [float(a==b)-1 for a,b in zip(d1['data'],d2['data'])]
        
        self.demodOut.send_pyobj(d1)
        self.demodOut.send_pyobj(d2)
        time.sleep(.5)
        dR1 = self.pollRecv_pyobj()
        dR2 = self.pollRecv_pyobj()
        dR3 = self.pollRecv_pyobj()
        dR4 = self.pollRecv_pyobj()

        print('name dR1 %s' % dR1['workerId'])
        print('name dR2 %s' % dR2['workerId'])

        print('data dR1 %s' % dR1['data'][:10])
        print('data d1 %s' % d1['data'][:10])
        print('trust d1 %s' % d1['trust'][:10])
        print('data dR2 %s' % dR2['data'][:10])
        print('data d2 %s' % d2['data'][:10])
        print('trust d2 %s' % d2['trust'][:10])
        
        #self.assertEqual(dR3,None,'Only two responses expected')
        #self.assertEqual(dR4,None,'Only two responses expected')

        self.assertEqual(len(dR1['data']),len(d1['data']),'data from worker 1 and received should be equal length')
        self.assertEqual(len(dR2['data']),len(d2['data']),'data from worker 1 and received should be equal length')
        self.assertEqual(len(dR1['trust']),len(d1['trust']),'trust from worker 1 and received should be equal length')
        self.assertEqual(len(dR2['trust']),len(d2['trust']),'trust from worker 1 and received should be equal length')

        self.assertTrue(np.all(np.array(d1['data'])==np.array(dR1['data'])),'data from worker 1 expected to be equal')
        self.assertTrue(np.all(np.array(d1['data'])==np.array(dR2['data'])),'voted data from worker 2 expected to be equal to worker 1')



    def testSendSingleWorkerData(self):
        """
        Only one worker. send random bits. 
        Check that all data comes back chronologically and unchanged.
        No voting will take place
        """
        print('testSendSingleWorkerData')
        time.sleep(2)
        N = 15

        dd1 = []; dt1 = []; ddR1 = []; dtR1 = []
        
        for i in range(N):
            d1 = generateRandomWorkerData(workerId='first')
        
            dd1.extend(d1['data'])
            dt1.extend(d1['trust'])
            

            d1['count'] = i
            self.demodOut.send_pyobj(d1)
            time.sleep(0.2)
            pkt = self.pollRecv_pyobj(timeout=500)

            if pkt:
                ddR1.extend(pkt['data'])
                dtR1.extend(pkt['trust'])

        for i in range(5):
            # In case some packets are delayed
            pkt = self.pollRecv_pyobj(timeout=500)
            if pkt:
                ddR1.extend(pkt['data'])
                dtR1.extend(pkt['trust'])


        # we expect all to be back
        
        self.assertEqual(len(ddR1),len(dtR1),'received data and trust should be equal length (reveived data {} and trust {})'.format(len(ddR1),len(dtR1)))
        self.assertEqual(len(ddR1),len(dd1),'received data should be equal length to transmitted (reveived received {} and transmitted {})'.format(len(ddR1),len(dd1)))


        self.assertTrue(np.all(np.array(ddR1)==np.array(dd1)),'received data should be identical to transmitted data')
        self.assertTrue(np.all(np.array(dtR1)==np.array(dt1)),'received trust should be identical to transmitted data')

    def testSendTwoWorkerDataInNoise(self):
        """
        Worker 2 contains corrupted data from worker 1 with appropriate trust levels (-1 for clipping)
        The data is out of sync and in between noise
        The voted array for both workers must be equal to each other and worker 1
        The voted trust for both workers must be equal to each other and different to worker 1
        """
        print('testSendTwoWorkerDataInNoise')
        time.sleep(2)
        N = 15
        dataIdx = [3,4,8]
        worker2Delay = 1000 + np.random.randint(1000) # 1000 bits = 104 ms delay 

        d2Buf = []
        d2TrustBuf = []

        dR1 = []
        dR2 = []
        dd1 = []
        dt1 = []
        dd2 = []
        dt2 = []
        for i in range(N):
            # Create the data. Either random bits or equal bits with some bits flipped
            if i in dataIdx:
                print('dataidx')
                # data
                d1 = generateRandomWorkerData(workerId='first')
                d2Data = np.array(d1['data'].copy(),dtype=DATATYPE)
                d2Data = (((d2Data.astype(TRUSTTYPE)+np.sqrt(.7)*np.random.randn(len(d2Data))) > 0.5).astype(DATATYPE)).tolist()

               #  generate trust data
                d1['trust'] = np.ones(len(d1['data'])).tolist()
                d2Trust = [float(a==b)-1 for a,b in zip(d1['data'],d2Data)]

                if len(d2Buf) > 0:
                    d2['data'][:len(d2Buf)] = d2Buf
                    d2['trust'][:len(d2Buf)] = d2TrustBuf
                d2['data'][worker2Delay:] = d2Data[:-worker2Delay]
                d2Buf = d2Data[-worker2Delay:]
                d2['trust'][worker2Delay:] = d2Trust[:-worker2Delay]
                d2TrustBuf = d2Trust[-worker2Delay:]

                print('errors %d' %(np.sum(np.array(d2['data'][worker2Delay:])!=np.array(d1['data'][:-worker2Delay]))))
                # assert(np.all(np.array(d2['data'][worker2Delay:])==np.array(d1['data'][:-worker2Delay])))
            else:
                print('noise')
                # noise
                d1 = generateRandomWorkerData(workerId='first')
                d2 = generateRandomWorkerData(workerId='second')
                d1['trust'] = np.ones(len(d1['data'])).tolist()
                d2['trust'] = np.abs(np.array(d2['trust'])).tolist()
                print('trust %d\t data %d' %(len(d2['data']),len(d2['trust'])))

                if len(d2Buf) > 0:
                    print('d2Buf %d\td2TrustBuf %d' %(len(d2Buf),len(d2TrustBuf)))
                    d2['data'][:len(d2Buf)] = d2Buf
                    d2['trust'][:len(d2Buf)] = d2TrustBuf
                    d2Buf = []
                    d2TrustBuf = []
            print('data %d\t trust %d' %(len(d2['data']),len(d2['trust'])))

            
            dd1.extend(d1['data'])
            dd2.extend(d2['data'])
            dt1.extend(d1['trust'])
            dt2.extend(d2['trust'])
            
            assert len(d2['data'])==len(d2['trust']), 'data and trust should be same length'

            d1['count'] = i
            d2['count'] = i
            self.demodOut.send_pyobj(d1)
            self.demodOut.send_pyobj(d2)
            time.sleep(0.2)
            for i in range(2):
                pkt = self.pollRecv_pyobj(timeout=500)
                if pkt:
                    if pkt['workerId'] == 'first':
                        dR1.append(pkt)
                    elif pkt['workerId'] == 'second':
                        dR2.append(pkt)
                    else:
                        raise Exception('Worker id %s not expected' %(pkt['workerId']))

        time.sleep(0.1)
        # wait for a couple of packets in case some data is delayed (due to the shifting that can occur in the processor)
        for i in range(2):
                pkt = self.pollRecv_pyobj(timeout=500)
                if pkt:
                    if pkt['workerId'] == 'first':
                        dR1.append(pkt)
                    elif pkt['workerId'] == 'second':
                        dR2.append(pkt)
                    else:
                        raise Exception('Worker id %s not expected' %(pkt['workerId']))

        # time.sleep(.5)

        # we expect all the data back
        ddR1 = []
        ddR2 = []
        dtR1 = []
        dtR2 = []


        ## Consolidate data and stats from the returned data
        wI1 = [d['workerId'] for d in dR1]
        wI2 = [d['workerId'] for d in dR2]
        print('workers1: %s' % wI1)
        print('workers2: %s' % wI2)
        [ddR1.extend(d['data']) for d in dR1]
        [ddR2.extend(d['data']) for d in dR2]
        [dtR1.extend(d['trust']) for d in dR1]
        [dtR2.extend(d['trust']) for d in dR2]
        # packet index
        dc1 = [] 
        [dc1.append(d['count']) for d in dR1]
        dc2 = []
        [dc2.append(d['count']) for d in dR2]
        # buffer length
        dL1 = []
        [dL1.append(len(d['data'])) for d in dR1]
        dL2 = []
        [dL2.append(len(d['data'])) for d in dR2]
        # voted
        dv1 = []
        [dv1.append(d['numSlaves'] == 1) for d in dR1]
        dv2 = []
        [dv2.append(d['numSlaves'] == 1) for d in dR2]

        
        np.savez('bitData_test',
                 ddR1 = ddR1[:-worker2Delay], ddR2 = ddR2[worker2Delay:],
                 dd1 = dd1[:-worker2Delay], dd2 = dd2[worker2Delay:])
        
        
        print('name dR1 %s' % dR1[0]['workerId'])
        print('name dR2 %s' % dR2[0]['workerId'])

        print('len data dR1 %s' % len(ddR1))
        print('len data d1 %s' %  len(dd1))
        print('len trust d1 %s' % len(dt1))
        print('len data dR2 %s' % len(ddR2))
        print('len data d2 %s' %  len(dd2))
        print('len trust d2 %s' % len(dt2))
        print('worker2Delay %d' % worker2Delay)
        print('worker1 sequence returned %s' %(str(dc1)))
        print('worker2 sequence returned %s' %(str(dc2)))
        print('worker1 data len %s' %(str(dL1)))
        print('worker2 data len %s' %(str(dL2)))
        print('worker1 voted %s' %(str(dv1)))
        print('worker2 voted %s' %(str(dv2)))
        #self.assertEqual(dR3,None,'Only two responses expected')
        #self.assertEqual(dR4,None,'Only two responses expected')

        # self.assertEqual(len(ddR1),len(dd1),'data from worker 1 and received should be equal length')
        if not len(ddR1) == len(dd1):
            print('data from worker 1 and received differ in length (worker 1 %d received %d)' %(len(ddR1),len(dd1)))
        self.assertTrue(np.all(ddR1==dd1[:len(ddR1)]),'data from worker 1 and received should be equal')
        if not len(ddR2) == len(dd2):
            print('data from worker 2 and received differ in length (worker 2 %d received %d)' %(len(ddR2),len(dd2)))
        self.assertTrue(np.all(ddR2[:8000]==dd2[:8000]),'first 8K bits in data from worker 2 and received should be equal')
        # self.assertEqual(len(ddR2),len(dd2),'data from worker 1 and received should be equal length')
        self.assertEqual(len(dtR1),len(ddR1),'worker 1 received data and trust should be equal length')
        self.assertEqual(len(dtR2),len(ddR2),'worker 2 received data and trust should be equal length')
        #self.assertTrue(np.all(dtR1==dt1[:len(dtR1)]),'trust from worker 1 and received should be equal')
        #self.assertTrue(np.all(dtR2==dt2[:len(dtR2)]),'trust from worker 2 and received should be equal')

        
        alignedData = [int(a==b) for a,b in zip(ddR1[:-worker2Delay], ddR2[worker2Delay:])]
        longest_consecutive_match = consecutiveOne(alignedData) # the first 'packet'
        longest_consecutive_match2 = consecutiveOne(alignedData[25000:]) # the second 'packet

        # whether the matching worked or not depends on the config settings. In some cases the match is too short, which is not an issue
        print('longest_consecutive_match1 %d' %(longest_consecutive_match))
        print('longest_consecutive_match2 %d' %(longest_consecutive_match2))

        if dv2[3:6] == [True]*3: # if indices 3:6 have voted all indices should match
            # all sequences should be matched
            print('packet1: all three blocks have been compared -- expecting 8000 matching bits')
            self.assertGreaterEqual(longest_consecutive_match,8000,'Expected 2 blocks (at least 8000 bits) to be equal. This can fail if the matching sensitivity is too high. This does not mean that the code is wrong')
        else:
            print('packet1: not all three blocks have been compared -- expecting 6000 matching bits')
            self.assertGreaterEqual(longest_consecutive_match,6000,'Expected 2 blocks (at least 6000 bits) to be equal. This can fail if the matching sensitivity is too high. This does not mean that the code is wrong')

        if dv2[8:10] == [True]*2: # if indices 3:6 have voted all indices should match
            # all sequences should be matched
            print('packet2: both blocks have been compared -- expecting 4000 matching bits')
            self.assertGreaterEqual(longest_consecutive_match2,4000,'Expected 1 block (at least 4000 bits) to be equal. This can fail if the matching sensitivity is too high. This does not mean that the code is wrong')
        else:
            print('packet2: not all blocks have been compared -- expecting 2000 matching bits')
            self.assertGreaterEqual(longest_consecutive_match,2000,'Expected 2 blocks (at least 6000 bits) to be equal. This can fail if the matching sensitivity is too high. This does not mean that the code is wrong')
        
            
        # The return values of worker 1 should remain unchanged
        self.assertTrue(np.all(np.array(dd1)==np.array(ddR1)),'data from worker 1 expected to be equal')


    def testSend6WorkerDataInNoise(self):
        self.sendNWorkerDataInNoise(6)

    def testSend4WorkerDataInNoise(self):
        self.sendNWorkerDataInNoise(4)

    def testSend3WorkerDataInNoise(self):
        self.sendNWorkerDataInNoise(3)

   
    def testSend2WorkerDataInNoise(self):
        self.sendNWorkerDataInNoise(2)

        
    def sendNWorkerDataInNoise(self,N=2):
        """
        Worker 2:N contains corrupted data from worker 1 with appropriate trust levels (-1 for clipping)
        The data is out of sync and in between noise
        Some random selected workers only have noise
        The voted array for both workers must be equal to each other and worker 1 besides the ones that have only noise
        The voted trust for both workers must be equal to each other and different to worker 1
        """
        
        print('testSendNWorkerDataInNoise -- {} workers'.format(N))
        
        time.sleep(2)
        T = 15 # number of iterations (packets)
        #N = 2 # number of workers
        dataIdx = [3,4,8]
        dataSize = 4000    
        workerDelay = 1000 + np.random.randint(0,1000,N-1) # 1000 bits = 104 ms delay 

        
        d2Buf = [[]]*(N-1)
        d2TrustBuf = [[]]*(N-1)

        dRd = [[] for n in range(N)] # rx data
        dRt = [[] for n in range(N)] # rx trust
        dRc = [[] for n in range(N)] # rx count
        dRL = [[] for n in range(N)] # rx data length
        dRv = [[] for n in range(N)] # rx number slaves
        dd = [[] for n in range(N)] # tx data
        dt = [[] for n in range(N)] # tx trust

        wNames = ['worker%d' %i for i in range(N)]

        mu = 0.7 #0.7 # variance for packet loss
        mu2 = 0.7
        for i in range(T):
            # Create the data. Either random bits or equal bits with some bits flipped
            if i in dataIdx:
                print('dataidx')
                # data
                d1 = generateRandomWorkerData(N=dataSize,workerId=wNames[0])
                d1['trust'] = np.ones(len(d1['data'])).tolist()
                d1['count'] = i
                
                dd[0].extend(d1['data'])
                dt[0].extend(d1['trust'])

                
                for n in range(N-1):
                    d2 = generateRandomWorkerData(N=dataSize,workerId=wNames[n+1])
                    d2Data = np.array(d1['data'].copy(),dtype=DATATYPE)
                    if n == 0:
                        # allows to set 1 to a different disturbance
                        d2Data = (((d2Data.astype(TRUSTTYPE)+np.sqrt(mu)*np.random.randn(len(d2Data))) > 0.5).astype(DATATYPE)).tolist()
                    else:
                        d2Data = (((d2Data.astype(TRUSTTYPE)+np.sqrt(mu2)*np.random.randn(len(d2Data))) > 0.5).astype(DATATYPE)).tolist()
                    #  generate trust data
                    d2Trust = [float(a==b)-1 for a,b in zip(d1['data'],d2Data)]

                    if len(d2Buf[n]) > 0:
                        # stuff old buffered data in first
                        d2['data'][:len(d2Buf[n])] = d2Buf[n]
                        d2['trust'][:len(d2Buf[n])] = d2TrustBuf[n]


                    # add new data after that
                    d2['data'][workerDelay[n]:] = d2Data[:-workerDelay[n]]
                    d2Buf[n] = d2Data[-workerDelay[n]:]
                    d2['trust'][workerDelay[n]:] = d2Trust[:-workerDelay[n]]
                    d2TrustBuf[n] = d2Trust[-workerDelay[n]:]
                    d2['count'] = i

                    
                    dd[n+1].extend(d2['data'])
                    dt[n+1].extend(d2['trust'])
                    
                    self.demodOut.send_pyobj(d2)

                self.demodOut.send_pyobj(d1)
                
                print('errors %d' %(np.sum(np.array(d2['data'][workerDelay[n]:])!=np.array(d1['data'][:-workerDelay[n]]))))
                # assert(np.all(np.array(d2['data'][worker2Delay:])==np.array(d1['data'][:-worker2Delay])))
            else:
                print('noise')
                # noise
                for n in range(N):
                    d1 = generateRandomWorkerData(N=dataSize,workerId=wNames[n])
                    if n == 0:
                        d1['trust'] = np.ones(len(d1['data'])).tolist()
                    else:
                        d1['trust'] = np.abs(np.array(d1['trust'])).tolist()
                    d1['count'] = i
                    if n > 0:
                        if len(d2Buf[n-1]) > 0:
                            print('d2Buf %d\td2TrustBuf %d' %(len(d2Buf[n-1]),len(d2TrustBuf[n-1])))
                            d1['data'][:len(d2Buf[n-1])] = d2Buf[n-1]
                            d1['trust'][:len(d2Buf[n-1])] = d2TrustBuf[n-1]
                            d2Buf[n-1] = []
                            d2TrustBuf[n-1] = []

                    dd[n].extend(d1['data'])
                    dt[n].extend(d1['trust'])
                    self.demodOut.send_pyobj(d1)
                                        

            
            
            # assert len(d2['data'])==len(d2['trust']), 'data and trust should be same length'

            time.sleep(0.2)
            for i in range(N):
                pkt = self.pollRecv_pyobj(timeout=500)
                if pkt:
                    workerNo = int(pkt['workerId'][6:])
                    print('received worker number {} packet number {}'.format(workerNo, pkt['count']))
                    dRd[workerNo].extend(pkt['data'])
                    dRt[workerNo].extend(pkt['trust'])
                    dRc[workerNo].append(pkt['count'])
                    dRL[workerNo].append(len(pkt['data']))
                    dRv[workerNo].append(pkt['numSlaves'])

        time.sleep(0.1)
        # wait for a couple of packets in case some data is delayed (due to the shifting that can occur in the processor)
        for i in range(2*N):
                pkt = self.pollRecv_pyobj(timeout=500)
                if pkt:
                    workerNo = int(pkt['workerId'][6:])
                    print('received worker number {}'.format(workerNo))
                    dRd[workerNo].extend(pkt['data'])
                    dRt[workerNo].extend(pkt['trust'])
                    dRc[workerNo].append(pkt['count'])
                    dRL[workerNo].append(len(pkt['data']))
                    dRv[workerNo].append(pkt['numSlaves'])

       


        
        # np.savez('bitData_test',
        #          ddR1 = ddR1[:-worker2Delay], ddR2 = ddR2[worker2Delay:],
        #          dd1 = dd1[:-worker2Delay], dd2 = dd2[worker2Delay:])
        
        np.savez('bitData_test',
                 dRd = dRd, dRt = dRt,
                 dd = dd, dt = dt,
                 workerDelay = workerDelay)
        
        

        
        print('len data sent (dd) %s' % str([len(d) for d in dd]))
        print('len data received (dRd) %s' % str([len(d) for d in dRd]))
        print('len trust sent (dt) %s' % str([len(d) for d in dt]))
        print('len trust received (dRt) %s' % str([len(d) for d in dRt]))

        print('workerDelay %s' % str(workerDelay))

        print('---------------------------------------------------------')
        for n in range(N):
            print('worker{} sequence returned {}'.format(n,str(dRc[n])))
        print('---------------------------------------------------------')
        for n in range(N):
            print('worker{} data lengths returned {}'.format(n,str(dRL[n])))
        print('---------------------------------------------------------')
        for n in range(N):
            print('worker{} data votes {}'.format(n,str(dRv[n])))


            

        if N == 2:
            if not len(dRd[0]) == len(dd[0]):
                print('data from worker 0 and received differ in length (worker 0 %d received %d)' %(len(dRd[0]),len(dd[0])))
            # this is only true if we have 2 workers
            self.assertTrue(np.all(dRd[0]==dd[0][:len(dRd[0])]),'data from worker 0 and received should be equal')
        else:
            # All bits have been voted for, so it will only match in the known intervals
            matchIdx = np.array([],dtype=int)
            for i in dataIdx:
                matchIdx = np.r_[matchIdx, np.arange(dataSize * i,dataSize*(i+1))]
            # print( matchIdx)
            # print(type(matchIdx[0]))
            # print(np.array(dRd[0])[matchIdx])
            self.assertTrue(np.all(np.array(dRd[0])[matchIdx]==np.array(dd[0])[matchIdx]),'data from worker 0 and received should be equal')
            
        for n in range(N):
            self.assertEqual(len(dRt[n]),len(dRd[n]),'worker {} received data and trust should be equal length'.format(n))
            
        # for n in range(1,N):
        #     if not len(dRd[n]) == len(dd[n]):
        #         print('data from worker %d and received differ in length (worker %d %d received %d)' %(n,n,len(dRd[n]),len(dd[n])))
        #     print('Worker {} to {} -- {} bits matched'.format(n,0,np.sum(dRd[n][workerDelay[n-1]:workerDelay[n-1]+8000]==dd[0][:8000])))
        #     self.assertTrue(np.all(dRd[n][workerDelay[n-1]:workerDelay[n-1]+8000]==dd[0][:8000]),'first 8K bits in data from worker {} and received should be equal'.format(n))


        
        
        for n in range(1,N):
            alignedData = [int(a==b) for a,b in zip(dRd[0][:-workerDelay[n-1]], dRd[n][workerDelay[n-1]:])]
            longest_consecutive_match = consecutiveOne(alignedData) # the first 'packet'
            longest_consecutive_match2 = consecutiveOne(alignedData[25000:]) # the second 'packet

            # whether the matching worked or not depends on the config settings. In some cases the match is too short, which is not an issue
            print('worker %d longest_consecutive_match packet 1 %d' %(n, longest_consecutive_match))
            print('worker %d longest_consecutive_match packet 2 %d' %(n, longest_consecutive_match2))

            if all([k>0 for k in dRv[n][3:6]]): # if indices 3:6 have voted all indices should match
                # all sequences should be matched
                print('worker %d packet1: all three blocks have been compared -- expecting 8000 matching bits' %(n))
                try:
                    self.assertGreaterEqual(longest_consecutive_match,8000,'Worker {} Expected 2 blocks (at least 8000 bits) to be equal. This can fail if the matching sensitivity is too high. This does not mean that the code is wrong'.format(n))
                except AssertionError as e: self.verificationErrors.append(str(e))
            else:
                print('worker %d packet 1: not all three blocks have been compared -- expecting 6000 matching bits' %(n))
                try:
                    self.assertGreaterEqual(longest_consecutive_match,6000,'worker {} Expected 2 blocks (at least 6000 bits) to be equal. This can fail if the matching sensitivity is too high. This does not mean that the code is wrong'.format(n))
                except AssertionError as e: self.verificationErrors.append(str(e))

            if all([k > 0 for k in dRv[n][8:10]]): # if indices 8:10 have voted all indices should match
                # all sequences should be matched
                print('worker {} packet 2: both blocks have been compared -- expecting 4000 matching bits'.format(n))
                try:
                    self.assertGreaterEqual(longest_consecutive_match2,4000,'worker {} Expected 1 block (at least 4000 bits) to be equal. This can fail if the matching sensitivity is too high. This does not mean that the code is wrong'.format(n))
                except AssertionError as  e: self.verificationErrors.append(str(e))

            else:
                print('worker {} packet 2: not all blocks have been compared -- expecting 2000 matching bits'.format(n))
                try:
                    self.assertGreaterEqual(longest_consecutive_match,2000,'worker {} Expected 2 blocks (at least 6000 bits) to be equal. This can fail if the matching sensitivity is too high. This does not mean that the code is wrong'.format(n))
                except AssertionError as e: self.verificationErrors.append(str(e))

        
            


        
class TestInitSoftCombiner(unittest.TestCase):
    """
    This class just tests the construction, starting and shutdown
    of the process.
    The process is set to run long enough to make the timers and timeouts
    fire a few times
    """

    def testConstructionDeconstruction(self):
        print('testConstructionDeconstruction')

        global conf
        self.softComb = softCombiner.SoftCombiner(conf)
        self.softComb.start()
        time.sleep(1)
        self.softComb.stop()
        self.softComb.join()


    
if __name__ == '__main__':
    global conf
    # conf = loadConfig.getConfigAndLog()
    conf = loadConfig.getConfigAndLog('conf_test.json')
    conf['SoftCombiner']['minProcessingLength'] = 1000 # in case we compare, make sure all data compares in the tests
    
    unittest.main()

    





