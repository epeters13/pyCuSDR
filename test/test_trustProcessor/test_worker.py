# Copyright: (c) 2021, Edwin G. W. Peters

import sys
sys.path.append('../../py-cuda-sdr/')
sys.path.append('../')

import importlib
import softCombiner
import json,rjsmin
importlib.reload(softCombiner)
import numpy as np
import matplotlib.pyplot as plt
import logging 
import zmq
import time
import unittest
import numpy as np

import loadConfig

DATATYPE = np.int8
TRUSTTYPE = np.int8



def generateRandomWorkerData(N=4000):
    workerD = {'workerId': 'testCase',
               'doppler': np.random.randn(),
               'doppler_std': np.random.randn(),
               'count' : 0,
               'timestamp': time.time(),
               'spSymEst': 16,
               'data': np.random.randint(0,2,N).tolist(),
               'trust': np.random.randn(N).tolist(),
               'voteGroup': 1}
    return workerD


class TestWorker(unittest.TestCase):

    def setUp(self):
        self.workerD = generateRandomWorkerData()
        
    def testInit(self):
        worker = softCombiner.Worker(self.workerD)


    def testInsert(self):
        worker = softCombiner.Worker(self.workerD)
        worker.insertData(generateRandomWorkerData())
        worker.insertData(generateRandomWorkerData())

    def testDataTypes(self):
        worker = softCombiner.Worker(self.workerD)

        data = worker.getSelf()
        expectedDataTypes = {'workerId': str,
                             'count': int,
                             'timestamp':float,
                             'doppler': float,
                             'doppler_std': float,
                             'spSymEst': float,
                             'data' : np.ndarray,
                             'trust' : np.ndarray,
                             'voteGroup' : int,
                             'SNR': list,
                             'baudRate': list,
                             'baudRate_est': list,
                             'sample_rate': list,
                             'protocol': list}
        
        for key in data:
            self.assertEqual(type(data[key]),expectedDataTypes[key],'key %s failed' %(key))
        
    def testInsertFalseWorker(self):
        worker = softCombiner.Worker(self.workerD)
        worker.insertData(generateRandomWorkerData())
        wFalse = generateRandomWorkerData()
        wFalse['workerId'] = 'falseId'
        with self.assertRaises(softCombiner.WorkerIdError):
            worker.insertData(wFalse)
        worker.insertData(generateRandomWorkerData())


    def testInsertandGetData(self):
        """
        Test if all data is returned (hwen this worker is slave)
        """
        data = np.array([] ,dtype=DATATYPE)
        trust = np.array([],dtype=TRUSTTYPE)
        d = generateRandomWorkerData()
        worker = softCombiner.Worker(d)
        data = np.r_[data,np.array(d['data'],dtype=DATATYPE)]
        trust = np.r_[trust,np.array(d['trust'],dtype=TRUSTTYPE)]
        for i in range(3):
            d = generateRandomWorkerData()
            data = np.r_[data,np.array(d['data'],dtype=DATATYPE)]
            trust = np.r_[trust,np.array(d['trust'],dtype=TRUSTTYPE)]
            worker.insertData(d)

        dOut, tOut = worker.getData()
        
        
        self.assertEqual(len(data),len(dOut))
        self.assertEqual(len(trust),len(tOut))
        self.assertTrue(np.all(dOut==data))
        self.assertTrue(np.all(tOut==trust))
        del worker



    def testInsertAndGetSelf(self):
        """
        Gets it's own data within the desired borders returned
        """

        data = np.array([] ,dtype=DATATYPE)
        trust = np.array([],dtype=TRUSTTYPE)
        d = generateRandomWorkerData()
        worker = softCombiner.Worker(d)
        data = np.r_[data,np.array(d['data'],dtype=DATATYPE)]
        trust = np.r_[trust,np.array(d['trust'],dtype=TRUSTTYPE)]
        for i in range(3):
            d = generateRandomWorkerData()
            data = np.r_[data,np.array(d['data'],dtype=DATATYPE)]
            trust = np.r_[trust,np.array(d['trust'],dtype=TRUSTTYPE)]
            worker.insertData(d)

        dRet = worker.getSelf()
        dOut, tOut = dRet['data'], dRet['trust']
        
        self.assertEqual(len(data),len(dOut))
        self.assertEqual(len(trust),len(tOut))
        self.assertTrue(np.all(dOut==data))
        self.assertTrue(np.all(tOut==trust))

        del worker


    def testInsertAndGetSelfMultipleTime(self):
        """
        Gets it's own data within the desired borders returned
        Checks if data gets removed when old
        Checks if the proper data is returned
        """

        T = 0.05 # short for testing
        N = 1000
        noPackets = 5
        data = np.array([] ,dtype=DATATYPE)
        trust = np.array([],dtype=TRUSTTYPE)
        d = generateRandomWorkerData(N)
        worker = softCombiner.Worker(d,timestampTimeOut = T)
        print('start: number of slaves %d' % len(worker.slaves))
        data = np.r_[data,np.array(d['data'],dtype=DATATYPE)]
        trust = np.r_[trust,np.array(d['trust'],dtype=TRUSTTYPE)]
        time.sleep(0.02)
        for i in range(noPackets - 1):
            d = generateRandomWorkerData(N)
            data = np.r_[data,np.array(d['data'],dtype=DATATYPE)]
            trust = np.r_[trust,np.array(d['trust'],dtype=TRUSTTYPE)]
            worker.insertData(d)
            time.sleep(0.02)
            

        import copy
        arrivalTimes = copy.deepcopy(worker.arrivalTimes)
        self.assertEqual(len(arrivalTimes),noPackets,'Expected as many arrival times as packets inserted')
        times = []
        for at in arrivalTimes:
            at['time'] -= time.time()
            times.append(at['time'])
        print('timestamps: %s' %(str(arrivalTimes)))

        # returns all current data
        dRet = worker.getSelf()
        
        self.assertEqual(len(dRet['data']),N*noPackets,'All data should be gotten (len dRet %d expected %d)'%(len(dRet['data']),N*noPackets))

        self.assertEqual(worker.tail , len(worker.data['data']),'tail should be at the end of the data')
        self.assertEqual(worker.head , len(worker.data['data']),'head should be at the end of the data')
        # should remain after removing the old data
        worker.removeOldData()

        print('slaves %d'%len(worker.slaves))
        self.assertEqual(worker.tail , len(worker.data['data']),'tail should be at the end of the data')
        self.assertEqual(worker.head , len(worker.data['data']),'head should be at the end of the data')
        
        
        arrivalTimes = worker.arrivalTimes
        print('new timestamps: %s' %(str(arrivalTimes)))
        self.assertEqual(len(arrivalTimes),np.sum(np.array(times)>-T),'Old data not removed')


        dRet = worker.getSelf()
        worker.removeOldData()
        # no data should be received
        self.assertEqual(len(dRet['data']),0,'Should be empty. Got %d bits' %(len(dRet['data'])))
        
        # insert new data
        d = generateRandomWorkerData(N)
        data2 = np.array(d['data'],dtype=DATATYPE)
        trust2 = np.array(d['trust'],dtype=TRUSTTYPE)
        worker.insertData(d)
        time.sleep(0.02)

        # only returns the newest data
        dRet = worker.getSelf()
        worker.removeOldData()
        
        dOut, tOut = dRet['data'], dRet['trust']
        
        self.assertEqual(len(data2),len(dOut),'Only the newest packet should be gotten (len data2 %d len dOut %d)'%(len(data2),len(dOut)))
        self.assertEqual(len(trust2),len(tOut),'Only the newest packet should be gotten')
        self.assertTrue(np.all(dOut==data2),'bits should remain unchanged')
        self.assertTrue(np.all(tOut==trust2),'trust should remain unchanged')

        dRet = worker.getSelf()

        print('head %d\t tail %d'%(worker.head,worker.tail))
        self.assertEqual(len(dRet['data']),0,'Expected nothing,since no new data was added')
        self.assertEqual(len(dRet['trust']),0,'Expected nothing,since no new data was added')

        # Now all besides the last arrival should be removed
        time.sleep(T)
        dRet = worker.getSelf()
        worker.removeOldData()
        arrivalTimes = worker.arrivalTimes
        self.assertEqual(len(arrivalTimes),1,'everything besides the newest data should have been removed')

        del worker

    def testInsertAndGetByMultipleSlaves(self):
        """
        Checks the following with a number of slaves:
        Gets it's own data within the desired borders returned
        Checks if data gets removed when old
        Checks if the proper data is returned
        """

        T = 0.05 # short for testing
        N = 1000
        noPackets = 5
        data = np.array([] ,dtype=DATATYPE)
        trust = np.array([],dtype=TRUSTTYPE)
        d = generateRandomWorkerData(N)

        worker = softCombiner.Worker(d,timestampTimeOut = T)
        data = np.r_[data,np.array(d['data'],dtype=DATATYPE)]
        trust = np.r_[trust,np.array(d['trust'],dtype=TRUSTTYPE)]
        time.sleep(0.02)
        for i in range(noPackets - 1):
            d = generateRandomWorkerData(N)
            data = np.r_[data,np.array(d['data'],dtype=DATATYPE)]
            trust = np.r_[trust,np.array(d['trust'],dtype=TRUSTTYPE)]
            worker.insertData(d)
            time.sleep(0.02)

        workerId1 = 'w1'
        workerId2 = 'w2'
        self.assertEqual(len(worker.slaves),0,'Expected no slaves to be present')
        self.assertEqual(worker.activeSlave,None,'no active slave should be registered')
        data1 = worker.getSelf(workerId1)

        self.assertEqual(len(worker.slaves),1,'Expected one slave to be present')
        self.assertEqual(worker.activeSlave.workerId,workerId1,'active slave1 should be registered')
        # check head and tail
        self.assertEqual(worker.activeSlave.head,worker.activeSlave.tail,'head should equal tail') 
        self.assertEqual(worker.activeSlave.head,noPackets*N,'head and tail should point to the end of the buffer') 
        
        data2 = worker.getSelf(workerId2)
        self.assertEqual(len(worker.slaves),2,'Expected two slaves to be present')
        self.assertEqual(worker.activeSlave.workerId,workerId2, 'active slave2 should be registered')

        # check head and tail
        self.assertEqual(worker.activeSlave.head,worker.activeSlave.tail,'head should equal tail') 
        self.assertEqual(worker.activeSlave.head,noPackets*N,'head and tail should point to the end of the buffer') 

        
        # Retrieved data should be noPackets * N bits long
        self.assertEqual(len(data1['data']),noPackets*N,'length does not fit')
        self.assertEqual(len(data2['data']),noPackets*N,'length does not fit')
        
        # all data should be equal:
        self.assertTrue(np.all(data1['data']==data2['data']),'data for two slaves should be equal')
        self.assertTrue(np.all(data1['trust']==data2['trust']), 'trust for two slaves should be equal')


        
        # should be empty:
        data2 = worker.getSelf(workerId2)
        self.assertTrue(len(data2['data'])==0,'Length of data for slave should be 0 since no new data is added')
        
        worker.removeOldData()
        dataw = worker.getSelf()
        # Here we expect no data, since the removeOldData sets the head and tail further ahead
        self.assertTrue(len(dataw['data'])==0,'Length of data should be 0 after removeOldData()')
        self.assertEqual(worker.activeSlave,None,'no active slave should be registered')

        

        ## insert new data
        worker.insertData(d)
        worker.removeOldData() # should not remove any unused data
        dataw = worker.getSelf()
        self.assertTrue(np.all(dataw['data']==d['data']),'all data should be identical to what is submitted')
        self.assertEqual(len(d['data']),len(dataw['data']),'expected %d bits, not %d' %(len(d['data']), len(dataw['data'])))
        
        data1 = worker.getSelf(workerId1)
        self.assertTrue(np.all(data1['data']==d['data']),'all data should be identical to what is submitted')
        self.assertEqual(len(d['data']),len(data1['data']),'expected %d bits, not %d' %(len(d['data']), len(data1['data'])))

        data2 = worker.getSelf(workerId2)
        self.assertTrue(np.all(data2['data']==d['data']),'all data should be identical to what is submitted')
        self.assertEqual(len(d['data']),len(data2['data']),'expected %d bits, not %d' %(len(d['data']), len(data2['data'])))

        
        # Change index in workerId2
        cutN = 300
        worker.updateIdx(cutN)

        self.assertEqual(worker.activeSlave.workerId,workerId2,'Expected to be editing worker2')
        self.assertEqual(worker.activeSlave.tail-worker.activeSlave.head,cutN,'head should be %d shorter than the current data (len %d)'%(cutN,len(d['data'])))
        self.assertEqual(worker.activeSlave.tail,len(worker.data['data']),'tail should point to the end of the worker data')
        
        worker.insertData(d)
        worker.removeOldData() # should not remove any unused data
        
        dataw = worker.getSelf()
        self.assertTrue(np.all(dataw['data']==d['data']),'all data should be identical to what is submitted')
        self.assertEqual(len(d['data']),len(dataw['data']),'expected %d bits, not %d' %(len(d['data']), len(dataw['data'])))
        
        data1 = worker.getSelf(workerId1)
        self.assertTrue(np.all(data1['data']==d['data']),'all data should be identical to what is submitted')
        self.assertEqual(len(d['data']),len(data1['data']),'expected %d bits, not %d' %(len(d['data']), len(data1['data'])))

        # worker 2 should now submit cutN more bits than the length of d
        data2 = worker.getSelf(workerId2)
        self.assertTrue(np.all(data2['data'][cutN:]==d['data']),'all data should be identical to what is submitted')
        self.assertEqual(len(d['data'])+cutN,len(data2['data']),'expected %d bits, not %d' %(len(d['data'])+cutN, len(data2['data'])))

        del worker
        
if __name__ == '__main__':

    loadConfig.getConfigAndLog('conf_test.json')
    unittest.main()

