# Copyright: (c) 2021, Edwin G. W. Peters

"""
Benchmark the SDR and measure BER

See README.md for instructions on how to run

Parameters can be edited in main. Remember to match the config file used for the SDR

"""

import numpy as np
import asyncio
import zmq
import zmq.asyncio
zmq.asyncio.install() # pyzmq < 17 needs this 

import logging
import time, sys, os

from create_signals import *

OUT_DIR = 'bench_logs/'

nDemodulators = 1 # can be set higher if multiple demodulators shall vote

BASEPORT = 5560
sockets_addr = [f'tcp://*:{BASEPORT + i}' for i in range(nDemodulators)]

class SendSignal():

    chunkSize = int(2**14) # just want it to be smaller than the block size in the modem to mimic GNU radio behaviour

    modemIn =  "tcp://127.0.0.1:5512"
    def __init__(self,sig,repeats,rate, bitData,SNR):
        """
        signal,
        number of loops
        sample rate
        """

        self.sig = sig.astype(np.complex64)
        self.repeats = repeats
        self.rate = rate
        self.bitData = bitData
        self.SNR = SNR
        
        self.delayTime = 1/rate*self.chunkSize # seconds

        self.isRunning = True

        self.tasks = asyncio.gather(self.sendToModem(),self.receiveFromModem())


        
    async def sendToModem(self):

        ctx = zmq.asyncio.Context()


        sockets = []
        for a in sockets_addr:
            sock = ctx.socket(zmq.PUB)
            sock.bind(a)
            sockets.append(sock)
        
        time.sleep(2) # wait for receive socket to be up
        
        N = len(self.sig)
        Nloops = N//self.chunkSize
        preLenBlocks = 5

        log.debug('sending pre data')
        async def sendBuffer():
            for r in range(preLenBlocks):
                for s in sockets:
                    dat = np.sqrt(0.1)*np.random.randn(self.chunkSize).astype(np.complex64)
                    await s.send(dat)
                    
                await asyncio.sleep(self.delayTime)
            
        await sendBuffer()
                
        for r in range(self.repeats):
            log.debug(f'repeat {r}')
            sigs = []
            for i in range(len(sockets)):
                sigs.append(awgn(self.sig,self.SNR).astype(np.complex64))

            for n in range(Nloops):
                idx = slice(n*self.chunkSize,(n+1)*self.chunkSize)
                for i,s in enumerate(sockets):
                    await s.send(sigs[i][idx])

                await asyncio.sleep(self.delayTime)

            if N > Nloops*self.chunkSize:

                for i,s in enumerate(sockets):
                    await s.send(sigs[i][Nloops*self.chunkSize:])

                time.sleep(self.delayTime)

        log.debug('sending end data')
        await sendBuffer()

        # log.info('sendToModem finished -- sleep before shutting down')
        await asyncio.sleep(2) # wait before shutting down receiver
        self.isRunning = False
        for s in sockets:
            s.close()

        log.info('sendToModem done')


    async def receiveFromModem(self):
        
        ctx = zmq.asyncio.Context()
        socket = ctx.socket(zmq.PULL)

        socket.connect(self.modemIn)

        poller = zmq.asyncio.Poller()
        poller.register(socket,zmq.POLLIN)

        timeOut = 3
        self.pktCNT = 0
        self.bitErrors = []
        while self.isRunning:

            s = await poller.poll(timeOut)
            if len(s) > 0 and s[0][1] == zmq.POLLIN:
                dataRaw = await socket.recv(zmq.NOBLOCK)
                data = np.frombuffer(dataRaw,np.int8)
                self.pktCNT += 1

                bitErrorsT  = len(np.where(data != self.bitData)[0])
                self.bitErrors.append(bitErrorsT)
                
                log.info(f'received packet number {self.pktCNT}\tbit errors {bitErrorsT}\tBER {bitErrorsT/len(self.bitData)}')

        if self.pktCNT > 0:
            BER = np.mean(np.array(self.bitErrors)/len(self.bitData))
        else:
            BER = -1
        socket.close()
        log.info(f'receiveFromModem done -- received {self.pktCNT}/{self.repeats} packets\tavg. BER {BER}')


def print_help():
    helpstr = """
    benchmark_sdr.py modscheme N SNR_low SNR_high SNR_step
    where:
        modscheme is the modulation sheme (FSK, GMSK, BPSK, GFSK)
        N is the number of simulations to run
        SNR_low is the lowest test SNR
        SNR_high is the highest test SNR
        SNR_step is the step size of the SNRs to loop through
    """
    print(helpstr)
    sys.exit(-1)
        
if __name__ == "__main__":

    if not len(sys.argv) == 6:
        print_help()
    modulation = sys.argv[1]
    nRuns = int(sys.argv[2])
    SNR_low = float(sys.argv[3])
    SNR_high = float(sys.argv[4])
    SNR_inc = float(sys.argv[5])

    if not modulation in ['GMSK','FSK','BPSK','GFSK']: # invalid modulation exit
        print_help()

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
        
    logName = '{}/{}_{}'.format(OUT_DIR,time.strftime("%Y_%m_%d_%H_%M", time.gmtime(time.time())), f'_bench_{modulation}.log')

    saveName = '{}/{}_{}'.format(OUT_DIR,time.strftime("%Y_%m_%d_%H_%M", time.gmtime(time.time())), f'_bench_results_{modulation}')

    FORMAT = '%(asctime)-16s %(message)s'
    log = logging.getLogger()
    logging.Formatter.converter = time.gmtime
    logFormatter = logging.Formatter(FORMAT,  "%Y-%m-%d %H:%M:%S")
    consoleHandler = logging.StreamHandler(sys.stdout)
    log.addHandler(consoleHandler)
    fileHandler = logging.FileHandler(logName)
    consoleHandler.setFormatter(logFormatter)
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)
    log.setLevel(logging.DEBUG)

    log.info(f'Benchmarking modulation scheme {modulation} over {nRuns} tests')

    
    spSym = 16
    baud = 9600
    fs = spSym*baud
    fsSim = fs * 10//nDemodulators # simulation speedup Can be set as high as the SDR can keep up
  
    bw = baud/0.7
    bw_gmsk = baud/0.7
    bw_bpsk = baud*1.5 # rrcos with beta = 0.5
    fsk_delta_f = baud/2
    bw_fsk = 2*baud + 2*fsk_delta_f # is this correct for EBN0 though? Only one frequency contains the power at one time
    bw_gfsk2 = 2*baud + 2*fsk_delta_f # is this correct for EBN0 though? Only one frequency contains the power at one time

    

    sig,bitData = get_padded_packet(modulation,spSym,fs)

        


    # SNRs = range(15,16,1) # for quick test
    SNRs = np.arange(SNR_low,SNR_high+SNR_inc,SNR_inc)
    numPackets = []
    bitErrors = []
    BER = []
    EBN0 = []
    loop = asyncio.get_event_loop()

    for snr in SNRs:
        log.info(f'Running bench with SNR {snr} dB')
        if modulation == 'GMSK':
            SNR_r = snr + 10*np.log10(bw_gmsk/fs) # for generating AWGN, the bandwidth and oversampling rate need to be taken into account
            bw = bw_gmsk
        elif modulation == 'FSK':
            SNR_r = snr + 10*np.log10(bw_fsk/fs) # for generating AWGN, the bandwidth and oversampling rate need to be taken into account
            bw = bw_fsk
        elif modulation == 'GFSK': 
            SNR_r = snr + 10*np.log10(bw_fsk/fs) # for generating AWGN, the bandwidth and oversampling rate need to be taken into account
            bw = bw_fsk
        elif modulation == 'BPSK':
            SNR_r = snr + 10*np.log10(bw_bpsk/fs) # for generating AWGN, the bandwidth and oversampling rate need to be taken into account
            bw = bw_bpsk
        else:
            print_help() # not recognized
            
        sigOut = SendSignal(sig,nRuns, fsSim ,bitData,SNR_r)
    
        loop.run_until_complete(sigOut.tasks)
        
        numPackets.append(sigOut.pktCNT)
        bitErrors.append(sigOut.bitErrors.copy())
        
        EBN0.append(snr+10*np.log10(bw/baud))
        if len(bitErrors[-1]) > 0:
            # mBER = np.median(np.array(bitErrors[-1]))
            # sBER = np.std(np.array(bitErrors[-1]))
            # BERval = np.array(bitErrors[-1])
            # valBER = BERval < np.max((mBER+0.05*sBER, 50))
            # log.info(f'mean BER {mBER} std BER {sBER} threshold {mBER+sBER}')
            BER.append(np.mean(np.array(bitErrors[-1])/len(bitData)))
            # if np.sum(valBER) > 0 :
                # BER.append(np.mean(BERval[valBER])/len(bitData))
            log.info(f'Corrected BER {BER[-1]}')
        else:
            BER.append(1)

        del sigOut

    loop.close()

    
    for S,E,r,B in zip(SNRs,EBN0,numPackets,BER):
        log.info(f'SNR {S} dB:\tEB/N0 {E:.2f} dB\tpackets {r}\tavg. BER {B} ')
    


    np.savez(saveName,
             SNR = SNRs,
             EBN0 = EBN0,
             bitErrors = bitErrors,
             numPackets = numPackets,
             BER = BER,
             nRuns = nRuns,
             bitData = bitData,
             lenBitData = len(bitData),
             fs = fs,
             baud = baud)
