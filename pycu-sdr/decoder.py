# Copyright: (c) 2021, Edwin G. W. Peters

from __global__ import *

import numpy as np
import time
import logging
import string
from enum import Enum
from protocol import PacketEndDetect, PacketLenEndianness

printableChars = set(string.printable)

log	= logging.getLogger(LOG_NAME+'.'+__name__)

class Decoder:
    ## class methods for findFrames:
    maxPacketLenBits = int(2**13)    # max packet length probably longer, but this is our limit
    minNumBitsBeforeProcessing = int(2**10)
    def __init__(self,config,protocol):

        self.conf = config
        self.protocol = protocol
        log.info(f'protocol {protocol.name}')
        self.preprocessor = self.protocol.decoderPreprocessor # does descrambling, dewhitening, NRZS etc
        self.postprocessor = self.protocol.decoderPostprocessor # TODO: does destuffing and all that
        
        self.mask = protocol.get_mask()
        self.syncSig = protocol.get_syncFlag()

        # Buffers
        self.numBitsOverlap = protocol.numBitsOverlap # Set by protocol. Should be at least the lenght of sync in bits
        self.bitsOverlapBuf = np.zeros(self.numBitsOverlap)  

        # state machine and buffers for packet reception
        self.headerFrameStartIdx = None
        self.packetBuffer = None          # Used to store bits across block boundaries
        self.headerMaskBitErrors = None

        self.packetEndDetectMode = protocol.packetEndDetectMode
        self.packetEndLenDecoder = protocol.packetEndLenDecoder

        # FLAGS
        self.packetSizes = protocol.packet_sizes

        # FIXED
        self.packetLen = protocol.packetLen # for fixed size packets only

        # IN_DATA
        self.packetEndLenField = protocol.packetEndLenField
        self.packetEndLenFieldNumBytes = protocol.packetEndLenFieldNumBytes
        
        self.Packet = protocol.Packet # class for storing the received data


        # debug to store the raw continuous bitstream
        if STORE_BITS_IN_FILE is True:
            self.all_bits = np.empty(0,dtype=np.int8)
            self.frames = np.empty(0,np.int)

        if self.packetEndDetectMode == PacketEndDetect.FIXED:
            log.info(f'Packet ends detected using {self.packetEndDetectMode.name} with packet length {self.packetLen} bits')
        elif self.packetEndDetectMode == PacketEndDetect.FLAGS:
            log.info(f'Packet ends detected using {self.packetEndDetectMode.name} with packet lengths {self.packetSizes} bits')
        elif self.packetEndDetectMode == PacketEndDetect.IN_DATA:
            log.info(f'Packet ends detected using {self.packetEndDetectMode.name} with packet length field at location {self.packetEndLenField} being {packetEndLenFieldNumBytes} bytes long')
            
    def findFrames(self,bits_raw, frameStartIdx, debugMode = False):
        """ This method finds packets based on the sync specified by the protocol
        bits_raw - is the raw bits coming from the descrambler
        thresh - is the allowed number of bit errors that we allow in a header match
        debugMode - can be used to trigger plotting of packets etc
    
        Returns:
        packets  - a list of objects with containing received objects
        sig_DS - descrambled bits
        numSyncSig - number of sync signal matches        
        """

        # do the preprocessing (descrambling, deNRZS, dewhitening etc. based on the protocol)
        bits_less_raw = self.preprocessor(bits_raw)

        if STORE_BITS_IN_FILE is True:
            self.all_bits = np.append(self.all_bits,bits_less_raw.astype(np.int8))
            self.frames = np.append(self.frames,len(self.all_bits))
            
            np.savez('../bits',self.all_bits,self.frames)

        rawBits_DS = np.concatenate((self.bitsOverlapBuf, bits_less_raw))
        self.bitsOverlapBuf = rawBits_DS[-self.numBitsOverlap:] # store the end of the frame in the buffer
        # if len(rawBits_DS) < self.numBitsOverlap:
        #     print(f'not enough bits in buffer -- len buf {len(rawBits_DS)}, overlap buf len {len(self.bitsOverlapBuf)} numBitsOverlap {self.numBitsOverlap}')
        
        # find packet candidates
        t = time.time()
        score = np.convolve(rawBits_DS,self.mask)
        if log.level == logging.DEBUG:
            log.debug('convolve time {} s'.format(time.time()-t))
        
        
        idxCand = np.where(score >= self.protocol.numOnesHeader - self.protocol.headerTol)[0] # only keep good matches

        # the spikes appear at the last bit of the mask. subtract the length of the test sequence
        packetIdx = idxCand - len(self.mask) + 1
        if log.level == logging.DEBUG:
            if len(idxCand) > 0:
                log.debug('Packet cand idx ' + str(idxCand) + '\t offset removed ' + str(packetIdx))
    
        # find the ends of the packet candidates.
        # The end is based on the first appearance of the frame

        syncSigs  = np.convolve(rawBits_DS.astype(np.int),self.syncSig)
        syncSigStartIdx = np.where(syncSigs >= self.protocol.numOnesSyncSig-self.protocol.syncSigTol)[0] 
        numSyncSig = len(syncSigStartIdx)

        if log.level == logging.DEBUG:
            log.debug('syncSigs ' + str(numSyncSig) +'\tpackets  ' + str(len(packetIdx)))

        tp = time.time()
        packets = []

        if self.packetEndDetectMode == PacketEndDetect.FLAGS:
            ## There are three end of packet conditions:
            # Sync flags start 
            # Read packet length field, if known where to look for it TODO
            # Fixed packet length TODO

            if self.headerFrameStartIdx != None:
                ## Resume frame from previous block
                # 2 cases:
                #     End of frame is in this block
                #     End of frame is not in this block
                if log.level == logging.DEBUG:
                    try:
                        log.debug('resuming frame. Current number of bits ' + str(len(self.packetBuffer)) + ' last bits in buffer ' + str(self.packetBuffer[-16:]))
                    except Exception as e:   #  just if packetBuffer is empty for some reason
                        log.debug('resuming frame. Current number of bits ' + str(len(self.packetBuffer)))

                # We have a header from the previous block
                # first take care of the packet in the buffer:
                firstFrameOffset = 0
                if log.level == logging.DEBUG:
                    log.debug('firstFrameOffset ' + str(firstFrameOffset))
                if numSyncSig == 0:
                    log.debug('no syncSig')
                    frameEnd = []
                else:
                    frameEndIdx = np.argmax(syncSigStartIdx > firstFrameOffset)
                    if log.level == logging.DEBUG:
                        log.debug('frameEndIdx ' + str(frameEndIdx))
                    if syncSigStartIdx[frameEndIdx] < self.protocol.numOnesSyncSig-self.protocol.syncSigTol:
                        frameEnd = []
                    else:
                        frameEnd = [np.min((syncSigStartIdx[frameEndIdx]+16,syncSigStartIdx[-1]))] # 8 bits of sync signal for checking
                if log.level == logging.DEBUG:
                    log.debug('frameEnd ' + str(frameEnd))
                if len(frameEnd) == 0:
                    # The end of this packet is not in this block, store
                    packetLenTol = self.maxPacketLenBits - len(self.packetBuffer)
                    if packetLenTol > len(bits_less_raw):
                        if log.level == logging.DEBUG:
                            log.debug('appended %d bits' %(len(bits_less_raw)))
                        try:
                            if log.level == logging.DEBUG:
                                log.debug('first bits ' + str(bits_less_raw[:8]))
                        except Exception as e:
                            pass
                        self.packetBuffer = np.append(self.packetBuffer,bits_less_raw)
                    else:
                        # Max length exceeded. create a struct and assume packet is finished
                        if log.level == logging.DEBUG:
                            log.debug('exceed max len')
                        np.append(self.packetBuffer,bits_less_raw[:packetLenTol])

                        packets.append(self.Packet(self.packetBuffer,
                                                   self.headerFrameStartIdx,
                                                   self.headerMaskBitErrors))
                        self.headerFrameStartIdx = None # for the state machine
                else:
                    # end of packet found:
                    numBitsInPreviousFrame = len(self.packetBuffer)
                    self.packetBuffer = np.append(self.packetBuffer,rawBits_DS[self.numBitsOverlap:frameEnd[0]])
                    # Create packet struct
                    packets.append(self.Packet(self.packetBuffer,
                                               self.headerFrameStartIdx,
                                               self.headerMaskBitErrors,
                                               frameSplitIdx = numBitsInPreviousFrame))
                    if log.level == logging.DEBUG:
                        log.debug('resumed frame finished\tlen: ' + str(len(self.packetBuffer)))                
                    self.headerFrameStartIdx = None # for the state machine



            if self.headerFrameStartIdx == None:
                ## No header from previous frames
                # Loop through headers and find ends of packets
                #    Either finds end of frame by sync sig
                #    Or frame does not end in this block
                for i in range(len(packetIdx)):
                    if log.level == logging.DEBUG:
                        log.debug('New header at index ' + str(packetIdx[i]))
                    if len(syncSigStartIdx) == 0:
                        frameEnd = []
                        if log.level == logging.DEBUG:
                            log.debug('No syncsigs in stream. syncSigStartIdx ' + str(syncSigStartIdx))
                    else:
                        # This method is fast, but if there is no synchronisation signal, it will return the index with best fit
                        frameEndIdx = np.argmax(syncSigStartIdx > packetIdx[i] + 120)
                        if log.level == logging.DEBUG:
                            log.debug('frameEndIdx ' + str(frameEndIdx))
                        if frameEndIdx == 0:
                            # no syncsignal found after header
                            frameEnd = []
                        elif syncSigStartIdx[frameEndIdx] < self.protocol.numOnesSyncSig-self.protocol.syncSigTol:
                            # verify that we actually found a synchronisation signal with tolerance
                            frameEnd = []
                        else:
                            # If we made it this far, then we found the end of the packet
                            # frameEnd = [syncSigStartIdx[frameEndIdx]-15-8] # Without any trialling bits
                            if log.level == logging.DEBUG:
                                log.debug('finding the minimum between ' + str(syncSigStartIdx[frameEndIdx]+16) + ' and ' + str(syncSigStartIdx[-1]))
                            frameEnd = [np.min((syncSigStartIdx[frameEndIdx]+16,syncSigStartIdx[-1]))] # With 8 trialling bits

                    log.debug('frameEnd\t' + str(frameEnd))


                    if len(frameEnd) == 0:
                        # The end of this frame is not found in this block:
                        if log.level == logging.DEBUG:
                            log.debug('End of packet not in this block')
                        self.packetBuffer = rawBits_DS[packetIdx[i]:]
                        self.headerFrameStartIdx = frameStartIdx + packetIdx[i]-self.numBitsOverlap
                        self.headerMaskBitErrors = self.protocol.numOnesHeader - score[idxCand[i]]
                    else:
                        ## end of packet found:
                        # Create packet struct
                        bits = rawBits_DS[packetIdx[i]:frameEnd[0]]
                        if log.level == logging.DEBUG:
                            log.debug('End of packet\tlen: ' + str(len(bits)))
                        if len(bits) >= 128: # Minimum frame length
                            packets.append(self.Packet(bits,
                                                       packetIdx[i] + frameStartIdx,
                                                       self.protocol.numOnesHeader - score[idxCand[i]]))

        elif self.packetEndDetectMode == PacketEndDetect.FIXED:
            """
            Fixed length packets
            """
            for i in range(len(packetIdx)):
                if log.level == logging.DEBUG:
                    log.debug('New header at index ' + str(packetIdx[i]))
                ts = time.time()
         
                if len(rawBits_DS) - packetIdx[i] < self.packetLen:
                    ## we have to wait for more data. postpone search
                    # ensure that all potential packet bits are added to the overlap buffer
         
                    startIdx = max((0,packetIdx[i]-20)) # take 20 extra bits before the location of the match
                    if len(rawBits_DS) - startIdx > self.numBitsOverlap:
                        self.bitsOverlapBuf = rawBits_DS[startIdx:]
                    if log.level == logging.DEBUG:
                        log.debug('Need more data in this packet')
                    break # all other headers occur later in the bit stream so will also not be processed
                else:
                    # enough bits are present to get the packet
                    bits = rawBits_DS[packetIdx[i]:packetIdx[i]+self.packetLen]
                    if len(bits) > 0: # somehow we sometimes get length zero packets
                        if log.level == logging.DEBUG:
                            log.debug(f'Found full packet -- number of bits {len(bits)}')
                        t = time.time()
                        packets.append(self.Packet(bits,
                                                   packetIdx[i],
                                                   self.protocol.numOnesHeader - score[idxCand[i]]))
                        if log.level == logging.DEBUG:
                            log.debug(f'time rawbits {(time.time()-t)*1000}ms')
                   
                    else:
                        log.error(f'length of bits = 0. len(rawBits_DS) = {len(rawBits_DS)}, idx start {packetIdx[i]} idx end {packetIdx[i] + self.packetLen}')
                   
                log.debug(f'packet time {(time.time()-ts)*1000}ms')


        elif self.packetEndDetectMode == PacketEndDetect.IN_DATA:
            for i in range(len(packetIdx)):
                if log.level == logging.DEBUG:
                    log.debug('New header at index ' + str(packetIdx[i]))
                # we have to dewhiten the data and then find the packet length
                bitsDec = self.protocol.packetDataProcessor()

        if log.level == logging.DEBUG:
            if len(packets) > 0:
                log.info(f'making packets time {(time.time()-tp)*1000}ms')
        return packets, bits_less_raw, numSyncSig

