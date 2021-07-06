# Copyright: (c) 2021, Edwin G. W. Peters

from __global__ import *
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import lib
from lib import cufft
import time
import logging
import os
import scipy
from enum import Enum


log	= logging.getLogger(LOG_NAME+'.'+__name__)
# log.setLevel(logging.INFO)

# Defaults for symbol overlap test
SYMBOL_CHECK_OVERLAP_OFFSET = 20
SYMBOL_CHECK_ERROR_THRESHOLD = 1000
SYMBOL_CHECK_MATCH_NUM_ERRORS_ALLOWED = 10


# only used for the nrz-s decoder. Used for BPSK at the moment, since we there use nrz-s to resolve the ambiguity. Ideally we should look at the phase and demodulate directly
SYMBOL_MISMATCHVAL = 0 # 0 or -1.  0 is better for softCombiner

# STORE_BITS_IN_FILE defined in __global__.py
BITS_FNAME = '../data/bits_file'

class Operations(Enum):
    CENTRES_ABS = 0;
    CENTRES_REAL = 1;
    CENTRES_IMAG = 2;

class Demodulator:

    ## GPU performance specific
    numThreads = 64 # for fft and other elementwise ops
    numThreadsS = 1024 # has to be 32**2 for the scan sum kernel
    batchSize = 1 # Not implemented yet, but usefull for batch fft processing at some stage
    
    # other GPU based parameters
    num_streams = 3 # For big IFFT kernel 3 is the magic number
    managed_memory_available = False # At the moment not used, but nice to check

    kernelFilePath = 'cuda_kernels.cu'
    num_SMs = None

    def __initInstanceVariables(self):
        """
        initialize instance variables
        """

        ## list of buffers and fft plans (used in __del__ to free all)
        self.GPU_buffers = []
        self.GPU_fftPlans = []

        ## Demodulation
        self.sigOverlap = 2**8
        self.sigOverlapWin = int(self.sigOverlap/2)

        # Window over which to find centres
        self.windowWidth = 3 # must be odd. 
        self.windowWidthOffset = int(self.windowWidth/2) # floor(windowWidth/2) on each side
        self.clippedPeakIPure = []       # Indices of the clipped peaks
        self.clippedPeakI = []           # Indices of clipped peaks with window around
        # constant stuff
        self.bitLUT = None # Provided by protocol
        self.symbolLUT = None # Provided by protocol

        self.poswinP = [] # stores bits across calls -- filled in the demodulator

    
    def __init__(self,conf,protocol,radioName):
        
        # log.setLevel(logging.DEBUG)
        self.__initInstanceVariables() # initialize instance variables and arrays


        global BITS_FNAME
        BITS_FNAME = f'{BITS_FNAME}_{radioName}'
        self.protocol = protocol 
        self.confRadio = conf['Radios']['Rx'][radioName]
        self.confGPU = conf['GPU'][self.confRadio['CUDA_settings']]
        self.radioName = radioName
        
        # find signal size,
        self.sigLen = 2**self.confGPU['blockSize']
        self.sigOverlap = 2**self.confGPU['overlap']
        self.sigOverlapWin = int(self.sigOverlap/2)
        self.clippedPeakSpan = self.confGPU['clippedPeakSpan'] # number of symbols to disqualify around a peak
        self.peakThresholdScale = self.confGPU['peakThresholdScale'] # thresholding factor for peak clipping
        self.disablePeakThresholding = self.confRadio.get('disablePeakThresholding',False)
        
        # for symbol overlap check
        self.overlapOffset = self.confGPU.get('symbol_check_overlap_offset',SYMBOL_CHECK_OVERLAP_OFFSET)
        self.symbol_check_error_threshold = self.confGPU.get('symbol_check_error_threshold',SYMBOL_CHECK_ERROR_THRESHOLD)
        self.symbol_check_match_threshold = self.overlapOffset - self.confGPU.get('symbol_check_match_num_errors_allowed',SYMBOL_CHECK_MATCH_NUM_ERRORS_ALLOWED)
        
        log.info(f'[{self.radioName}]: symbol_check_overlap_offset {self.overlapOffset}, symbol_check_error_threshold {self.symbol_check_error_threshold}, symbol_check_match_threshold {self.symbol_check_match_threshold}')
            
        self.spsym = spsym = self.confRadio['samplesPerSym'] # samples per symbol
        self.spsymMin = int(spsym/2)                         # minimum allowed samples pr. symbol
        self.baudRate = self.confRadio['baud']
        self.sampleRate = self.baudRate * self.spsym
        try:
            self.voteWeight = self.confRadio['voteWeight'] # weight of voting for this channel
        except KeyError:
            self.voteWeight = 1
        log.info('[{}]: vote weight set to {}'.format(self.radioName,self.voteWeight))
            
        self.Nfft = int(self.sigLen) # this is the convolution block length
        
        # Window over which to find centres
        self.windowWidth = self.confGPU['bitWindowWidth'] # must be odd. 
        self.windowWidthOffset = int(self.windowWidth/2) # floor(windowWidth/2) on each side

        # code search offset. 0 includes all masks 1 excludes the only 0 and only 1 masks (tends to give better results)
        self.CODE_SEARCH_MASK_OFFSET = 0

        # If exists and true, then sum all masks prior to Doppler search. Better when not looking for a specific signature
        try:
            self.SUM_ALL_MASKS_PYTHON = protocol.SUM_ALL_MASKS_PYTHON
        except:
            self.SUM_ALL_MASKS_PYTHON = False
        log.info(f'[{self.radioName}]: Sum masks prior to Doppler search {self.SUM_ALL_MASKS_PYTHON}')
        
        # compute the indexes where to search the carrier based on rangerate and sample rate
        self.num_dopplers = self.confRadio['doppCarrierSteps']
        self.centreFreqOffset = self.confRadio['frequencyOffset_Hz'] # when the IF is offtuned
        Fc = self.confRadio['frequency_Hz'] - self.centreFreqOffset
        log.warning(f"Fc {self.confRadio['frequency_Hz']}, IF {Fc}")

        
        self.doppOffset = self.centreFreqOffset/self.baudRate/self.spsym
        self.doppOffsetIdx = np.int32(self.doppOffset * self.Nfft) # offset for STX mode
        if self.doppOffsetIdx < 0:
            self.doppOffsetIdx += self.Nfft

            
        rangeRateMax = conf['Radios']['rangeRateMax']
        doppMax = rangeRateMax * Fc/scipy.constants.speed_of_light
        doppMaxNorm = doppMax/self.sampleRate
        doppIdxMin = self.doppOffset - doppMaxNorm
        doppIdxMax = self.doppOffset + doppMaxNorm

        # channel to measure noise
        # noiseOfftuneHz = self.confRadio.get('noise_measure_offset_Hz',-self.centreFreqOffset)
        noiseOfftuneHz = self.confRadio.get('noise_measure_offset_Hz',False)
        if noiseOfftuneHz:
            noiseOfftuneIdx = noiseOfftuneHz/self.baudRate/self.spsym
            self.doppIdxNorm = np.concatenate((np.array([noiseOfftuneIdx]),np.linspace(doppIdxMin,doppIdxMax,self.num_dopplers)))
        else:
            self.doppIdxNorm = np.linspace(doppIdxMin,doppIdxMax,self.num_dopplers)
        
        # the fraction of the bw that contains the carrier
        self.doppIdxArrayLen = len(self.doppIdxNorm) # so we have the correct array length
        self.doppIdxArrayOffset = self.doppIdxArrayLen - self.num_dopplers

        # self.doppIdxNorm = np.linspace(0,0.5,self.num_dopplers)
        self.doppHzLUT = self.doppIdxNorm * self.spsym * self.baudRate # scaled by spsym (for compatibility)
        # Normalize the proposed doppler offsets
        self.doppCyperSymNorm = np.round(self.doppIdxNorm*self.Nfft).astype(np.int32)
        self.doppCyperSymNorm[self.doppCyperSymNorm < 0] += self.Nfft  # negative frequencies are at the right half of the spectrum
        
        log.info('[{}]: Fc {:.0f} Doppler scanning range {:.0f} to {:.0f} Hz of Fc, resolution {:.2f} Hz'.format(self.radioName,Fc,self.doppHzLUT[0],self.doppHzLUT[-1],(doppIdxMax-doppIdxMin)/self.num_dopplers*spsym*self.baudRate))

        
        # GPU parameters
        self.numThreads  = self.confGPU['CUDA']['numThreads']  # for fft and other elementwise ops
        self.numThreadsS = self.confGPU['CUDA']['numThreadsS'] # has to be 32**2 for the scan sum kernel
        self.batchSize   = self.confGPU['CUDA']['batchSize'] # Batch size for fft processing of inverse FFT's of cross correlations
        self.num_streams = self.confGPU['CUDA']['streams'] # For big IFFT kernel 3 is the magic number
    
        
        cuda.init()
        device = cuda.Device(self.confGPU['CUDA']['device'])
        self.getDeviceAttributes(device) # get relevant device attributes
        self.ctx = device.make_context()
        log.info('[{}]: Initializing Cuda on {}'.format(self.radioName,device.name()))

        self.warp_size = device.get_attribute(cuda.device_attribute.WARP_SIZE)
        # Not yet used, but maybe for future
        self.managed_memory_available = device.get_attribute(cuda.device_attribute.MANAGED_MEMORY)

        # GPU grid Shapes
        if self.Nfft/self.numThreadsS != np.round(self.Nfft/self.numThreadsS):
            raise ValueError('[{}]: the size of the input signal has to be divisible by {}'.format(
                self.radioName,self.numThreadsS))
        
   

        # generate masks TODO: masks need to be integrated based on protocol and radio 
        try:
            self.num_masks, masks = protocol.get_filter(self.Nfft,self.spsym,self.confGPU['xcorrMaskSize'])
            self.__uploadMaskToGPU(masks)
            if self.num_masks > 32: # warning for too many masks
                log.warning('[{}]: more than 32 masks is not supported at this time'.format(self.radioName))
            log.info('[{}]: mask len {} totalling {} masks'.format(self.radioName,self.confGPU['xcorrMaskSize'],self.num_masks))
            
        except Exception:
            log.error('[{}]: Exception occured in protocol {} while preparing filters'.format(
                self.radioName,protocol.name))
            raise
        try:
            self.bitLUT, self.symbolLUT = protocol.get_symbolLUT2(self.confGPU['xcorrMaskSize'])
        except Exception:
            log.error('[{}]: Exception occured in protocol {} while preparing symbol lookup table'.format(
                self.radioName,protocol.name))
            raise

        # load kernels from file
        self.CudaKernels = SourceModule(self.__loadKernelsFromFile())
        # allocate GPU buffers
        self.__initializeSharedBuffers()
        # load and prepare CUDA kernels
        self.__initializeKernels()
            
        # upload doppler shift indices to GPU
        cuda.memcpy_htod(self.GPU_bufDoppIdx,self.doppCyperSymNorm)
        
        # if we want to do local storing of bits for debugging
        if STORE_BITS_IN_FILE is True:
            log.warning('----- Storing demodulated bits in file. Can lead to performance issues for long passes! -----')
            import tables
            self.all_bits = np.empty(0,dtype=DATATYPE)
            self.frames = np.empty(0,np.int32)
            self.all_trust = np.empty(0,TRUSTTYPE)
            self.sum_match = np.empty((0,self.num_dopplers),np.float32)
            self.code_rate = np.empty(0,np.float32)
            self.code_phase = np.empty(0,np.float32)
            self.masks = masks
            
            # Xcorr output is stored in HDF5
            self.xcorrFile = tables.open_file(BITS_FNAME+'.h5',mode='w')
            self.xcorrResS = np.zeros((self.num_masks,self.Nfft),dtype=np.complex64)
            self.xcorrResS = np.expand_dims(self.xcorrResS,0)
            
            self.xcorrOut = self.xcorrFile.create_earray(self.xcorrFile.root,'xcorrOut',obj=self.xcorrResS)
 

        log.info('[{}]: Initialization done'.format(self.radioName))


    def __uploadMaskToGPU(self,masks):
        """
        This method uploads the set of filters that the protocol provides to the GPU
        """

        # do some type and shape checking
        if masks.shape != (self.num_masks,self.Nfft):
            raise ValueError("Masks provided by protocol {} expected to be of dimensions {}, got dimensions {}".format(self.protocol.name,
                                                                                                                       (self.num_masks,self.Nfft),
                                                                                                                      masks.shape))
        if not isinstance(masks[0,0],np.complex64):
            raise TypeError("Datatype of masks {}, expected {}".format(type(masks[0,0]),np.complex64))

        # allocate buffers
        self.GPU_bufBitsMask = cuda.mem_alloc(masks.nbytes)
        self.GPU_buffers.append(self.GPU_bufBitsMask) # ensure it gets freed at the end

        cuda.memcpy_htod(self.GPU_bufBitsMask,masks)
        
        
    def getDeviceAttributes(self,dev):
        self.CUDA_WARP_SIZE = dev.get_attribute(cuda.device_attribute.WARP_SIZE)
        self.CUDA_NUM_THREADS_PER_SMX = lib.ConvertSMVer2Cores(*dev.compute_capability())
        self.CUDA_NUM_SMX = dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
        self.CUDA_NUM_THREADS = self.CUDA_NUM_SMX * self.CUDA_NUM_THREADS_PER_SMX
        self.CUDA_NUM_WARPS = int(self.CUDA_NUM_THREADS/self.CUDA_WARP_SIZE)



    def __initializeDemodIfftPlan(self):
        """
        This is one batch of FFT's for the demodulation. Just needs to be as large as the number of masks
        """
        fftBatch = self.num_masks
        self.fftPlanDemod = cufft.cufftPlan1d(self.Nfft,cufft.CUFFT_C2C,fftBatch)
        self.GPU_fftPlans.append(self.fftPlanDemod)


    def __initializeSNRFftPlan(self):
        """
        This plan is used for computing the SNR
        """
        self.fftPlan = cufft.cufftPlan1d(self.Nfft,cufft.CUFFT_C2C,1)
        self.GPU_fftPlans.append(self.fftPlan)


    def __initializeDopplerIfftPlan(self):
        """
        Initialize the FFT plan(s) for the inverse fft(s) after convolution. These consume a lot of time
        Can run batched or non-batched.  Depending on the GPU and CUDA/driver version one may be faster than the other
        """
        
        
        fftBatch = self.doppIdxArrayLen*self.num_masks

        if self.confGPU['CUDA']['batchSize'] == 0:
            self.FFTNoBatches = 1
        else:
            noBatches = self.confGPU['CUDA']['batchSize']
            self.FFTNoBatches = int(fftBatch/noBatches)
            if self.FFTNoBatches != fftBatch/noBatches:
                log.error("FFT batch size has to be an integer divider of xcorrNumMasks * doppCarrierSteps")
                raise Exception("FFT batch size has to be an integer divider of xcorrNumMasks * doppCarrierSteps")
            

        FFTBatchSize = fftBatch/self.FFTNoBatches
        assert FFTBatchSize % 1 == 0, 'Invalid FFT batch size of %f FFT batch size has to be integer. Select a different number of batches' %(FFTBatchSize)
        self.FFTBatchSize = int(FFTBatchSize)
        
        log.debug('FFT batch size %d\t %d batches'%(self.FFTBatchSize,self.FFTNoBatches))
        if self.FFTNoBatches == 1:
            # We only need one transform
            self.fftPlanDopplers = cufft.cufftPlan1d(self.Nfft,cufft.CUFFT_C2C,fftBatch)
            self.GPU_fftPlans.append(self.fftPlanDopplers)
            self.buffAddr = [int(self.GPU_bufXcorr)]

        else:
            # more than one batch needed
            self.fftLoops = int(np.ceil(self.num_masks*self.doppIdxArrayLen/self.num_streams/self.FFTBatchSize))

            # We have to set up streams, input addresses and such
            self.streams, self.fftPlanDopplers = [], []
            for i in range(self.num_streams):
                self.streams.append(cuda.Stream())
                self.fftPlanDopplers.append(cufft.cufftPlan1d(self.Nfft,cufft.CUFFT_C2C,self.FFTBatchSize))
                cufft.cufftSetStream(self.fftPlanDopplers[i],int(self.streams[i].handle))

            self.GPU_fftPlans.extend(self.fftPlanDopplers)
            # input addresses to the batch
            batchOffset = np.complex64().nbytes*self.Nfft*self.FFTBatchSize
            self.buffAddr = []
            for i in range(self.FFTNoBatches):
                self.buffAddr.append(int(int(self.GPU_bufXcorr) + i*batchOffset))
        
        
    def __initializeKernels(self):
        """
        Initialize the cuda kernels and the block and grid dimensions
        """
        # FFT plans:
        self.__initializeDopplerIfftPlan() # for Doppler Ifft
        self.__initializeDemodIfftPlan() # for demod 
        self.__initializeSNRFftPlan() # for findSNR
        
        # GPU kernels
        kernel = self.CudaKernels
        ## kernels for initialization
        self.GPU_multInputVectorWithMasks = kernel.get_function('multInputVectorWithMasks').prepare('PPP')
        
        self.GPU_complexConj = kernel.get_function('complexConj').prepare('P')
        self.GPU_scaleComplexByScalar = kernel.get_function('scaleComplexByScalar').prepare('Pf')
        self.GPU_setComplexArrayToZeros = kernel.get_function('setComplexArrayToZeros').prepare('P')
        
        ## kernels for doppler search
        self.GPU_filterMasks = kernel.get_function('multInputVectorWithShiftedMasksDopp').prepare('PPPPii')
        # for multInputVectorWithShiftedMasks
        self.numBlocks = self.Nfft/self.numThreads
        self.bShapeVecMasks = (int(self.numThreads),1,1)
        self.gShapeVecMasks = (int(self.numBlocks),1)
        assert self.bShapeVecMasks[0]*self.gShapeVecMasks[0]==self.Nfft,'Dimension mismatch'

        self.GPU_absSumDoppler = kernel.get_function('blockAbsSumAtomic').prepare('PPi')
         # for the absSumKernel to sum the rows together
        self.bShapeAbsSum = (128,1,1) # 128 and 2 in next line is just picked TODO: should be config val
        self.gShapeAbsSum = (2,int(self.doppIdxArrayLen)) # tweak these

        assert self.Nfft % self.bShapeAbsSum[0]*self.gShapeAbsSum[0] == 0,'Nfft has to be dividable by block and grid dimensions'

        self.GPU_estDoppler = kernel.get_function('findDopplerEst').prepare('PPPii')
         # for the small kernel that finds the doppler
        self.bShapeDopp = (self.num_masks,1,1)
        self.gShapeDopp = (1,1)

        self.GPU_setArrayToZeros = kernel.get_function('setArrayToZeros').prepare('P')
        # for the set to zero kernel for the sum
        self.bShapeZero = (int(self.num_masks),1,1)
        self.gShapeZero = (int(self.doppIdxArrayLen),1)

        ## for demodulation
        self.bShapeVecMasks2 = (int(256),1,1) ## 256 is just picked, TODO: should be config val
        self.gShapeVecMasks2 = (int(self.Nfft/self.bShapeVecMasks2[0]),1)
        self.complexShiftMulMasks = kernel.get_function('multInputVectorWithShiftedMask').prepare('PPPi')
        self.complexHeterodyne = kernel.get_function('complexHeterodyne').prepare('PPfffi')
        self.findcentres = kernel.get_function('findCentres').prepare('PPPPffii')
        self.bShapeCentres = (256,1,1)  ## 256 is just picked, TODO: should be config val

        
        
    def __loadKernelsFromFile(self):
        currentPath = os.path.realpath(__file__).strip(os.path.basename(__file__))
        log.debug('loading CUDA kernels from file ' + currentPath)

        kernel_header_str = """
        #define WARP_SIZE {WARP_SIZE}// %(warp_size)i
        #define LOG_WARP_SIZE {LOG_WARP_SIZE} // saves us a log2 computation here
        #define NUM_WARPS {NUM_WARPS}
        #define FULL_MASK 0xffffffff // For shfl instructions
        #define SUM_ALL_MASKS {SUM_ALL_MASKS}
        
        #define NUM_MASKS {NUM_MASKS}
        #define WINDOW_WIDTH {WINDOW_WIDTH} 		// Number of samples around the centre to take into account
        #define WINDOW_LEFT_OFFSET WINDOW_WIDTH/2
        #define CODE_SEARCH_MASK_OFFSET {CODE_SEARCH_MASK_OFFSET} // used in sumXCorrBuffMasks  0: all masks are used, 1: exclude the 0 and 1 masks (first and last (old CPU setting))

        """.format(
            NUM_MASKS = self.num_masks,
            WINDOW_WIDTH=self.windowWidth,
            WARP_SIZE = self.CUDA_WARP_SIZE,
            LOG_WARP_SIZE = int(np.log2(self.CUDA_WARP_SIZE)),
            NUM_WARPS = np.min((self.CUDA_NUM_WARPS,32)),
            SUM_ALL_MASKS = int(self.SUM_ALL_MASKS_PYTHON), # bool has to be casted to int. Else C does not process it properly
            CODE_SEARCH_MASK_OFFSET = self.CODE_SEARCH_MASK_OFFSET,
        )
        # print(kernel_header_str)
        if len(currentPath) > 0 :
            self.kernelFilePath = '/' + currentPath + '/' + self.kernelFilePath

        with open(self.kernelFilePath,'r') as f:
            kernelCode = f.read()

        kernelCode = kernel_header_str + kernelCode
                
        log.info(f'code search offset: {self.CODE_SEARCH_MASK_OFFSET}')    
            
        return kernelCode

        
    def __initializeSharedBuffers(self):
        ## The last dimension is the flat one in memory
        # 3D array: len(doppCyperSym) x num_masks x Nfft 
        self.GPU_bufXcorr = cuda.mem_alloc(np.complex64().nbytes*self.Nfft*self.doppIdxArrayLen*self.num_masks)
        self.GPU_buffers.append(self.GPU_bufXcorr)

        # 2D array: len(doppCyperSym) x num_masks
        self.GPU_bufDoppSum = cuda.mem_alloc(np.float32().nbytes*self.doppIdxArrayLen*self.num_masks)
        self.GPU_buffers.append(self.GPU_bufDoppSum)
        
        # stores the result [mean, sdev]
        self.GPU_bufDoppResult = cuda.mem_alloc(np.float32().nbytes*2) 
        self.GPU_buffers.append(self.GPU_bufDoppResult)
        
        # buffer for the doppler shift indices. This can be omitted once the GPU computes them
        self.GPU_bufDoppIdx = cuda.mem_alloc(np.int32().nbytes*self.doppIdxArrayLen)
        self.GPU_buffers.append(self.GPU_bufDoppIdx)
        
        # for testing the find dopp
        self.GPU_bufFindDoppTmp = cuda.mem_alloc(np.float32().nbytes*self.num_masks)
        self.GPU_buffers.append(self.GPU_bufFindDoppTmp)
        
        # to keep the signal
        self.GPU_bufSignalTime_cpu_handle = cuda.pagelocked_empty((self.Nfft,), np.complex64, mem_flags=cuda.host_alloc_flags.DEVICEMAP)
        self.GPU_bufSignalTime = np.intp(self.GPU_bufSignalTime_cpu_handle.base.get_device_pointer())

        self.GPU_bufSignalFreq_cpu_handle = cuda.pagelocked_empty((self.Nfft,), np.complex64, mem_flags=cuda.host_alloc_flags.DEVICEMAP)
        self.GPU_bufSignalFreq = np.intp(self.GPU_bufSignalFreq_cpu_handle.base.get_device_pointer())
        
        # self.SNRDataSpecBuf = np.empty(self.Nfft,dtype=np.complex64) # to store the spectrum for computing the SNR

        # self.GPU_bufSignalFreq = cuda.mem_alloc(np.complex64().nbytes*self.Nfft)
        # self.GPU_buffers.append(self.GPU_bufSignalFreq)
        
        # demodulation buffers to find the centres and locations
        self.GPU_symbols = cuda.mem_alloc(np.int32().nbytes*int(self.Nfft/self.spsymMin))
        self.GPU_buffers.append(self.GPU_symbols)
        self.GPU_centres = cuda.mem_alloc(np.int32().nbytes*int(self.Nfft/self.spsymMin))
        self.GPU_buffers.append(self.GPU_centres)
        self.GPU_magnitude = cuda.mem_alloc(TRUSTTYPE().nbytes*int(self.Nfft/self.spsymMin))
        self.GPU_buffers.append(self.GPU_magnitude)
        # arrays local for each stream

        self._init_CodeRateAndPhaseBuffers()


    def _init_CodeRateAndPhaseBuffers(self):
        """
        This method initializes all CodeRateAndPhaseBuffers that are needed for the findCodeRateAndPhaseGPU routine
        """

        # Grid and blocks
        MAX_BLOCK_X = 1024 # current limit (32 warps of 32 threads = 1024 which is max block size x)
        numThreads = int(np.min((self.CUDA_WARP_SIZE * self.CUDA_NUM_WARPS, MAX_BLOCK_X))) # TODO: check with atomic operation to allow more than 32 warps
        
        self.block_codeMax = (numThreads,1,1)
        self.grid_codeMax = (1,1) # we only want one grid at the moment. Maybe 2 later, but need to implement atomic operation in the algorithm first!
        self.block_codeXCorrSum = (int(self.CUDA_NUM_THREADS/self.CUDA_NUM_SMX),1,1) # 1 block pr 
        self.grid_codeXCorrSum = (int(self.CUDA_NUM_SMX*4),1) # Make some more grids than SMX's, so there is plenty to do for the GPU while it is waiting for data

        ## buffers
        self.GPU_bufCodeAndPhase = cuda.mem_alloc(int(self.Nfft*np.float32().nbytes)) # output of sum is real, In place R2C fft uses Nfft/2*complex64 bytes which is same length as Nfft*float32
        self.GPU_bufCodeAndPhaseOut = cuda.mem_alloc(int(self.Nfft*np.complex64().nbytes)) # for arrays of len >= 2**20, the in-place R2C transform does not seem to work
        self.GPU_buffers.append(self.GPU_bufCodeAndPhase)
        self.GPU_bufCodeAndPhaseResult = cuda.mem_alloc(np.float32().nbytes*3) # holds [index of max (offset complensated), argument of max (normalized), value of max]
        self.GPU_buffers.append(self.GPU_bufCodeAndPhaseResult)
        
        ## fft Plan
        self.fftPlan_codeRate_R2C = cufft.cufftPlan1d(self.Nfft,cufft.CUFFT_R2C,1) # R2C, 1 batch
        self.GPU_fftPlans.append(self.fftPlan_codeRate_R2C)

        ## CUDA kernels
        self.GPU_sumXCorrBuffMasks = self.CudaKernels.get_function('sumXCorrBuffMasks').prepare('PPi')
        self.GPU_findCodeRateAndPhase = self.CudaKernels.get_function('findCodeRateAndPhase').prepare('PPii')
        ## constants
        self.symsTolLow = 0.9*self.spsym  # Ignore low symbol rates. avoids locking onto harmonics
        self.symsTolHigh = 1.1*self.spsym # Ignore high symbol rates (avoids false results due to slow noise and DC)

        self.codeRateAndPhaseOffsetLow = int(self.Nfft/self.symsTolLow)
        self.codeRateAndPhaseOffsetHigh = int(self.Nfft/self.symsTolHigh)
        ## numpy buffers
        self.__CodeRateAndPhaseResult = np.empty(3,dtype=np.float32) # holds [index of max (offset complensated), argument of max (normalized), value of max]


    def __del__(self):

        # free all buffers
        for buf in self.GPU_buffers:
            buf.free()

        # destroy all fft plans
        for plan in self.GPU_fftPlans:
            cufft.cufftDestroy(plan)

        try:
            self.ctx.pop()
        except AttributeError:
            pass

        if STORE_BITS_IN_FILE is True:
            self.xcorrFile.close()



    def uploadAndFindUHF(self,samples):
        """
        Performs:
            thresholding -- remove large spikes of interference
            uploadToGPU  -- store data on GPU and perform FFT
            findUHF      -- find the UHF modulated carrier
        """
        self.__thresholdInput(samples)
        self.uploadToGPU(samples)
        return self.__findUHF(samples)
        
    def uploadToGPU(self,samples):
        """
        Upload samples to GPU and perform FFT
        
        Data is in pinned memory and transfered directly
        """
        # load samples on to GPU and FFT
        # cuda.memcpy_htod(self.GPU_bufSignalTime,samples.astype(np.complex64))
        # self.GPU_bufSignalTime_cpu_handle[:self.Nfft]  = samples.astype(np.complex64)
        cufft.cufftExecC2C(self.fftPlan, int(self.GPU_bufSignalTime),
                   int(self.GPU_bufSignalFreq), cufft.CUFFT_FORWARD)

        
    def thresholdInput(self,samples):
        """
        calls thresholding function to reduce large spikes
        """
        self.__thresholdInput(samples)
        
    def __findUHF(self,samples):
        """
        Search for modulated carrier within a frequency range in the samples. Returns doppler estimate
        """    
        self.GPU_setArrayToZeros.prepared_call(self.gShapeZero,self.bShapeZero,
                                               self.GPU_bufDoppSum)

        self.GPU_filterMasks.prepared_call(self.gShapeVecMasks,self.bShapeVecMasks,
                             self.GPU_bufXcorr,self.GPU_bufSignalFreq,self.GPU_bufBitsMask,self.GPU_bufDoppIdx,
                             np.int32(self.Nfft),np.int32(self.doppIdxArrayLen))

        if self.FFTNoBatches == 1:
            cufft.cufftExecC2C(self.fftPlanDopplers,int(self.GPU_bufXcorr),int(self.GPU_bufXcorr),cufft.CUFFT_INVERSE)

        
        else:
            # It is at the moment faster to call single fft loops than batched fft loops (don't ask me why). This can however be changed in the config filte. Therefore all these loops
            for j in range(self.fftLoops):
                for i in range(self.num_streams):
                    if j*self.num_streams + i < self.FFTNoBatches:
                        buffAddr = self.buffAddr[j*self.num_streams+i] # compute the address in the large buffer
                        cufft.cufftExecC2C(self.fftPlanDopplers[i],buffAddr,buffAddr,cufft.CUFFT_INVERSE)
            
        self.GPU_absSumDoppler.prepared_call(self.gShapeAbsSum,self.bShapeAbsSum,
                               self.GPU_bufDoppSum,self.GPU_bufXcorr,np.int32(self.Nfft))

        if STORE_BITS_IN_FILE:
            tstore = time.time()
            tmpArr = np.empty((self.doppIdxArrayLen,self.num_masks),dtype=np.float32)
            cuda.memcpy_dtoh(tmpArr,self.GPU_bufDoppSum)
            self.sum_match = np.vstack([self.sum_match,np.sum(tmpArr,axis=1)])
            sumBest = np.argmax(np.sum(tmpArr,axis=1))
            log.debug(f'demodulator store in file 1 (save array) time {time.time()-tstore} s')

        self.GPU_estDoppler.prepared_call(self.gShapeDopp,self.bShapeDopp,
                                          self.GPU_bufDoppResult,self.GPU_bufFindDoppTmp,self.GPU_bufDoppSum,np.int32(self.num_dopplers),np.int32(self.doppIdxArrayOffset))

        bestDoppler = np.empty(2,dtype=np.float32) # contains [doppler, standard_dev]
        cuda.memcpy_dtoh(bestDoppler,self.GPU_bufDoppResult) # fetch the result
        # do scaling with the best Doppler before returning
        # Keep the scaling in indices. This is handier for the demodulation

        try:
            lowIdx = int(bestDoppler[0])
            highIdx = int(np.ceil(bestDoppler[0]))
            lowVal = self.doppHzLUT[lowIdx]
            highVal = self.doppHzLUT[highIdx]
            # This one is for our stats
            bestDopplerScaled = lowVal + (highVal-lowVal) * (bestDoppler[0] % 1) # scale with decimal offset

            # this one is for the demodulator. Contains the index
            self.dopplerIdxlast = np.int32(np.round(self.doppCyperSymNorm[lowIdx] + (self.doppCyperSymNorm[highIdx]-self.doppCyperSymNorm[lowIdx]) * (bestDoppler[0] % 1))) # directly for use in the demodulator
            # print(f'bestDopplerScaled {self.dopplerIdxlast}')
            SNR = self.computeSNR(lowIdx,highIdx,5)
            # SNR = 20.
            freqOffset = bestDopplerScaled - self.centreFreqOffset # we only want the detected frequency offset. Not the IF offset
            sdev_Hz = bestDoppler[1]/self.Nfft * self.sampleRate # not used when SUM_ALL_MASKS = 1

        except ValueError as e:
            log.error(f'Error occurred during find_UHF -- skipping block. Message: {e}')
            self.dopplerIdxlast = 0
            freqOffset = 0.
            sdev_Hz = 0.
            SNR = 0.
            
        return freqOffset, sdev_Hz, self.clippedPeakIPure, SNR


    def computeSNR(self,doppMatchLow,doppMatchHigh,windowWidth):
        """
        Computes the SNR based on the found doppler. Uses the frequency domain signal from the GPU
        The signal is detected as doppMatchLow-windowWidth : doppMatchHight + windowWidth
        The noise power is computed from the negative part of the spectrum (the signal resides in the positive part)

        Gives inaccurate results if the digitizer is not offtuned properly due to the presense of IF carriers
        """
        # print(f'SNR params: low {doppMatchLow} high {doppMatchHigh} width {windowWidth}')
        doppMatchLow_FFT_idx = self.doppCyperSymNorm[doppMatchLow]
        doppMatchHigh_FFT_idx = self.doppCyperSymNorm[doppMatchHigh]
        # print(f'SNR {doppMatchLow_FFT_idx} {doppMatchHigh_FFT_idx}')
        noiseIdxLow_FFT_idx = (doppMatchLow_FFT_idx + int(self.Nfft//2)) % self.Nfft
        noiseIdxHigh_FFT_idx = (doppMatchHigh_FFT_idx + int(self.Nfft//2)) % self.Nfft
            
        t = time.time()
        cuda.Context.synchronize()

        if doppMatchLow_FFT_idx > doppMatchHigh_FFT_idx: # the signal is around zero Hz IF
            sigPwr = np.mean(np.concatenate((np.abs(self.GPU_bufSignalFreq_cpu_handle[doppMatchLow_FFT_idx-windowWidth:]),np.abs(self.GPU_bufSignalFreq_cpu_handle[:doppMatchHigh_FFT_idx+windowWidth]))))
        else:
            sigPwr = np.mean(np.abs(self.GPU_bufSignalFreq_cpu_handle[doppMatchLow_FFT_idx-windowWidth:doppMatchHigh_FFT_idx+windowWidth]))

        if noiseIdxLow_FFT_idx > noiseIdxHigh_FFT_idx: # the signal is around zero Hz IF
            noisePwr = np.mean(np.concatenate((np.abs(self.GPU_bufSignalFreq_cpu_handle[noiseIdxLow_FFT_idx-windowWidth:]),np.abs(self.GPU_bufSignalFreq_cpu_handle[:noiseIdxHigh_FFT_idx+windowWidth]))))
        else:
            noisePwr = np.mean(np.abs(self.GPU_bufSignalFreq_cpu_handle[noiseIdxLow_FFT_idx-windowWidth:noiseIdxHigh_FFT_idx+windowWidth]))
            
        SNR = 20*np.log10(sigPwr/noisePwr - 1)
        # print(f'SNR {SNR:.1f} sigPwr {sigPwr:.6f} noisePwr {noisePwr:.6f}  dopp idx: {doppMatchLow_FFT_idx} {doppMatchHigh_FFT_idx} noise idx : {noiseIdxLow_FFT_idx} {noiseIdxHigh_FFT_idx}')

        # log.error(f'time SNR {(time.time()-t)*1000:.3f} ms')
        return SNR
            
        
    def __thresholdInput(self,samples):
        """
        This is necessery if there is local burst interference that is significantly stronger than the desired signal

        """
        absSamples = np.abs(samples) # 1 ms
        thresh = self.peakThresholdScale*np.mean(absSamples) # 0.2 ms
        i = np.where(absSamples>thresh)[0] # 1e-5 s
        samples[i] = thresh * (samples[i]/absSamples[i]) #  8e-5 s
        # Do it again in case the spikes were really loud
        absSamples[i] = np.abs(samples[i])
        thresh = self.peakThresholdScale*np.mean(absSamples)
        i = np.where(absSamples>thresh)[0]
        self.clippedPeakIPure = i # All peaks that are clipped at first round are clipped again. Requires that the peaks in first round are not set to 0
        samples[i] = thresh * (samples[i]/absSamples[i])
        # Mark peaks close to each other
        if len(self.clippedPeakIPure)>0:
            # t = time.time()
            # Mark peaks close to each other as continuous
            diffPeaks = np.diff(self.clippedPeakIPure)
            gapsAll = np.where(diffPeaks>1)[0]
            self.peakMinGap = 100
            gaps = np.where(diffPeaks[gapsAll] < self.peakMinGap)[0] # find gaps smaller than 100
            gapsLen = diffPeaks[gapsAll[gaps]] # length of the gaps
            gapsIdx = gapsAll[gaps]            # Index of all gaps


            # fill the gaps smaller than self.peakMinGap
            pp = np.zeros(self.Nfft,dtype=np.int8)
            pp[self.clippedPeakIPure] = 1
            for i in range(len(gapsLen)):
                pp[self.clippedPeakIPure[gapsIdx[i]]:self.clippedPeakIPure[gapsIdx[i]]+gapsLen[i]] = 1

            self.clippedPeakI = np.where(pp==1)[0]
        else:
            self.clippedPeakI = self.clippedPeakIPure.copy()
        if log.level == logging.DEBUG:
            log.debug('clipped peaks ' + str(len(self.clippedPeakIPure)))


          
    def findCodeRateAndPhaseGPU(self):
        """
        Find the code rate and phase on the GPU. It uses the first NUM_MASK rows in GPU_bufXcorr, which should contain the rough Doppler compensated carrier
        """

        # Abs**2 sum all Masks 
        self.GPU_sumXCorrBuffMasks.prepared_call(self.grid_codeXCorrSum,self.block_codeXCorrSum,
                                             self.GPU_bufCodeAndPhase,self.GPU_bufXcorr,np.int32(self.Nfft))

        # fft real to complex in-place output is Nfft/2*complex64 input is Nfft*float32
        cufft.cufftExecR2C(self.fftPlan_codeRate_R2C,int(self.GPU_bufCodeAndPhase),int(self.GPU_bufCodeAndPhaseOut))

        # find the code rate from the magnitude and phase from the argument

        self.GPU_findCodeRateAndPhase.prepared_call(self.grid_codeMax, self.block_codeMax,
                                                self.GPU_bufCodeAndPhaseResult,self.GPU_bufCodeAndPhaseOut,np.int32(self.codeRateAndPhaseOffsetHigh),np.int32(self.codeRateAndPhaseOffsetLow-self.codeRateAndPhaseOffsetHigh))

        if log.level == logging.DEBUG:
            log.debug(f'Code rate index: low {self.codeRateAndPhaseOffsetLow}\t high {self.codeRateAndPhaseOffsetHigh}')
        cuda.memcpy_dtoh(self.__CodeRateAndPhaseResult,self.GPU_bufCodeAndPhaseResult)


        try:
            # compute symbol rate
            spSym = self.Nfft/self.__CodeRateAndPhaseResult[0]

        except:
            log.error(self.__CodeRateAndPhaseResult)
            log.error('Code rate result 0 should not happen but happened -- fixing it to 10')
            spSym = 10
            
            
        try:
            # compute codeOffset
            codeOffset = -self.__CodeRateAndPhaseResult[1]/np.pi*spSym/2  #
            if codeOffset < 0: # wrap negative values
                codeOffset += spSym - 1
        except:
            log.warning('Error while computing code offset: codeOffset from GPU {}, index {}, max val {}'.format(self.__CodeRateAndPhaseResult[1],self.__CodeRateAndPhaseResult[0],self.__CodeRateAndPhaseResult[2]))
            codeOffset = 0

        return spSym, codeOffset


    def demodulateUHF(self):
        return self.__demodulate()

    def demodulateSTX(self):
        # self.dopplerIdxlast = self.Nfft//4  # quarter bandwidth off tuned
        self.dopplerIdxlast = self.doppOffsetIdx
        return self.__demodulate()
        
        

    def __demodulate(self,samples= None,doppEst=0,plotResults = False):
        """
        Demodulation of the radio signal when the doppler has been found
        """

        doppShift = self.dopplerIdxlast # found in the Doppler search
        # the signal still resides in the GPU buffer and still be fft'd
        # Fine tune the frequency offset

        ts = time.time()
      
        self.complexShiftMulMasks.prepared_call(self.gShapeVecMasks2,
                                                self.bShapeVecMasks2,
                                                self.GPU_bufXcorr,
                                                self.GPU_bufSignalFreq,
                                                self.GPU_bufBitsMask,
                                                np.int32(doppShift))


        # Ifft the data in one batch
        cufft.cufftExecC2C(self.fftPlanDemod,int(self.GPU_bufXcorr),int(self.GPU_bufXcorr),cufft.CUFFT_INVERSE)
        if log.level == logging.DEBUG:
            log.debug(f'Time demodulate FFT {time.time()-ts} s')
        

        tC = time.time()
        spSym,codeOffset = self.findCodeRateAndPhaseGPU()
        if log.level == logging.DEBUG:
            log.debug(f'Time demodulate findCodeRateAndPhase {time.time()-tC} s')

          
        ops = Operations.CENTRES_ABS # the operation to do on the matched filter.

        ## Decoding:
        if log.level == logging.DEBUG:
            log.debug('Time demodulate %f',time.time() - tC)
        tA = time.time()
        # t = time.time()
        idxSymbol, amplitudes, centres, symbols2, amplitudes2, trustSymbol = self.cudaFindCentres(spSym,codeOffset,ops)
        # print('time spent on centre search {}'.format(time.time()-t))
        # tb = time.time()
        dataBits, symError_t = self.extractBits(centres,idxSymbol)
        noError = len(symError_t)
        
        ## Remove the overlapping parts - up to 1 ms
        tSO = time.time()
        centresWin,dataBitsWin,trustSymbolWin, idxSymbolWin = self.checkSymbolOverlap(noError,centres,idxSymbol,dataBits,trustSymbol)
        if log.level <= logging.DEBUG:
            log.debug('Time demodulate overlap time {}'.format(time.time()-tSO))

        tP = time.time()

        ## tag the interference on the bits  - less than 100 us
        if len(self.clippedPeakIPure) > 0:
            cPSpan = self.clippedPeakSpan # This is the distance in symbols to where we assume a clipped peak can affect a bit
            clippedPeaks = self.clippedPeakIPure[self.clippedPeakIPure>centresWin[0]-cPSpan*spSym] # remove the first peaks that are out of the window
            if len(clippedPeaks) > 0:                
                clippedPeaks = clippedPeaks[clippedPeaks < centresWin[-1]+cPSpan*spSym] # cut of the end that is out of the window
        else:
            clippedPeaks = []

        if log.level == logging.DEBUG:
            log.debug('Time demodulate find peaks %f',time.time()-tP)
        t = time.time()

        # tag clipped peaks in the trust
        pp = np.zeros(self.Nfft,dtype = np.bool)
        spSymc = int(np.ceil(spSym))
        for cp in self.clippedPeakIPure:
            pp[cp-2*spSymc:cp+2*spSymc+1] = 1

        idxVal = pp[centresWin]
        trustSymbolWin[idxVal] = -2
        if log.level == logging.DEBUG:
            log.debug('Time demodulate peaks %f new ',time.time()-t)

        if STORE_BITS_IN_FILE is True:
            tstore = time.time()
            self.all_bits = np.append(self.all_bits,dataBitsWin.astype(DATATYPE))
            self.all_trust = np.append(self.all_trust,trustSymbolWin)
            self.frames = np.append(self.frames,len(self.all_bits))
            self.code_rate = np.append(self.code_rate, spSym)
            self.code_phase = np.append(self.code_phase, codeOffset)

            xcorrResS = np.empty((self.num_masks,self.Nfft),dtype=np.complex64)
            cuda.memcpy_dtoh(xcorrResS,self.GPU_bufXcorr)
            xcorrResS = np.expand_dims(xcorrResS,0)
            self.xcorrOut.append(xcorrResS)
            
            np.savez(BITS_FNAME,all_bits=self.all_bits,all_trust=self.all_trust, frames=self.frames,doppMatch = self.sum_match, code_rate = self.code_rate, code_phase = self.code_phase,masks = self.masks)
            if log.level == logging.DEBUG:
                log.debug(f'demodulator store in file 2 (save to file) time {time.time()-tstore} s')
        # Phase windup computations not done at the moment
        # return dataBitsWin, centres, idxSymbolWin,idxSymbolPreWin,idxSymbolPostWin, trustSymbolWin, amplitudes ,spSym
        return dataBitsWin.astype(np.uint8), centresWin.astype(np.uint8), trustSymbolWin.astype(np.uint8), spSym

    

    def checkSymbolOverlap(self,noError,centres,idxSymbol,dataBits,trustSymbol):
        """
        Checks whether the symbols overlap. This is only done if noError < SYMBOL_CHECK_ERROR_THRESHOLD
        It checks the symbols within SYMBOL_CHECK_OVERLAP_OFFSET at the beginning and end of the frame
        to align the bit sequences across frames.
        The alignment is done on the symbols rather than the decoded bits.

        Input:
           noError -- number of errors, decides whether the check is done
           centres -- symbol location in signal
           idxSymbol -- the symbol that is found
           dataBits -- the raw data bits
           trustSymbol -- the trust value (for soft decisions)

        Output: 
            centresWin -- aligned centres
            dataBitsWin -- aligned databits
            trustSymbolWin -- aligned trust
            idxSymbolWin -- aligned symbols
        """

  
        # It showed that when there is an overlap between the last window and the current in the first ~spSym samples.
        # Therefore, we only look at the samples after this overlap
        tO = time.time()
        startOverlap = np.where(centres >= (self.sigOverlapWin))[0][0] # we need to include the boundary once
        endOverlap = np.where(centres > (self.Nfft - self.sigOverlapWin))[0][0]
        
        idxSymbolWin = dataBits[startOverlap:endOverlap]
        idxSymbolPreWin = dataBits[:startOverlap]
        idxSymbolPostWin = dataBits[endOverlap:]

        if log.level == logging.DEBUG:
            try:
                log.debug(f'len bits {len(idxSymbol)}\tlen centres {len(centres)}')
                log.debug(f'\npreWin:  {idxSymbolPreWin.astype(np.int)}\nWinLast: {self.posSymEnd.astype(np.int)}\nCurWinStart: {idxSymbolWin[:len(self.poswinP)].astype(np.int)} \nposSymLast:  {self.poswinP.astype(np.int)}')
            except:
                pass
        
        tmpL = len(idxSymbolPreWin) + self.overlapOffset
        tmpP = len(idxSymbolPostWin) + self.overlapOffset

        if log.level == logging.DEBUG:
            log.debug('centresOffset[0] ' + str(centres[startOverlap] - self.sigOverlapWin) + '\tcentresOffset[-1] ' + str(self.Nfft-self.sigOverlapWin-centres[endOverlap]))
        try:
            if noError > self.symbol_check_error_threshold:
                if log.level == logging.DEBUG:
                    log.debug('Too many symbol errors')
            elif len(self.poswinP) > 0:
                if log.level == logging.DEBUG:
                    log.debug('Check symbol overlap')
                if np.all(self.poswinP[:self.overlapOffset] == idxSymbolWin[:self.overlapOffset]) or np.all(self.posSymEnd[-self.overlapOffset:] == idxSymbolPreWin[-self.overlapOffset:]):
                    if log.level == logging.DEBUG:
                        log.debug('overlap good')
                else:
                    # compute matches
                    symPre = np.sum(self.poswinP[:self.overlapOffset] == idxSymbolWin[:self.overlapOffset])
                    symPos =np.sum(self.posSymEnd[-self.overlapOffset:] == idxSymbolPreWin[-self.overlapOffset:])
                    symEarlyPre = np.sum(self.poswinP[:self.overlapOffset] == idxSymbolWin[1:self.overlapOffset+1])
                    symEarlyPos = np.sum(self.posSymEnd[-self.overlapOffset-1:-1] == idxSymbolPreWin[-self.overlapOffset:])
                    symLatePre = np.sum(self.poswinP[1:self.overlapOffset+1] == idxSymbolWin[0:self.overlapOffset])
                    symLatePos = np.sum(self.posSymEnd[-self.overlapOffset:] == idxSymbolPreWin[-self.overlapOffset-1:-1])
                    if log.level == logging.DEBUG:
                        log.debug('sum posw == symbWin %d\t sum posw == symbWinE %d\t sum posw == symbWinL %d'
                                  %(symPre,
                                    symEarlyPre,
                                    symLatePre))
                        log.debug('sum symO == symPre %d\t sum symO == symbPreE %d\t sum symO == symbPreL %d'
                                  %(symPos,
                                    symEarlyPos,
                                    symLatePos))
                        log.debug('poswin [-%d ; -1] %s' %(self.overlapOffset+1, str(self.posSymEnd[-self.overlapOffset-1:-1].astype(np.int))))
                        log.debug('prewin [-%d ; -1] %s' %(self.overlapOffset+1, str(idxSymbolPreWin[-self.overlapOffset-1:-1].astype(np.int))))

                        
                    maxPre = np.max((symPre,symEarlyPre,symLatePre))
                    maxPos = np.max((symPos,symEarlyPos,symLatePos))
                    
                    # check early
                    if self.symbol_check_match_threshold < symEarlyPre and symEarlyPre == maxPre:
                        if log.level == logging.DEBUG:
                            log.debug('posWin[:%d]==symbolWin[1:%d] passed' %(self.overlapOffset,self.overlapOffset+1))
                        if self.symbol_check_match_threshold < symEarlyPos and symEarlyPos == maxPos:
                            if log.level == logging.DEBUG:
                                log.debug('removed first bit')
                            startOverlap += 1
                            idxSymbolWin = idxSymbolWin[1:]
                    # Check late
                    elif self.symbol_check_match_threshold < symLatePre and symLatePre == maxPre:
                        if log.level == logging.DEBUG:
                            log.debug('posWin[1:%d]==symbolWin[:%d] passed' % (self.overlapOffset+1,self.overlapOffset))                     
                        if self.symbol_check_match_threshold < symLatePos and symLatePos == maxPos:
                            log.debug('inserted first bit')
                            startOverlap -= 1
                            idxSymbolWin = np.r_[idxSymbolPreWin[-1], idxSymbolWin]
                    if log.level == logging.DEBUG:
                        log.debug('new postWin last        ' + str(self.poswinP[:self.overlapOffset].astype(np.int)))
                        log.debug('new symbolsWin current  ' + str(idxSymbolWin[:self.overlapOffset].astype(np.int)))

            else:
                if log.level == logging.DEBUG:
                    log.debug('Skipping bit alignment, reason: no poswin saved')
        except Exception as e:
            log.error('symbol overlap failed. reason:')
            log.exception(e)


        dataBitsWin = dataBits[startOverlap:endOverlap]
        dataBitsPreWin = dataBits[:startOverlap]
        dataBitsPostWin = dataBits[endOverlap:]
        trustSymbolWin = trustSymbol[startOverlap:endOverlap]
        centresWin = centres[startOverlap:endOverlap]

        
        self.poswinP = dataBitsPostWin 
        try:
            self.posSymEnd = dataBitsWin[- self.overlapOffset-1:] # one bit extra
        except Exception:
            # just to guard the case that there are less than 10 symbols in the window
            log.error('Symbols for offset checking not saved -- Less than 10 symbols in the window')

        if log.level == logging.DEBUG:
            log.debug('time overlap %f',time.time()-tO)


        return centresWin,dataBitsWin,trustSymbolWin,idxSymbolWin
        
    
    def cudaFindCentres(self, spSym, codePhase, operation = Operations.CENTRES_ABS):
        # xcorrRes is not used and only added for compatibility
        # operation: 0 = abs, 1 = real, 2 = imag
        if spSym < self.spsymMin:
            spSym = self.spsymMin    # too low values makes us write outside of preallocated memory
        gShapeCentres = (int(np.ceil(self.Nfft/spSym/self.bShapeCentres[0])),1) # The GPU code takes care of array boundaries, so we deliberately make too many threads
        self.findcentres.prepared_call(gShapeCentres,self.bShapeCentres,self.GPU_symbols,self.GPU_centres,self.GPU_magnitude,self.GPU_bufXcorr,np.float32(spSym),np.float32(codePhase),np.int32(self.Nfft),np.int32(operation.value))

        symbols = np.empty(int(self.Nfft/spSym),dtype=np.int32)
        cuda.memcpy_dtoh(symbols,self.GPU_symbols)
        
        centres = np.empty_like(symbols)
        cuda.memcpy_dtoh(centres,self.GPU_centres)

        magnitudes = np.empty(int(self.Nfft/spSym),dtype=TRUSTTYPE)
        cuda.memcpy_dtoh(magnitudes,self.GPU_magnitude)
        trust =  magnitudes # TODO: trust based on amplitude or so

        return symbols,[],centres,0,0, trust


    def extractBits(self,centres,symbols):
        if self.bitLUT is None:
            if len(self.symbolLUT.shape) == 3:
                return self.extractBitsNRZs(centres,symbols)
        
            return self.extractBitsOld(centres,symbols)
        else:
            dataBits = self.bitLUT[symbols]

            # TODO: use the symbolLUT to provide trust values

            return dataBits, []


    def extractBitsNRZs(self, centresCoherent, symbols):
        ## Used for NRZs decoding currently used in BPSK

        # does the same as extractBitsOld, but allows multiple symbols representing each bits
        # The symbolLUT array is 3 dimensional:
        #   symbolLUT[symbol][binary0][nextSymbols]
        #   symbolLUT[symbol][binary1][nextSymbols]
        symbols2 = symbols[np.newaxis]
        # log.info(f'symbolLUT shape {self.symbolLUT.shape}')
        # log.info(f'symbols shape {symbols2.shape}')
        # log.info(f'symbolLUT shape {self.symbolLUT[symbols[:-1],0,:].shape}')
        # log.info(f'symbols shape {symbols2[:,1:].T.shape}')
        # tmp = symbols2[:,1:].T ==  self.symbolLUT[symbols[:-1],0,:]
        # log.info(f'tmp  {tmp}')
        # log.info(f'tmp shape {tmp.shape}')
        res1 = np.any(symbols2[:,1:].T == self.symbolLUT[symbols[:-1],0,:],axis=1) # all the ones
        res0 = np.any(symbols2[:,1:].T == self.symbolLUT[symbols[:-1],1,:],axis=1) # all the zeros

        #log.debug(f'res1 shape {res1.shape} res0 shape {res0.shape}')
        res = res1+res0 # everything that is zero here is an error

        symError = np.where(res==0)[0].tolist()

        res1[symError] = np.int(SYMBOL_MISMATCHVAL)

        return res1,symError

        
     
    def get_signalBufferHostPointer(self):
        """ 
        Clean way to get the host handle to the pinned memory of the signal buffer
        Returns the handle
        """
        return self.GPU_bufSignalTime_cpu_handle

           
        
