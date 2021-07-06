// Copyright: (c) 2021, Edwin G. W. Peters

#include <math.h>
#include <cufft.h>
#include <cufftXt.h>

/*
  Helper macros for Cuda indexing
*/
#define GRID_X (threadIdx.x + blockDim.x*blockIdx.x)
#define GRID_Y (threadIdx.y + blockDim.y*blockIdx.y)
#define FLAT_X (GRID_Y*gridDim.x*blockDim.x + GRID_X)
#define FULL_MASK 0xffffffff // For shfl instructions


/*
  There are two methods to find the Doppler. This is configured using #SUM_ALL_MASKS
  if #SUM_ALL_MASKS == false
  The maximum magnitude is found for each mask among each Doppler. 
  The Doppler is then found by a weighted average between the two masks and/or Dopplers with the largest magnitude 
  This method is robust if some masks weakly correlate with other disturbances in the signal

  if #SUM_ALL_MASKS == true
  The magnitude of all masks are summed together for each Doppler.
  The Doppler is then found by a weighted average between the two Dopplers with the largest magnitude
  This method is robust when aliasing happens for some masks, for example when using FSK modulation schemes

*/

/*
  The below is dynamically coded from Python 
  NUM_MASKS : the number of masks to correlate

  # Demodulator:
  WINDOW_WIDTH : the window around the centre bit to search for when locating the symbols
  CODE_SEARCH_MASK_OFFSET : the first and last N masks from the code search. Can for some modulation schemes improve the accuracy of the symbol synchronisation
*/


typedef float2 Complex;

static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b);
static __device__ __host__ inline Complex ComplexConj(Complex a);
static __device__ __host__ inline Complex ScaleComplex(Complex a, const float b);
static __device__ __host__ inline Complex ComplexZero(void);
static __device__ __host__ inline Complex ComplexOne(void);
static __device__ __host__ inline Complex ComplexNumber(float a, float b);
static __device__ __host__ inline Complex ComplexCarrier(float in);
static __device__ __host__ inline float ComplexAbs(Complex in);
static __device__ __host__ inline float ComplexAbs2(Complex in);
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b);
static __device__ __host__ inline void atomicComplexAdd(Complex *a, Complex b);
static __device__ __host__ inline float ComplexReal(Complex a);
static __device__ __host__ inline float ComplexImag(Complex a);
static __device__ __host__ inline float ComplexAbsSquared(Complex in);
static __device__ __host__ inline float ComplexRealAbs(Complex in);
static __device__ __host__ inline float ComplexImagAbs(Complex in);

// Helper functions for sum
__inline__ __device__ float warpReduceSum(float val);
__inline__ __device__ float blockReduceSum(float val);


__device__ float sum;

// Callable methods


__device__
cufftCallbackLoadR d_loadCallbackPtr;


enum signalOp{Abs,Real,Imag};
#define CENTRES_ABS 0
#define CENTRES_REAL 1
#define CENTRES_IMAG 2

__global__ void findCentres(int *outSym, int *outIdx, float *magnitude, Complex *sig, const float spSym, const float offset, const int lenSig, const enum signalOp signalFunc)
{
  /* the output vectors outSym, outIdx have to be long enough to contain lenSig/spSym symbols with the lowest spSym possible
     signalFunc:
     0:  abs 
     1:  real 
     2:  imag
  */
  int x = GRID_X; // this is the symbol index in out

  int arrayIdx = (int) (((float) x * spSym) - WINDOW_LEFT_OFFSET + offset);
  int maxArrayIdx = arrayIdx + WINDOW_WIDTH; // defined from python

  int offsetComp = (int) offset;
    
  // input boundaries
  if (arrayIdx < 0)
  {
    offsetComp -= arrayIdx; 	// to keep track of the extra compensation in the final index 
    arrayIdx = 0;
  }
  if (maxArrayIdx > lenSig)
    maxArrayIdx = lenSig;

  maxArrayIdx -= arrayIdx;
  int maxIdx = -1;
  int maxCentreIdx = -1;
  if (arrayIdx < lenSig) 	// only array indexes wihtin the signal are used
  { 
    float (*mathFunc)(Complex); // this allows multiple scalings to be used, ie. real, imag or abs
    switch(signalFunc)
    {
    case Abs:
      mathFunc = &ComplexAbsSquared;
      break;
    case Real:
      mathFunc = &ComplexRealAbs;
      break;
    case Imag:
      mathFunc = &ComplexImagAbs;
      break;
    default:
      mathFunc = &ComplexAbsSquared;
    }
    float maxVal = 0;
    float tmp;
    int cnt = 0;
    for (int i = arrayIdx; i < lenSig*NUM_MASKS; i += lenSig) // loop over rows
    {
      for (int k = 0; k < maxArrayIdx; k++) // loop over symbols in window
      {	    
	tmp = mathFunc(sig[i+k]);
	if (tmp > maxVal)
	{
	  maxVal = tmp;
	  maxIdx = cnt;
	  maxCentreIdx = k;
	}
	  
      }
      cnt++;
    }
    // sig[x] = ComplexNumber(x,0);
    outSym[x] = maxIdx;
    outIdx[x] = (int)(x*spSym-WINDOW_LEFT_OFFSET+maxCentreIdx+offsetComp);
    magnitude[x] = maxVal;

  }
}







__global__ void multInputVectorWithMasks(Complex *out, Complex *vec, Complex *masks)
{
  // Elementwise multiplication of vector with rows of a matrix
  const int x = GRID_X;
  // const int y = GRID_Y;
  const int flatIdx = FLAT_X;
  out[flatIdx] = ComplexMul(vec[x],masks[flatIdx]);
  // out[flatIdx] = ComplexNumber(flatIdx,x);
  
}

__global__ void complexShiftMul(Complex *out, Complex *sig, Complex *mask,
				const int shift, const int len)
{
  // multiply a shifted array
  const int gIdx = GRID_X;
  out[gIdx] = ComplexMul(sig[(gIdx+shift) % (gridDim.x*blockDim.x)],mask[gIdx]);   
}


__global__ void multInputVectorWithShiftedMask(Complex *out, Complex *vec, Complex *masks, const int shift)
{
  int x = GRID_X;
  Complex sig = vec[(x+shift) % (gridDim.x*blockDim.x)];
  
  
  for (int i = x; i < NUM_MASKS*gridDim.x*blockDim.x; i+= (gridDim.x*blockDim.x))
  {
    out[i] = ComplexMul(sig,masks[i]);    
  }
    
}


/*
  Buffer sum for code phase and rate estimation
*/
__global__ void sumXCorrBuffMasks(float *out, Complex *in, const int inLen)
{
  // sums all the buffers for all masks squared element-wise together in an output buffer
  // used to find the code rate and phase
  for (int x = GRID_X; x<inLen; x+= blockDim.x*gridDim.x) // iterate over blocks
  {
    float tmpSum = 0;
    for (int i=CODE_SEARCH_MASK_OFFSET*inLen; i<(NUM_MASKS-CODE_SEARCH_MASK_OFFSET)*inLen; i+=inLen)
    {
      tmpSum += ComplexAbsSquared(in[x+i]);
      //tmpSum += tmp*tmp;
    }
    out[x] = tmpSum;
  }
}


__inline__ __device__ float2 warpArgMaxValsOffset(int maxIdx, float maxVal,int mask=FULL_MASK)
{
  /*
   * Finds the maximum value and index within a warp
   */
  float tmp;
  int tmpIdx;
  for (int i = WARP_SIZE >> 1; i > 0; i = i >> 1)
  {
    __syncwarp(mask);
    tmp = __shfl_xor_sync(mask,maxVal,i);
    tmpIdx =  __shfl_xor_sync(mask,maxIdx,i);

    if (maxVal < tmp) // I believe this has to be smaller than
    {
      maxVal = tmp;
      maxIdx = tmpIdx;
    }
    
  }
  return make_float2((float)maxIdx,maxVal);
  
}



__global__ void findCodeRateAndPhase(float *out, Complex *in, const int offset, const int len)
{
  /*
   * This function finds the maximum of the magnitude of the input from offset and len samples ahead/
   * Input:
   *    out     - len 3 float array (see Output)
   *    in      - Complex64 array with fft values
   *    offset  - the offset index where to start
   *    len     - the number of indices to look ahead from offset
   * Output:
   *    Length 3 array with:
   *        [index of max (compensated for offset),
   *	      argument at max index (normalized),
   *	      value at max (abs()**2)]
   */
    
  int arrayIdx;
  float maxValCand;
  float2 maxIdxVal = make_float2(0,0); // [index, value]
  __shared__ float2 sMaxIdxVal[NUM_WARPS]; // one pr. warp

  for (int x=GRID_X; x<len; x+=gridDim.x*blockDim.x) // current warp loops until end
  {
    arrayIdx = x + offset;
    maxValCand = ComplexAbsSquared(in[arrayIdx]);
    // maxValCand = maxValCand * maxValCand; // square
	
    if (maxIdxVal.y > maxValCand)
    { // is new value smaller than the already stored?
      maxValCand = maxIdxVal.y; // value
      arrayIdx = (int) (maxIdxVal.x);// index
    }
    __syncwarp();
    int activeMask = __activemask(); // find the active threads
       
    maxIdxVal = warpArgMaxValsOffset(arrayIdx,maxValCand,activeMask); // Each thread in the warp now has the max value and index	
    // __syncwarp();
  }

  // We now have the max values and indices in all threads in each warp.
  // Next step is to let warp 0 reduce this. The first thread in each warp writes the result to shared memory

  int warp_id = threadIdx.x >> LOG_WARP_SIZE; // everyone is going to need this
  int lane_id = threadIdx.x & (WARP_SIZE - 1);
  if (lane_id == 0) // only the first lane in each warp does this
  {
    sMaxIdxVal[warp_id] = maxIdxVal;
  }
    
    
  __syncthreads(); // all threads need to reach this stage before the final computation can be done
      

  // only warp 0 does the last computation from the shared memory
  if (warp_id == 0)
  {
    for ( ; lane_id < NUM_WARPS; lane_id += WARP_SIZE) // FUTURE: in case more than 32 warps would be active (require more than 1024 threads/block)
    {
      if (lane_id < NUM_WARPS)
      {
	arrayIdx = (int) sMaxIdxVal[lane_id].x;
	maxValCand = sMaxIdxVal[lane_id].y;
	int activeLanes = __activemask(); // only lanes inside this loop should do the next steps
	// __syncwarp(activeLanes);
	// __syncwarp();
	maxIdxVal = warpArgMaxValsOffset(arrayIdx,maxValCand,activeLanes);

	// __syncwarp(activeLanes);
	if (lane_id % WARP_SIZE == 0)
	{
	  out[0] = maxIdxVal.x; // found the max index
	  Complex inVal = in[(int)maxIdxVal.x];
	    
	  out[1] = (float)atan2f(inVal.y,inVal.x); // argument
	  out[2] = maxIdxVal.y; // magnitude (for debugging)
	}
	//debugging
	// in[lane_id] = ComplexNumber(sMaxIdxVal[lane_id].x,sMaxIdxVal[lane_id].y); // shared mem buffer
	// in[lane_id+NUM_WARPS] = ComplexNumber(maxIdxVal.x,maxIdxVal.y); // lane max value and index
      }
    }

  }    
    
}

/*
 * multiplication in the cross correlation
 */


__inline__ __device__  void multWithMasks(Complex *out, Complex inp,Complex *masks, int *flatIdx)
{
  /*
   * This inline function multiplies one input with a list of masks residing in registers. The output is immediately written to global memory. 
   */
  for (int i = 0; i < NUM_MASKS; i++)
  {
    out[flatIdx[i]] = ComplexMul(inp,masks[i]);
  }
  
}

__global__ void multInputVectorWithShiftedMasksDopp(Complex *out, Complex*  vec, Complex* masks,
						    int* dopplerShifts,
						    const int vecLen, const int dopplerLen)
{
  /* 
   * The input is a length vecLen Complex vector (vec), a vecLen x NUM_MASKS 2D vector (masks) and a length dopplerLen vector with Doppler shifts.
   * The output is a vecLen x NUM_MASKS x dopplerLen 3D vector 
   * Elementwise multiplication of a shifted vector with rows of a matrix
   */
  const int x = GRID_X;
  int shift;
  int flatOutIdx;
  Complex localMasks[NUM_MASKS];
  int flatIdx[NUM_MASKS];
  

  // load the masks to registers. These will be used a lot
  for (int i = 0; i < NUM_MASKS; i++)
  {
    flatIdx[i] = i*(gridDim.x*blockDim.x) + x;
    localMasks[i] = masks[flatIdx[i]];	
  }
  
  for (int j = 0; j < dopplerLen; j++)
  {
    // Loop through dopplers
    shift = dopplerShifts[j]; // Dictates how many samples to shift in the input vector
    flatOutIdx = j*(NUM_MASKS*gridDim.x*blockDim.x); // flattened output index

    // Convolve all masks with this input and doppler shift
    // A tiny bit slower than doing it in the function here, but we can afford it (less than 100 us on 40 loops)
    multWithMasks(&out[flatOutIdx],vec[(x+shift) % vecLen], localMasks,flatIdx);
  }
  
}


/*
  Not sorted yet
*/



__global__ void warpAbsSumAtomic(float *out, Complex* __restrict__ in, const int num_elements)
{
  // Ensure that num_elements is a multiple of WARP_SIZE
  // This method utilizes direct register access which is faster than shared memory
  // but relies on multiple atomic operations, which may result in lower performance
  // when too many atomic operations block the threads
  // lower accuracy than the block sum
  float sum[NUM_MASKS] = {0};
  int y = GRID_Y; 		// The index of the doppler
  
  int rowLen = num_elements;
  int flatxOut = rowLen*GRID_Y; // The flat index of the first element in the rows of dopplers
  int flatx = flatxOut * NUM_MASKS;
  
  // Each thread in the warp sums entries from each block in the grid together in local memory
  // This is done for each row in the matrix
  for (int i = blockIdx.x* blockDim.x + threadIdx.x;
       i < num_elements;
       i += blockDim.x * gridDim.x)
  {
    for (int j = 0; j < NUM_MASKS; j++)
    {
      sum[j] = ComplexAbsSquared(in[i+rowLen*j+flatx]);
      //sum[j] += tmp*tmp;
    } 
  }
  // The entire warp sums the local sums together
  for (int j = 0; j < NUM_MASKS; j++)
    sum[j] = warpReduceSum(sum[j]);


  // Let the first NUM_MASKS threads add the local sums to the output array
  int lIdx = threadIdx.x % WARP_SIZE; // local index
  if (lIdx < NUM_MASKS)
    atomicAdd(&out[lIdx+NUM_MASKS*y],sum[lIdx]);
  
}


__global__ void blockAbsSumAtomic(float *out, Complex* in, const int num_elements)
{
  // Ensure that num_elements is a multiple of WARP_SIZE
  // This method relies on shared memory which saves atomic operations
  // Shared memory is slower than direct register access, but prevents threads blocking
  // due to atomic operations
  // Higher accuracy than the warp sum
  float sum[NUM_MASKS] = {0};
  int y = GRID_Y;
  int rowLen = num_elements;
  int flatxOut = rowLen*GRID_Y; // The flat index of the first element in the rows of dopplers
  int flatx = flatxOut * NUM_MASKS;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < num_elements;
       i += blockDim.x * gridDim.x)
  {
    for (int j = 0; j < NUM_MASKS; j++)
    {
      // float tmp = ComplexAbs2(in[j*rowLen + i + flatx])/262144; // pow(2,18); Pow computed offline saves 1.5 ms
      // sum[j] += tmp*tmp*tmp*tmp;  // Cube for more power on match
      float tmp = ComplexAbsSquared(in[j*rowLen + i + flatx])/262144; // pow(2,18); Faster than the above by additional 1 ms 10 ksps to 14 ksps in total
      sum[j] += tmp;  // Cube for more power on match
    }
  }

  for (int j = 0; j < NUM_MASKS; j++)
    sum[j] = blockReduceSum(sum[j]);

 
  int lIdx = threadIdx.x % WARP_SIZE; // local index

#if SUM_ALL_MASKS == true
  /*
    The magnitudes of all samples and masks for each Doppler are summed and stored for each Doppler
  */
  if (lIdx == 0)
  {
    for (int i = 1; i < NUM_MASKS; i++)
    {
      sum[0] += sum[i];
    }
    atomicAdd(&out[y*NUM_MASKS+lIdx],sum[0]);
  }

#else
  /*
    The normal method:
    Magnitudes of all samples for each Doppler and each mask are summed and stored individually for each mask and Doppler
  */
  
  if (lIdx < NUM_MASKS)
  {
    atomicAdd(&out[y*NUM_MASKS+lIdx],sum[lIdx]);
  }

#endif
  
  
}



/*
 * Next kernel and methods are used to find the optimal Doppler and standard deviation
 */

__inline__ __device__
float activeWarpReduceSum(float val)
{
  // Warp reduction sum in warp
  __syncwarp();
  int mask = __activemask();
  
  for (int i = WARP_SIZE/2; i > 0; i/=2)
    val += __shfl_xor_sync(mask,val,i);
  return val;
}


__global__ void findDopplerEst(float *res, float* __restrict__ tmpIdx, float* __restrict__ in, const int num_elements,const int element_offset)
{
  /*
   * This kernel finds the doppler estimation and standard deviation of the estimate.
   * The input in contains a matrix with each column being a mask and each row a doppler estimate
   * The kernel is designed to run in one warp, thus the max limit is 32 masks in current GPU architectures
   * 
   * Arguments:
   *    *res  -- length two vector for output: [doppler index, standard deviation]
   *    *tmpIdx -- not used
   *    *in   -- input matrix of dimensions num_elements X NUM_MASKS
   *    num_elements -- number of elements (number of Dopplers)
   *    element_offset -- number of elements (Dopplers) to skip in search. This is to exclude the noise measurement etc.
   *
   * Grid size: (1,1)
   * Block size: (NUM_MASKS,1,1)
   * 
   * base description of algorithm
   * Finds the 2 highest values in a matrix
   * Each thread finds the two largest values and their indices for a mask amongst all dopplers (num_elements)
   * Next, each thread computes the weighted average index of those two values
   * Then the local average indices get summed accross all threads in the warp 
   * The standard deviation is then computed in the warp
   */

  int x = GRID_X;
  float maxVal[2] = {0};
  int maxIdx[2] = {0};
  int curSecondMaxValIdx = 0;
  float tmp;
  
  // Locally find the 2 highest Doppler estimates for each mask. Each thread takes care of 1 mask
  for (int i = element_offset; i < num_elements+element_offset; i++)
  {
    tmp = in[x+i*NUM_MASKS];
    if (tmp > maxVal[curSecondMaxValIdx])
    {
      maxVal[curSecondMaxValIdx] = tmp;
      maxIdx[curSecondMaxValIdx] = i;
      curSecondMaxValIdx = (maxVal[0] >= maxVal[1]) ? 1 : 0; // update index
           
    }
  }
  // At this stage each thread has the two max values and indexes of the Doppler. Find the weighted average index
  tmp = maxIdx[0]*maxVal[0]+maxIdx[1]*maxVal[1];
  float maxDoppIdxL = (tmp)/(maxVal[0]+maxVal[1]);
  float maxDoppValL = (tmp)/(maxIdx[0]+maxIdx[1]);

  if (element_offset > 0)
  {
    // if element_offset == 1, then the first array contains the correlation of the filters with the noise
    maxDoppValL = maxVal[(curSecondMaxValIdx + 1) %2]/in[x]; 
  }
  
#if SUM_ALL_MASKS == true
  /*
    All masks are summed, so each thread has the best Doppler -- thread 0 stores
  */
  if ( x == 0)
  {
    res[0] = maxDoppIdxL;
    // res[1] = -1;
    
    res[1] = 10*log10(maxDoppValL);
    //res[1] = abs(maxIdx[0]-maxIdx[1]);  // return the difference of the two value used
  }

#else
  /*
    Each thread contains the optimal Doppler idx and value for one mask.
    Find the best Doppler among all masks
  */

  // Now each thread has the local optimal idx for the Doppler for each their own mask. Find the average optimal index accross all masks
  float maxDoppIdx = activeWarpReduceSum(maxDoppIdxL)/NUM_MASKS;

  __syncwarp();
  float maxDoppVal = activeWarpReduceSum(maxDoppValL)/NUM_MASKS;

  // Compute the standard deviation of the estimates among the local optimals from all threads (last part is done below)
  // maxDoppIdxL = maxDoppIdxL - maxDoppIdx;
  // float maxDoppStd = activeWarpReduceSum(maxDoppIdxL*maxDoppIdxL);
  

  // Thread 0 writes the reduced result out to the output array
  if ( x == 0 )
  {
    res[0] = maxDoppIdx;
    // res[1] = sqrtf(maxDoppStd/NUM_MASKS);
    res[1] = 10*log10(maxDoppVal);
  }
#endif
  
  
}


/*
 * a function to find the two max vals in a warp (unused, and probably un-tested)
 */



__inline__ __device__ int warpArgMaxVals(float maxVal,int mask=FULL_MASK)
{
  float tmp;
  int maxIdx = GRID_X;
  
  for (int i = WARP_SIZE/2; i > 0; i/=2)
  {
    tmp = __shfl_xor_sync(mask,maxVal,i);
    if (maxVal < tmp) // I believe this has to be smaller than
    {
      maxVal = tmp;
      maxIdx = __shfl_xor_sync(mask,maxIdx,i);
    }
    
  }
  return maxIdx;
  
}


__global__ void argMax2Vals(float *in, int *maxIdxOut, float *maxValOut, const int num_elements)
{
  /*
    This function returns  the highest values in an array
  */
  float maxTmp[2] = {0};
  int maxIdxTmp[2];
  int minIdx = 0;
  // float tmp;
  static int2 __shared__ maxIdx[WARP_SIZE];
  static float2 __shared__ maxVals[WARP_SIZE];
  float tmpVal;
  

  // Loop through memory to get max values
  for (int i = blockIdx.x* blockDim.x + threadIdx.x;
       i < num_elements;
       i += blockDim.x * gridDim.x)
  {
    minIdx = maxTmp[0] < maxTmp[1] ? 0 : 1;

    tmpVal = in[i];
    if (tmpVal > maxTmp[minIdx])
    {
      maxTmp[minIdx] = tmpVal;
      maxIdxTmp[minIdx] = i;
      
    }
  }

  // Find the max values in the warp
  const int warpIdx = threadIdx.x / WARP_SIZE;
  int maxIdxWarp = warpArgMaxVals(maxTmp[0]);
  if (maxIdxWarp == threadIdx.x)
  {
    maxVals[warpIdx].x = maxTmp[0];
    maxIdx[warpIdx].x = maxIdxTmp[0];
    maxTmp[0] = 0;
  }
  maxIdxWarp = warpArgMaxVals(maxTmp[0]);
  if (maxIdxWarp == threadIdx.x)
  {
    maxVals[warpIdx].y = maxTmp[0];
    maxIdx[warpIdx].y = maxIdxTmp[0];
  }

  maxIdxWarp = warpArgMaxVals(maxTmp[1]);
  if (maxIdxWarp == threadIdx.x)
  {
    if (maxTmp[1] > maxVals[warpIdx].y)
    {
      maxVals[warpIdx].y = maxTmp[1];
      maxIdx[warpIdx].y = maxIdxTmp[1];
    }
    maxTmp[1] = 0;
  }
  if (maxVals[warpIdx].x < maxVals[warpIdx].y)
  {
//  If x is smaller than y, we have to consider the last value
    maxIdxWarp = warpArgMaxVals(maxTmp[1]);
    if (maxIdxWarp == threadIdx.x)
    {
      if (maxTmp[1]  > maxVals[warpIdx].x)
      {
	maxVals[warpIdx].x = maxVals[warpIdx].y;
	maxIdx[warpIdx].x = maxIdx[warpIdx].y;
	maxVals[warpIdx].y = maxTmp[1];
	maxIdx[warpIdx].y = maxIdxTmp[1];
      }
    }
	
	
  }

  // Now the first warp finds the max 2 values in shared memory
  // __syncThreads();
  
  if (threadIdx.x < WARP_SIZE)
  {
    
    float2 maxValL = maxVals[threadIdx.x];
    int2 maxIdxL = maxIdx[threadIdx.x];

    maxIdxWarp = warpArgMaxVals(maxValL.x);
    if (threadIdx.x == maxIdxWarp)
    {
      maxIdxOut[0] = maxIdxL.x;
      maxValOut[0] = maxValL.x;      
    }

    maxIdxWarp = warpArgMaxVals(maxValL.y);
    if (threadIdx.x == maxIdxWarp)
    {
      maxIdxOut[1] = maxIdxL.y;
      maxValOut[1] = maxValL.y;      
    }

    

 
     
  }
  
    
  
}





// 1D stuff
__global__ void complexConjMul(Complex *a, Complex *b)
{
  // Performs a = a * conj(b)  float maxDoppIdx = activeWarpReduceSum(maxDoppIdxL)/NUM_MASKS;

  const int i = GRID_X;
  a[i] = ComplexMul(a[i],ComplexConj(b[i]));
}

__global__ void complexMul(Complex *out, Complex *a, Complex *b)
{
  // Performs a = a * b
  const int i = GRID_X;
  out[i] = ComplexMul(a[i],b[i]);
}



__global__ void complexHeterodyne(Complex *out, Complex *in, const float a,const float b, const float c,const int sigLen)
{
  // heterodynes signal in with theta = a*x^2+b*x+c
  // Heterodynes signal in with exp(j*theta)
  // iterates over the number of masks 
  int x = GRID_X;
  Complex val;
  // float theta = fmod(a*powf(x,2) + b*x + c,2*(float) M_PI);
  //int y = x;
    
  float theta = ((a*x)+b)*x;
  // float theta = fmod(a*y,(float)100000);
  // theta += fmod((theta+b)*x,(float)100000) + c;
  theta = fmod(theta,2*(float) M_PI)+c;
  val.x = cosf(theta);
  val.y = sinf(theta);
    
  for ( ;x < sigLen * NUM_MASKS; x += sigLen)
  {
    out[x] = ComplexMul(in[x],val);
    // out[x] = ComplexNumber(GRID_X,0);
    // out[x] = val;
  }
}



__global__ void complexConj(Complex *a)
{
  // Performs a = conj(a)
  // const int x = GRID_X;
  const int flatIdx = GRID_X + GRID_Y*blockDim.x*gridDim.x;
  
  a[flatIdx] = ComplexConj(a[flatIdx]);
}

__global__ void scaleComplexByScalar(Complex *a, const float b)
{
  // Performs a = b * a where b is a constant scalar
  const int i = GRID_X;
  a[i] = ScaleComplex(a[i],b);
}

__global__ void upsampleMask(Complex *out, Complex *mask, const int spsym)
{
  // Pads zeros between the short form binary mask
  const int i = GRID_X;
  int idx = i/spsym;
  if( i % spsym == 0 )      
    out[i] = ComplexNumber((float)i,(float)idx);
  else
    out[i] = ComplexNumber((float)i,-1);
}


__global__ void mixAndPadSignalWithCarrier(Complex *out, Complex *in, const float phi, const int length)
{
  int i = GRID_X;
  if (i < length)
  {
    Complex carrier = ComplexCarrier(phi*((float) i));
    out[i] = ComplexMul(in[i],ComplexConj(carrier));
  }
  else
    out[i] = ComplexZero();
  
}

__device__ cufftComplex CB_mixAndPadSignalWithCarrier(Complex *in, const float phi, const int length)
{
  int i = GRID_X;
  if (i < length)
  {
    Complex carrier = ComplexCarrier(phi*((float) i));
    return (cufftComplex) ComplexMul(in[i],ComplexConj(carrier));
  }
  else
    return (cufftComplex) ComplexZero();
  
}

__global__ void mixSignalWithCarrier(Complex *out, Complex *in, const float phi, const int length)
{
  // Computes out = in * exp(j*phi)
  int i = GRID_X;
  Complex carrier = ComplexCarrier(phi*((float) i));

  if (i < length)
    out[i] = ComplexMul(in[i],ComplexConj(carrier));
  
}

__global__ void setComplexArrayToZeros(Complex *a)
{
  const int i = GRID_X;
  a[i] = ComplexZero();
}

__global__ void setArrayToZeros(float *a)
{
  const int i = GRID_X;
  a[i] = 0;
}


__global__ void complexAbs(Complex *a)
{
  int x = GRID_X;

  a[x] = ComplexNumber(ComplexAbs(a[x]),0);
  
}


__global__ void sumComplexVectorN(Complex *in, float *out, int num_elements, int store_loc)
{
  int globalIdx = GRID_X;
  float temp = 0;
  __shared__ float buf[WARP_SIZE];
  const int lane = threadIdx.x % WARP_SIZE;


  // each warp has 32 elements.
  while(globalIdx < num_elements)
  {
    temp = ComplexAbs(in[globalIdx]);
    // temp = 1;
    // All threads compute the sum of all threads in their warp. sing butterfly method
    // see https://www.google.com.au/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0ahUKEwjk8PiOz9vYAhUFipQKHRTiAKEQFgg2MAE&url=https%3A%2F%2Fpeople.maths.ox.ac.uk%2Fgilesm%2Fcuda%2Flecs%2Flec4.pdf&usg=AOvVaw2zBTrE9zyIN1DXWQjqB8dt
    // This requires log2(WARP_SIZE/2) loops
    // #pragma unroll
    for (int i=WARP_SIZE/2; i>0 ; i/=2 )
      temp += __shfl_xor_sync(FULL_MASK,temp,i,WARP_SIZE);
    // temp += __shfl_xor(temp,i);

    // write the result to shared memory
    if (lane == 0)
      buf[threadIdx.x / WARP_SIZE] = temp;
    // buf[threadIdx.x / WARP_SIZE] = 1;

    __syncthreads();

    // add the 32 partial sums in shared mem using one single warp
    if (threadIdx.x < WARP_SIZE)
    {
      temp = buf[threadIdx.x];

      // #pragma unroll
      for (int i=WARP_SIZE/2; i>0 ; i/=2 )
	temp += __shfl_xor_sync(FULL_MASK,temp,i);

      // only thread 0 has to do this
      
    }
    if (threadIdx.x == 0)
    {
      atomicAdd(&out[store_loc],temp); // we do not want others to disturb us while doing this
    }
    // Jump ahead to the next block of data

    globalIdx += blockDim.x * gridDim.x;
    __syncthreads();
  }


}

// Static scalar methods to be called from within gpu kernels

static __device__ __host__ inline Complex ComplexNumber(float a, float b)
{
  // initialize a complex number
  Complex c;
  c.x = a;
  c.y = b;
  return c;
}

static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
  // Internal complex multiplication
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b)
{
  // Internal complex addition
  Complex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}


static __device__ __host__ inline float ComplexReal(Complex a)
{
  return a.x;
}

static __device__ __host__ inline float ComplexImag(Complex a)
{
  return a.y;
}

static __device__ __host__ inline Complex ScaleComplex(Complex a, const float b)
{
  // Internal complex scale
  Complex c;
  c.x = a.x * b;
  c.y = a.y * b;
  return c;
}

static __device__ __host__ inline Complex ComplexConj(Complex a)
{
  // Internal complex conjugate
  a.y = -a.y;
  return a;
}

static __device__ __host__ inline Complex ComplexZero(void)
{
  // initializes a complex type with zeros
  Complex c;
  c.x = 0;
  c.y = 0;
  return c;
}

static __device__ __host__ inline Complex ComplexOne(void)
{
  // initializes a complex type with ones
  Complex c;
  c.x = 1;
  c.y = 0;
  return c;
}

static __device__ __host__ inline Complex ComplexCarrier(float in)
{
  // Generates exp(j * in)
  Complex c;
  c.x = cosf(in);
  c.y = sinf(in);
  return c;
}

static __device__ __host__ inline float ComplexAbs(Complex in)
{
  return sqrtf(in.x*in.x + in.y*in.y);
}

static __device__ __host__ inline float ComplexAbs3(Complex in)
{
// Improved numeric stability compared to ComplexAbs
  const float s = max(abs(in.x),abs(in.y));
  if (s == 0) 
    return s;
  float x = in.x/s;
  float y = in.y/s;
  return s * sqrt(x*x + y*y);
}


static __device__ __host__ inline float ComplexAbsSquared(Complex in)
{
  //float val = ComplexAbs2(in);
  return in.x*in.x+in.y*in.y; //val*val;
}

static __device__ __host__ inline float ComplexRealAbs(Complex in)
{
  float val = ComplexReal(in);
  return abs(val);
}

static __device__ __host__ inline float ComplexImagAbs(Complex in)
{
  float val = ComplexImag(in);
  return abs(val);
}


static __device__ __host__ inline float ComplexAbs2(Complex in)
{
// Improved numeric stability compared to ComplexAbs based on Nvidias reference api
  float x = fabsf(in.x);
  float y = fabsf(in.y);
  float v,w,t;

  if (x>y)
  {
    v = x;
    w = y;
  }
  else
  {
    v = y;
    w = x;
  }
  t = w/v;
  t = 1.0f + t*t;

  t = v * sqrtf(t);
  if ((v == 0.0f) || (v>3.402823466e38f) || (w > 3.402823466e38f))
  {
    t = v+w;
  }
  return t;
  
}


// Helper functions for sum reduction algorithms

__inline__ __device__
float blockReduceSum(float val)
{
  static __shared__ float buf[WARP_SIZE]; // Shared mem to hold results of 32 partial sums
  const int lane = threadIdx.x % WARP_SIZE;
  const int warpIdx = threadIdx.x / WARP_SIZE;

  val = warpReduceSum(val); 	// Each warp performs partial reduction

  if (lane == 0)
    buf[warpIdx] = val; 	// Write reduced value to shared mem

  __syncthreads(); 		// Wait for all partial reductions in the block to finish

  // only the first warp in the grid needs to do the last partial sum
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? buf[lane] : 0;

  if (warpIdx == 0)
    val = warpReduceSum(val); // Final reduction

  return val;
}
 
__inline__ __device__
float warpReduceSum(float val)
{
  for (int i = WARP_SIZE/2; i > 0; i/=2)
    val += __shfl_xor_sync(FULL_MASK,val,i);
  return val;
}



