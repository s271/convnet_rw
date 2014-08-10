/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>

#include <layer_kernels.cuh>

template <int B_X, bool add>
__global__ void kEltwiseMaxGrad(float* actGrad, float* input, float* output, float* target,
                                const int numElements) {
    for (int i = B_X * blockIdx.x + threadIdx.x; i < numElements; i += B_X * gridDim.x) {
        if (add) {
            target[i] += actGrad[i] * (output[i] == input[i]);
        } else {
            target[i] = actGrad[i] * (output[i] == input[i]);
        }
    }
}

#include "tt.h"

#define CONST_AREA_SIZE 256
__device__ __constant__ float const_area[CONST_AREA_SIZE];

template <int sizeArr>
__global__ void kEltwiseFuncAct(const float* input, float* const target,
								const uint imgInPixels, const uint numCases,
								const uint strideInp, const uint strideTag,
								const uint sizeIn, const uint sizeOut) {

	const int numPixelsPerGroup = imgInPixels/sizeIn;	

//    dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(out_width, ELTWISE_THREADS_X)),
//                std::min(NUM_BLOCKS_MAX, DIVUP(numPixelsPerGroup, ELTWISE_THREADS_Y)));

// ix, iy == 0 almost always
    for (uint iy = 0; iy < numPixelsPerGroup; iy += gridDim.y*blockDim.y) {

        for (uint ix = 0; ix < numCases; ix += gridDim.x*blockDim.x) {	
			
			float inpVal[sizeArr];//use shared instead?
#pragma unroll
			for (uint inp_i = 0; inp_i < sizeIn; inp_i++) {	
				Offset inpOffset;
				inpOffset << Index(inp_i)
				<< numPixelsPerGroup
				<< Index(iy) << Index(blockDim.y, blockIdx.y) << Index(threadIdx.y)
				<< strideInp
				<< Index(ix ) << Index(blockDim.x, blockIdx.x) << Index(threadIdx.x);

				float val = input[inpOffset._offset];
				inpVal[inp_i] = val;
			}
#pragma unroll		
			for (uint out_i = 0; out_i < sizeOut; out_i++) {
				int out_par = out_i*sizeIn*2;

				float output = 0;
#pragma unroll			
				for (uint inp_i = 0; inp_i < sizeIn; inp_i++)
				{		
					float param = const_area[out_par + inp_i];
					float paramM = const_area[out_par + sizeIn + inp_i];
					float val = inpVal[inp_i];
					output += param*val + paramM*fmax(val, 0);
				}// inp_i

				Offset tagOffset;
				tagOffset << Index(out_i)
				<< numPixelsPerGroup
				<< Index(iy) << Index(blockDim.y, blockIdx.y) << Index(threadIdx.y)
				<< strideTag
				<< Index(ix ) << Index(blockDim.x, blockIdx.x) << Index(threadIdx.x);
				target[tagOffset._offset] = output;
			}//out_i
        }
    }


}

template <int B_X, int B_Y, int sizeArr>
__global__ void kEltwiseFuncAct_t(const float* input, float* const target,
								const uint imgInPixels, const uint numCases,
								const uint strideInp, const uint strideTag,
								const uint sizeIn, const uint sizeOut) {

	const int numPixelsPerGroup = imgInPixels/sizeIn;	

//    dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(out_width, ELTWISE_THREADS_X)),
//                std::min(NUM_BLOCKS_MAX, DIVUP(numPixelsPerGroup, ELTWISE_THREADS_Y)));

// ix, iy == 0 almost always

//go over output group
    const uint idxX = blockIdx.x * B_X + threadIdx.x;

//go over cases
    const uint idxY = blockIdx.y * B_Y + threadIdx.y;

	//gridDim.y is DIVUP(numPixelsPerGroup, ELTWISE_THREADS_Y)
    //for (uint iy = 0; iy < numPixelsPerGroup; iy += gridDim.y * B_Y)
	{

        //for (uint ix = 0; ix < numCases; ix += gridDim.x * B_X)
		{	

		float inpVal[sizeArr];//use shared instead?

			uint x = idxX;// + ix;
			uint yg = idxY;// + iy;

			for (uint inp_i = 0; inp_i < sizeIn; inp_i++) {	
				int yt = yg + inp_i*numPixelsPerGroup;
				float val = input[yt * strideInp + x];
				inpVal[inp_i] = val;
			}		
	
			for (uint out_i = 0; out_i < sizeOut; out_i++) {
				int out_par = out_i*sizeIn*2;

				float output = 0;
			
				for (uint inp_i = 0; inp_i < sizeIn; inp_i++)
				{		
					float param = const_area[out_par + inp_i];
					float paramM = const_area[out_par + sizeIn + inp_i];
					float val = inpVal[inp_i];
					output += param*val + paramM*fmax(val, 0);
				}// inp_i

				int yo = yg;//+ out_i*numPixelsPerGroup;
				int offseTag = yo * strideTag + x;
				target[offseTag] = output;
			}//out_i

        }
    }
}

template <int sizeArr>
__global__ void kEltwiseFuncGrad(const float* actGrad, const float* input, float* const target,
								const uint imgInPixels, const uint numCases,
								const uint strideInp, const uint strideOut,
								const uint sizeIn, const uint sizeOut) {


	const int numPixelsPerGroup = imgInPixels/sizeIn;	
	const int inStep = strideInp*numPixelsPerGroup;
	const int outStep = strideOut*numPixelsPerGroup;
//with no N_SUM ix, iy == 0 almost always
    for (uint iy = 0; iy < numPixelsPerGroup; iy += gridDim.y*blockDim.y) {
        for (uint ix = 0; ix < numCases; ix += gridDim.x*blockDim.x) {	

			float grad_next[sizeArr];

			Offset offset;
			offset 
			<< Index(iy) << Index(blockDim.y, blockIdx.y) << Index(threadIdx.y)
			<< strideInp
			<< Index(ix ) << Index(blockDim.x, blockIdx.x) << Index(threadIdx.x);

			for (uint out_i = 0; out_i < sizeOut; out_i++)
			{
				grad_next[out_i] = actGrad[offset._offset + outStep*out_i];
			}//out_i

			for (uint inp_i = 0; inp_i < sizeIn; inp_i++) {	
				int inp_offset = offset._offset + inp_i*inStep;

				float val = input[inp_offset];
				float vsign = (val > 0);
				float sum_grad = 0;
				
				for (uint out_i = 0; out_i < sizeOut; out_i++)	
					sum_grad += grad_next[out_i]
					*(vsign*const_area[out_i*sizeIn*2 + sizeIn + inp_i]
						+ const_area[out_i*sizeIn*2 + inp_i]);

				target[inp_offset] = sum_grad;
			}	

		}//ix
	}//iy

}

template <int B_X, int B_Y, int sizeArr>
__global__ void kEltwiseFuncGrad_t(const float* actGrad, const float* input, float* const target,
								const uint imgInPixels, const uint numCases,
								const uint strideInp, const uint strideOut,
								const uint sizeIn, const uint sizeOut) {

//go over output group
    const uint idxX = blockIdx.x * B_X + threadIdx.x;
//go over cases
    const uint idxY = blockIdx.y * B_Y + threadIdx.y;

	const int numPixelsPerGroup = imgInPixels/sizeIn;
//	const int inStep = strideInp*numPixelsPerGroup;
//	const int outStep = strideOut*numPixelsPerGroup;

	//gridDim.y is DIVUP(numOutPixelsPerGroup, ELTWISE_THREADS_Y)
    for (uint iy = idxY; iy < numPixelsPerGroup; iy += gridDim.y * B_Y) {

        for (uint ix = idxX; ix < numCases; ix += gridDim.x * B_X) {

			float grad_next[sizeArr];

			for (uint out_i = 0; out_i < sizeOut; out_i++)
				grad_next[out_i] = actGrad[(iy + out_i*numPixelsPerGroup)*strideOut + ix];

			for (uint inp_i = 0; inp_i < sizeIn; inp_i++) {	
				int yt = iy + inp_i*numPixelsPerGroup;
				int offset = yt * strideInp + ix;
				float val = input[offset];
				float vsign = (val > 0);
				float sum_grad = 0;
				
				for (uint out_i = 0; out_i < sizeOut; out_i++)	
					sum_grad += grad_next[out_i]*(vsign*const_area[out_i*sizeIn*2 + sizeIn + inp_i]
						+ const_area[out_i*sizeIn*2 + inp_i]); //optimize away later

				target[offset] = sum_grad;

			}	

			//int offset = y * stride + x;

			//float in0 = input0[offset];
			//float in1 = input1[offset];
			//float in2 = input2[offset];
			//float grad_next = actGrad[offset];

			//float val0 = param0 + param3*(in0 > 0);
			//float val1 = param1 + param4*(in1 > 0);
			//float val2 = param2 + param5*(in2 > 0);

			//target0[offset] = val0*grad_next;
			//target1[offset] = val1*grad_next;
			//target2[offset] = val2*grad_next;

   //         //float val = param0*in0 + param1*in1 + param2*in2 + param3*fm0 + param4*fm1 + param5*fm2;

		}
   }

}

__global__ void kEltwiseFuncParamGradSingle(float* actGrad, float* input, float* target, float* target_m,
								const uint pin, const uint pout, const uint imgInPixels, const uint numCases,
								const uint strideInp, const uint strideOut, const uint strideTag,
								const uint sizeIn, const uint sizeOut)
{
	const int numPixelsPerGroup = imgInPixels/sizeIn;	


	float sum = 0;
	float sum_m = 0;

#pragma unroll	
    for (uint iy = 0; iy < numPixelsPerGroup; iy += gridDim.y*blockDim.y) {
#pragma unroll
      for (uint ix = 0; ix < numCases; ix += gridDim.x*blockDim.x) {	

			Offset offsetInp;
			offsetInp
			<< Index(pin)
			<< numPixelsPerGroup
			<< Index(iy) << Index(blockDim.y, blockIdx.y) << Index(threadIdx.y)
			<< strideInp
			<< Index(ix ) << Index(blockDim.x, blockIdx.x) << Index(threadIdx.x);
			
			float in_val = input[offsetInp._offset];

			Offset offsetOut;
			offsetOut
			<< Index(pout)
			<< numPixelsPerGroup
			<< Index(iy) << Index(blockDim.y, blockIdx.y) << Index(threadIdx.y)
			<< strideOut
			<< Index(ix ) << Index(blockDim.x, blockIdx.x) << Index(threadIdx.x);

			float grad_next = actGrad[offsetOut._offset];

			float val_m = fmax(in_val, 0);
			sum += grad_next*in_val;
			sum_m += grad_next*val_m;
		}
	}

	Offset offsetTag;
	offsetTag
	<< Index(blockDim.y, blockIdx.y) << Index(threadIdx.y)
	<< strideTag
	<< Index(blockDim.x, blockIdx.x) << Index(threadIdx.x);

	target[offsetTag._offset] = sum;
	target_m[offsetTag._offset] = sum_m;

}


template <int B_X, int B_Y>
__global__ void kEltwiseFuncParamGradSingle_t(float* actGrad, float* input, float* target, float* target_m,
								const uint pin, const uint pout, const uint imgInPixels, const uint numCases,
								const uint strideInp, const uint strideOut, const uint strideTag,
								const uint sizeIn, const uint sizeOut)
{
	const int numPixelsPerGroup = imgInPixels/sizeIn;	
    const uint idxX = blockIdx.x * B_X + threadIdx.x;
    const uint idxY = blockIdx.y * B_Y + threadIdx.y;

	float sum = 0;
	float sum_m = 0;

#pragma unroll
    for (uint y = idxY; y < numPixelsPerGroup; y += gridDim.y * B_Y) {
#pragma unroll
        for (uint x = idxX; x < numCases; x += gridDim.x * B_X) {
			int offset = y * strideInp + x;
			float in_val = input[offset + pin*numPixelsPerGroup* strideInp ];

			float grad_next = actGrad[y * strideOut + x + pout*numPixelsPerGroup* strideInp ];

			float val_m = fmax(in_val, 0);
			sum += grad_next*in_val;
			sum_m += grad_next*val_m;
		}
	}
	int tagOffset = (threadIdx.x + blockIdx.x*blockDim.x) +  (threadIdx.y + blockIdx.y*blockDim.y)*strideTag;

	target[tagOffset] = sum;
	target_m[tagOffset] = sum_m;

}
#define SMEM(X, Y, sdata) sdata[(X)*smem_sizeY+(Y)]
#define SHARED_MEM(x, y, z, RSH, getVal, sdata) \
    SMEM((RSH) + sx, (RSH) + sy, sdata) = getVal(x, y, z);\
    if (sx < (RSH)) {\
        SMEM(sx, (RSH) + sy, sdata) = getVal(max(x - (RSH), 0), y, z);\
        SMEM((RSH) + bw + sx, (RSH) + sy, sdata) = getVal(min(x + bw, imgSizeX), y, z);\
    }\
    if (sy < (RSH)) {\
        SMEM((RSH) + sx, sy, sdata) = getVal(x, max(y - (RSH), 0), z);\
        SMEM((RSH) + sx, (RSH) + bh + sy, sdata) = getVal(x, min(y + bh, imgSizeY), z);\
    }\
    if ((sx < (RSH)) && (sy < (RSH))) {\
        SMEM(sx, sy, sdata) = getVal(max(x - (RSH), 0), max(y - (RSH), 0), z);\
        SMEM(sx, (RSH) + bh + sy, sdata) = getVal(max(x - (RSH), 0), min(y + bh, imgSizeY), z);\
        SMEM((RSH) + bw + sx, sy, sdata) = getVal(min(x + bh, imgSizeX), max(y - (RSH), 0), z);\
        SMEM((RSH) + bw + sx, (RSH) + bh + sy, sdata) = getVal(min(x + bw, imgSizeX), min(y + bh, imgSizeY), z);\
    }

#define getValInput(X, Y, Z) input[channelOffset + (X)*widthyz+(Y)*widthz + (Z)]

__global__ void kMicroConvAct4Channel(const float* input, float* const target,
								const uint numCases, const uint channels, const uint numFilters, 
								const uint modulesPerBlockX, const uint modulesPerBlockY,
								const uint sizeModule, const uint lobe,
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels)
{
	extern __shared__ float sdata[];
//order x>y>z, *not* y>x
    const int  bw = blockDim.x;
    const int  bh = blockDim.y;

	
    const int  ix = threadIdx.x/imgSizeY;
    const int  iy = threadIdx.y - ix*imgSizeY;

	const int widthz = numCases;
	const int widthyz = imgSizeX*numCases;

	const int sizeModule2 = sizeModule*sizeModule;

	const int smem_sizeY = modulesPerBlockY + 2*lobe;
    const int  sx = threadIdx.y/smem_sizeY;
    const int  sy = threadIdx.y - sx*smem_sizeY;

//put pragme unroll here	
	for(int channelInd = 0; channelInd < channels; channelInd++)
		for(int zs = 0; zs < gridDim.x; zs++)
		{	
			const int channelOffset = channelInd*imgPixels*numCases;
			int z = threadIdx.x + zs*blockDim.x;
			if(z < numCases)
			{
				SHARED_MEM(ix, iy, z, lobe, getValInput, sdata)	
				__syncthreads();

				for(int filterID = 0; filterID <  numFilters; filterID++)
				{
						float sum = 0;
						for(int isx = 0; isx <  sizeModule; isx++)
						for(int isy = 0; isy <  sizeModule; isy++)
							sum += sdata[(sx + isx - lobe)*smem_sizeY+(sy + isy - lobe)]
									*const_area[filterID*sizeModule2 + isy*sizeModule + isx];

						target[numFilters*channelOffset + filterID*imgPixels*numCases + ix*widthyz + iy*widthz + z] = sum;

				}
			}
		}
}
#define getValAct(X, Y, Z) actGrad[filterOffset + (X)*widthyz+(Y)*widthz + (Z)]

__global__ void kMicroConvGrad(const float* actGrad, float* const target,
								const uint numCases, const uint channels, const uint numFilters, 
								const uint modulesPerBlockX, const uint modulesPerBlockY,
								const uint sizeModule, const uint lobe,
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels)
{
	extern __shared__ float sdata[];
//order x>y>z, *not* y>x
    const int  bw = blockDim.x;
    const int  bh = blockDim.y;

	
    const int  ix = threadIdx.x/imgSizeY;
    const int  iy = threadIdx.y - ix*imgSizeY;

	const int widthz = numCases;
	const int widthyz = imgSizeX*numCases;

	const int sizeModule2 = sizeModule*sizeModule;

	const int smem_sizeY = modulesPerBlockY + 2*lobe;
    const int  sx = threadIdx.y/smem_sizeY;
    const int  sy = threadIdx.y - sx*smem_sizeY;


//pragma unroll here
	for(int channelInd = 0; channelInd < channels; channelInd++)
	{
		const int channelOffset = channelInd*imgPixels*numCases;

		for(int zs = 0; zs < gridDim.x; zs++)
		{	
			int z = threadIdx.x + zs*blockDim.x;
			if(z < numCases)
			{
				float sum = 0;
				for(int filterID = 0; filterID <  numFilters; filterID++)
				{
					const int filterOffset = numFilters*channelOffset + filterID*imgPixels*numCases;
					SHARED_MEM(ix, iy, z, lobe, getValAct, sdata)	
					__syncthreads();

					
					for(int dsx = - lobe; dsx < lobe+1; dsx++)
					for(int dsy = - lobe; dsy <  lobe+1; dsy++)
						sum += sdata[(sx + dsx + lobe)*smem_sizeY+(sy + dsy + lobe)]
								*const_area[filterID*sizeModule2 + (-dsy + lobe)*sizeModule +(-dsx + lobe)];

				}
				target[channelOffset + ix*widthyz + iy*widthz + z] = sum;
			}
		}
	}
}

__global__ void kMicroConvWeightGrad(const float* actGrad, const float* input, float** const target,
								const uint target_size, const uint numCases,
								const uint channels, const uint numFilters, 
								const uint modulesPerBlockX, const uint modulesPerBlockY,
								const uint lobe, const uint sizeModule,
								const uint imgSizeX, const uint imgSizeY, const uint imgPixels)
{

//order x>y>z, *not* y>x
	extern __shared__ float sdataAct[];	
	extern __shared__ float sdataImg[];	

    const int  bw = blockDim.x;
    const int  bh = blockDim.y;

	
    const int  ix = threadIdx.x/imgSizeY;
    const int  iy = threadIdx.y - ix*imgSizeY;

	const int widthz = numCases;
	const int widthyz = imgSizeX*numCases;

	const int sizeModule2 = sizeModule*sizeModule;

	const int smem_sizeY = modulesPerBlockY + 2*lobe;
    const int  sx = threadIdx.y/smem_sizeY;
    const int  sy = threadIdx.y - sx*smem_sizeY;
	const int imgSize = imgSizeX*imgSizeY;


//pragma unroll here
	for(int channelInd = 0; channelInd < channels; channelInd++)
	{
		const int channelOffset = channelInd*imgSize*numCases;

		for(int zs = 0; zs < gridDim.x; zs++)
		{	
			int z = threadIdx.x + zs*blockDim.x;
			if(z < numCases)
			{
				float sum = 0;
				for(int filterID = 0; filterID <  numFilters; filterID++)
				{
					const int filterOffset = numFilters*channelOffset + filterID*imgSize*numCases;

					SHARED_MEM(ix, iy, z, lobe, getValAct, sdataAct)	
					SHARED_MEM(ix, iy, z, lobe, getValAct, sdataImg)
					__syncthreads();				

					
					for(int dsx = - lobe; dsx < lobe+1; dsx++)
					for(int dsy = - lobe; dsy <  lobe+1; dsy++)
					{
						sum = 0;
						sum += sdataAct[(sx + dsx + lobe)*smem_sizeY+(sy + dsy + lobe)]*sdataAct[sx*smem_sizeY+ sy];
						int ind_coeff = filterID*sizeModule2 + (-dsy + lobe)*sizeModule +(-dsx + lobe);
						*(target + ind_coeff*target_size)[channelOffset + ix*widthyz + iy*widthz + z] = sum;
					}
				}
			}
		}
	}

}


void computeEltwiseMaxGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& output, NVMatrix& target, bool add) {
    assert(actGrad.isContiguous());
    assert(output.isContiguous());
    assert(input.isContiguous());
    assert(actGrad.isSameDims(input));
    assert(actGrad.isSameDims(output));
  
    dim3 blocks(DIVUP(actGrad.getNumElements(), 128));
    dim3 threads(128);
    if (add) {
        assert(actGrad.isSameDims(target));
        cudaFuncSetCacheConfig(kEltwiseMaxGrad<128, true>, cudaFuncCachePreferL1);
        kEltwiseMaxGrad<128, true><<<blocks, threads>>>(actGrad.getDevData(), input.getDevData(), output.getDevData(), target.getDevData(), actGrad.getNumElements());
    } else {
        target.resize(actGrad);
        cudaFuncSetCacheConfig(kEltwiseMaxGrad<128, false>, cudaFuncCachePreferL1);
        kEltwiseMaxGrad<128, false><<<blocks, threads>>>(actGrad.getDevData(), input.getDevData(), output.getDevData(), target.getDevData(), actGrad.getNumElements());
    }
    
    cutilCheckMsg("computeEltwiseMaxGrad: Kernel execution failed");
}

void computeEltwiseFuncParamGradSingle(NVMatrix& actGrad, NVMatrix& input,
								 NVMatrix& target, NVMatrix& target_m,
								 int pin, int pout, int size_in, int size_out)
{

    int inp_width = input.getNumCols(); 
    int inp_height = input.getNumRows();


	int numPixelsPerGroup = inp_height/size_in;
//	printf("inp_height %i numPixelsPerGroup %i \n", inp_height, numPixelsPerGroup);
#define N_SUM 1
    dim3 threads(min(ELTWISE_THREADS_X, inp_width), ELTWISE_THREADS_Y);
    dim3 blocks(std::min(NUM_BLOCKS_MAX, (int)DIVUP(inp_width, threads.x)),
                std::min(NUM_BLOCKS_MAX, (int)DIVUP(numPixelsPerGroup/N_SUM, ELTWISE_THREADS_Y)));
#undef N_SUM

	int sizeX = blocks.x*threads.x;
	int sizeY = blocks.y*threads.y;

    if (target.getNumCols() != sizeX || target.getNumRows() != sizeY) {
		//printf(" tresize %i %i \n", sizeX, sizeY);
        target.resize(sizeY, sizeX);// numRows, numCols !
    }

	//printf(" target.getNumCols() %i target.getNumRows() %i elem %i   \n", target.getNumCols(), target.getNumRows(), target.getNumElements());

	//printf(" aft memset\n");
	//float rr = target.sum();
	//printf(" sum %f \n", rr);

    if (!target_m.isSameDims(target)) {
        target_m.resize(target);
    }
	//cudaMemset(target_m.getDevData(), 0, sizeX*sizeY*sizeof(float));
	//cudaMemset(target.getDevData(), 0, sizeX*sizeY*sizeof(float));	

	//printf(" target.getStride() %i sizeX %i sizeY %i target.isTrans() %i actGrad.getStride() %i \n", 
	//	target.getStride(),sizeX, sizeY, target.isTrans(), actGrad.getStride());


	//printf(" numPixelsPerGroup %i actGrad.getNumRows %i \n", numPixelsPerGroup, actGrad.getNumRows());

	//float ar1 = actGrad.sum();
	//float ir2 = input.sum();
	//printf("sum actGrad  %f input %f \n", ar1, ir2);


	cudaFuncSetCacheConfig(kEltwiseFuncParamGradSingle, cudaFuncCachePreferL1);

	//kEltwiseFuncParamGradSingle_t<ELTWISE_THREADS_X, ELTWISE_THREADS_Y><<<blocks, threads>>>(actGrad.getDevData(),
	//	input.getDevData(), target.getDevData(), target_m.getDevData(),
	//	pin, pout, inp_height, inp_width,
	//	input.getStride(), actGrad.getStride(), target.getStride(),
	//	size_in, size_out);

	//float rr11 = target.sum();
	//float rr21 = target_m.sum();
	//printf(" sum aft %f %f \n", rr11, rr21);
	//printf("sum1 actGrad  %f input %f \n", ar1, ir2);

	kEltwiseFuncParamGradSingle<<<blocks, threads>>>(actGrad.getDevData(),
		input.getDevData(), target.getDevData(), target_m.getDevData(),
		pin, pout, inp_height, inp_width,
		input.getStride(), actGrad.getStride(), target.getStride(),
		size_in, size_out);

	//float rr1 = target.sum();
	//float rr2 = target_m.sum();
	//printf(" sum1 aft %f %f \n", rr1, rr2);

 /*       int height = input0.getFollowingDim(), width = input0.getLeadingDim();

        dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
                    std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
        dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);

		int sizeX = blocks.x*threads.x;
		int sizeY = blocks.y*threads.y;

        if (target0.getNumRows() != sizeX || target0.getNumCols() !=  sizeY) {
            target0.resize(sizeX, sizeY);
        }

		////debug
  //      if (!target0.isSameDims(input0)) {
  //          target0.resize(input0);
  //      }//shortening is not working

        if (!target1.isSameDims(target0)) {
            target1.resize(target0);
        }

        if (!target2.isSameDims(target0)) {
            target2.resize(target0);
        }

        if (!target3.isSameDims(target0)) {
            target3.resize(target0);
        }

        if (!target4.isSameDims(target0)) {
            target4.resize(target0);
        }

        if (!target5.isSameDims(target0)) {
            target5.resize(target0);
        }

		cudaFuncSetCacheConfig(kEltwiseFuncParamGrad<ELTWISE_THREADS_X, ELTWISE_THREADS_Y>, cudaFuncCachePreferL1);

		kEltwiseFuncParamGrad<ELTWISE_THREADS_X, ELTWISE_THREADS_Y><<<blocks, threads>>>(actGrad.getDevData(), 
			input0.getDevData(), input1.getDevData(), input2.getDevData(),
			target0.getDevData(), target1.getDevData(), target2.getDevData(), target3.getDevData(), target4.getDevData(), target5.getDevData(),
			height, width, input0.getStride(), sizeX);
*/
		cutilCheckMsg("kEltwiseFuncParamGrad: Kernel execution failed");
};

void computeEltwiseFuncAct(NVMatrix& input, NVMatrix& target, vector<double>& param, int size_in, int size_out)
{

	assert(size_in <= 4 || size_in == 6 || size_in == 8 || size_in == 12 || size_in == 16);
	//int height = input.getFollowingDim(), width = input.getLeadingDim();	
    //int numCases = input.getNumCols(); 
    //int numIn = input.getNumRows(); 

    int inp_width = input.getNumCols(); 
    int inp_height = input.getNumRows();

	int out_width = inp_width;
	int out_height = (inp_height*size_out)/size_in;

	//printf(" inp_height %i inp_width %i \n",inp_height, inp_width);
	//printf(" size_in %i size_out %i \n", size_in, size_out);
	//printf(" out_height %i out_width %i \n",out_height, out_width);

    if (target.getNumCols() != out_width || target.getNumRows() != out_height) {
        target.resize(out_height, out_width);
		//printf("**resize out_height %i out_width %i \n",out_height, out_width);
    }

	float temp[CONST_AREA_SIZE];
	assert(param.size() <= CONST_AREA_SIZE);
	memset(temp, 0, sizeof(temp));
	for(int i = 0; i < param.size(); i++)
		temp[i] = (float)param[i];
	cudaMemcpyToSymbol(const_area, temp, sizeof(float)*CONST_AREA_SIZE, 0, cudaMemcpyHostToDevice);

	int numPixelsPerGroup = out_height/size_out;

    dim3 threads(min(ELTWISE_THREADS_X, inp_width), ELTWISE_THREADS_Y);
    dim3 blocks(std::min(NUM_BLOCKS_MAX, (int)DIVUP(out_width, threads.x)),
                std::min(NUM_BLOCKS_MAX, DIVUP(numPixelsPerGroup, ELTWISE_THREADS_Y)));

//debug
	//printf("kEltwiseFuncAct -------------\n");
	//printf("temp %f %f %f  %f %f %f \n", temp[0],temp[1],temp[2],temp[3],temp[4],temp[5]);
	//input.nan2zero();
	//float sum = input.sum();
	//printf(" size_in %i size_out %i sum %f \n", size_in, size_out, sum);
//	const int numPixelsPerGroup1 = inp_height/size_in;
//	printf(" numPixelsPerGroup %i numPixelsPerGroup1 %i target.getNumRows %i \n", numPixelsPerGroup, numPixelsPerGroup1, target.getNumRows());
//	//cudaMemset(target.getDevData(), 0, target.getNumElements()*sizeof(float));
//printf(" target.getStride() %i target.getNumRows() %i target.getNumCols() %i \n", target.getStride(), target.getNumRows(), target.getNumCols());

	//kEltwiseFuncAct_t<ELTWISE_THREADS_X, ELTWISE_THREADS_Y, 3><<<blocks, threads>>>(input.getDevData(),
	//target.getDevData(), inp_height, inp_width, input.getStride(), target.getStride(), size_in, size_out);
	//float sumt0 = target.sum();
	//printf("kEltwiseFuncAct_t sumt_0 %f \n", sumt0);


#define ELT_ACT(SIZE_ARR) \
	if(size_in == SIZE_ARR){\
	cudaFuncSetCacheConfig(kEltwiseFuncAct<SIZE_ARR>, cudaFuncCachePreferL1);\
	kEltwiseFuncAct<SIZE_ARR><<<blocks, threads>>>(input.getDevData(),\
	target.getDevData(), inp_height, inp_width, input.getStride(), target.getStride(), size_in, size_out);};
	ELT_ACT(1)
	ELT_ACT(2)
	ELT_ACT(3)
	ELT_ACT(4)
	ELT_ACT(6)
	ELT_ACT(8)
	ELT_ACT(12)
	ELT_ACT(16)
#undef ELT_ACT

//float sumt = target.sum();
//	printf("kEltwiseFuncAct sumt %f \n", sumt);

	cutilCheckMsg("computeEltwiseFuncAct: Kernel execution failed");
}

void computeEltwiseFuncGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& target,
								 vector<double>& param, int size_in, int size_out)
{


	assert(size_out <= 4 || size_out == 6 || size_out == 8 || size_out == 12 || size_out == 16);
	//int height = input.getFollowingDim(), width = input.getLeadingDim();	
    //int numCases = input.getNumCols(); 
    //int numIn = input.getNumRows(); 

    int inp_width = input.getNumCols(); 
    int inp_height = input.getNumRows();

    if (target.getNumCols() != inp_width || target.getNumRows() != inp_height) {
        target.resize(inp_height, inp_width);
    }

	float temp[CONST_AREA_SIZE];
	assert(param.size() <= CONST_AREA_SIZE);
	memset(temp, 0, sizeof(temp));
	for(int i = 0; i < param.size(); i++)
		temp[i] = (float)param[i];
	cudaMemcpyToSymbol(const_area, temp, sizeof(float)*CONST_AREA_SIZE, 0, cudaMemcpyHostToDevice);


	int numPixelsPerGroup = inp_height/size_in;

    dim3 threads(min(ELTWISE_THREADS_X, inp_width), ELTWISE_THREADS_Y);

    dim3 blocks(std::min(NUM_BLOCKS_MAX, (int)DIVUP(inp_width, threads.x)),
                std::min(NUM_BLOCKS_MAX, DIVUP(numPixelsPerGroup, ELTWISE_THREADS_Y)));

	//printf("computeEltwiseFuncGrad numPixelsPerGroup %i --------------------\n", numPixelsPerGroup);
	//float sumA = actGrad.sum();
	//float sumI = input.sum();

	//printf("sum actGrad %f input %f \n", sumA, sumI);
	//printf(" size_in %i size_out %i tag size %i sumt %f \n", size_in, size_out,  target.getNumElements());
	//printf(" target.getStride() %i actGrad  %i input %i \n", target.getStride(), actGrad.getNumRows(), input.getNumRows());

	//kEltwiseFuncGrad_t<ELTWISE_THREADS_X, ELTWISE_THREADS_Y, 3><<<blocks, threads>>>(actGrad.getDevData(),
	//	input.getDevData(), target.getDevData(), inp_height, inp_width,
	//	input.getStride(), actGrad.getStride(), size_in, size_out);

	//float sumtt = target.sum();
	//printf("sum_test_tag %f \n", sumtt);


#define ELT_GRAD(SIZE_ARR) \
		if(size_out == SIZE_ARR){\
			cudaFuncSetCacheConfig(kEltwiseFuncGrad<SIZE_ARR>, cudaFuncCachePreferL1);\
			kEltwiseFuncGrad<SIZE_ARR><<<blocks, threads>>>(actGrad.getDevData(),\
				input.getDevData(), target.getDevData(), inp_height, inp_width,\
				input.getStride(), actGrad.getStride(), size_in, size_out);};
		ELT_GRAD(1)
		ELT_GRAD(2)
		ELT_GRAD(3)
		ELT_GRAD(4)
		ELT_GRAD(6)
		ELT_GRAD(8)
		ELT_GRAD(12)
		ELT_GRAD(16)
#undef ELT_GRAD

	float sumt = target.sum();

//	printf("sum_tag %f \n", sumt);


		cutilCheckMsg("computeEltwiseFuncGrad: Kernel execution failed");
};


void computeMicroConvAct(NVMatrix& input, NVMatrix& target, vector<double>& param, int sizeModule, int channels,
						 int imgSize, int imgPixels, int numFilters, int filterChannels, int groups)
{
	int out_width = input.getNumCols();
	int out_height = input.getNumRows()*numFilters;

    if (target.getNumCols() != out_width || target.getNumRows() != out_height) {
        target.resize(out_height, out_width);
		//printf("**resize out_height %i out_width %i \n",out_height, out_width);
    }

	int numImages = out_width;
	int imgSizeX = imgSize;
	int imgSizeY = imgSize;

	int img_threads_x = 16;
	int img_threads_y = 16;
	int imgsPerThread = 16;//~number of blocks
	int case_threads = DIVUP(numImages, imgsPerThread); 

	int lobe = sizeModule/2;

	//blocks.x = imgsPerThread

	int sharedX = lobe*2 + img_threads_x;
	int sharedY = lobe*2 + img_threads_y;
	int shared_size = sharedX*sharedY*case_threads*imgsPerThread;

	dim3 threads(case_threads, img_threads_x*img_threads_y);


	dim3 blocks = dim3(DIVUP(numImages,threads.x*imgsPerThread), DIVUP(imgSizeY,img_threads_x) * DIVUP(imgSizeX,img_threads_y));

    //dim3 threads(min(ELTWISE_THREADS_X, inp_width), ELTWISE_THREADS_Y);
    //dim3 blocks(std::min(NUM_BLOCKS_MAX, (int)DIVUP(out_width, threads.x)),
    //            std::min(NUM_BLOCKS_MAX, DIVUP(numPixelsPerGroup, ELTWISE_THREADS_Y)));


	float temp[CONST_AREA_SIZE];
	assert(param.size() <= CONST_AREA_SIZE);
	memset(temp, 0, sizeof(temp));
	for(int i = 0; i < param.size(); i++)
		temp[i] = (float)param[i];
	cudaMemcpyToSymbol(const_area, temp, sizeof(float)*CONST_AREA_SIZE, 0, cudaMemcpyHostToDevice);

	//cudaFuncSetCacheConfig(kMicroConvAct, cudaFuncCachePreferShared );

};