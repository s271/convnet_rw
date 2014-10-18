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
//-------------------------------------------------------------
//EltwiseMax
//-------------------------------------------------------------
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

//-------------------------------------------------------------
//EltwiseFunc
//-------------------------------------------------------------
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
//-------------------------------------------------------------
//MicroConv
//-------------------------------------------------------------
#define SMEM(X, Y, sdata) sdata[(X)*sharedY+(Y) + sOffset]

#define SHARED_MEM(x, y, z, LOBE, getVal, sdata) \
    SMEM((LOBE) + sx, (LOBE) + sy, sdata) = getVal(x, y, z);\
    if (sx < (LOBE)) {\
        SMEM(sx, (LOBE) + sy, sdata) = getVal(max(x - (LOBE), 0), y, z);\
        SMEM((LOBE) + bw + sx, (LOBE) + sy, sdata) = getVal(min(x + bw, imgSizeX-1), y, z);\
    }\
    if (sy < (LOBE)) {\
        SMEM((LOBE) + sx, sy, sdata) = getVal(x, max(y - (LOBE), 0), z);\
        SMEM((LOBE) + sx, (LOBE) + bh + sy, sdata) = getVal(x, min(y + bh, imgSizeY-1), z);\
    }\
    if ((sx < (LOBE)) && (sy < (LOBE))) {\
        SMEM(sx, sy, sdata) = getVal(max(x - (LOBE), 0), max(y - (LOBE), 0), z);\
        SMEM(sx, (LOBE) + bh + sy, sdata) = getVal(max(x - (LOBE), 0), min(y + bh, imgSizeY-1), z);\
        SMEM((LOBE) + bw + sx, sy, sdata) = getVal(min(x + bw, imgSizeX-1), max(y - (LOBE), 0), z);\
        SMEM((LOBE) + bw + sx, (LOBE) + bh + sy, sdata) = getVal(min(x + bw, imgSizeX-1), min(y + bh, imgSizeY-1), z);\
    }

#define getValInput(X, Y, Z) input[channelOffset + (X)*widthyz+(Y)*widthz + (Z)]

template < int LOBE, int SIZE_CONV>
__global__ void kMicroConvFilterAct(const float* input, float* const target,
								const uint numCases, const uint channels, const uint numFilters, const uint casePerThread,
								const uint sharedY, const uint modulesPerBlockX,  const uint modulesPerBlockY, 
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels)
{
	extern __shared__ float sdata[];
//order x>y>z, *not* y>x
	const int bsizeX = imgSizeX/modulesPerBlockX;
	const int bsizeY = imgSizeY/modulesPerBlockY;
	const int startX = (blockIdx.y/bsizeY)*modulesPerBlockX;
	const int startY = (blockIdx.y%bsizeY)*modulesPerBlockY;

    const int  bw = modulesPerBlockX;
    const int  bh = modulesPerBlockY;
    const int  sx = threadIdx.y/modulesPerBlockY;
    const int  sy = threadIdx.y - sx*modulesPerBlockY;

	const int  ix = sx+startX;
	const int  iy = sy+startY;

	const int widthz = numCases;
	const int widthyz = imgSizeY*numCases;

	const int sizeConv2 = SIZE_CONV*SIZE_CONV;
	const int sharedY2 = sharedY*sharedY;


//put pragme unroll here	
	for(int zind = 0; zind < casePerThread; zind++)
	{
		const int z = threadIdx.x + blockIdx.x*blockDim.x + zind*blockDim.x*gridDim.x;			
		for(int channelInd = 0; channelInd < channels; channelInd++)
		{	
			const int sOffset = channelInd*sharedY2*blockDim.x + threadIdx.x*sharedY2;
			const int channelOffset = channelInd*imgPixels*numCases;

			if(z < numCases)
			{

				SHARED_MEM(ix, iy, z, LOBE, getValInput, sdata)	
			}
		}

		__syncthreads();

		for(int channelInd = 0; channelInd < channels; channelInd++)
		{	
			const int sOffset = channelInd*sharedY2*blockDim.x + threadIdx.x*sharedY2;
			const int channelOffset = channelInd*imgPixels*numCases;

			if(z < numCases)
			{
				for(int filterID = 0; filterID <  numFilters; filterID++)
				{
						float sum = 0;

						for(int dsx = - LOBE; dsx < LOBE+1; dsx++)
						for(int dsy = - LOBE; dsy <  LOBE+1; dsy++)
						{
							int idx = min(max(ix + dsx, 0), imgSizeX-1);
							int idy = min(max(iy + dsy, 0), imgSizeY-1);

							float sd = sdata[(sx + dsx + LOBE)*sharedY+(sy + dsy + LOBE) + sOffset];

							sum += sd*const_area[channelInd*sizeConv2*numFilters + filterID*sizeConv2 + (dsy + LOBE)*SIZE_CONV +(dsx + LOBE)];
						}
									
						target[numFilters*channelOffset + filterID*imgPixels*numCases + ix*widthyz + iy*widthz + z] = sum;

				}//filter
			}//if
		}//channel
	}//zind
}
#define getValAct(X, Y, Z) actGrad[filterOffset + (X)*widthyz+(Y)*widthz + (Z)]

__global__ void kMicroConvActGrad(const float* actGrad, float* const target,
								const uint numCases, const uint channels, const uint numFilters, const uint casePerThread,
								const uint modulesPerBlockX, const uint modulesPerBlockY,
								const uint sharedY, const uint sizeModule, const uint lobe,
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels)
{
	extern __shared__ float sdata[];
//order x>y>z, *not* y>x
	
	const int bsizeX = imgSizeX/modulesPerBlockX;
	const int bsizeY = imgSizeY/modulesPerBlockY;
	const int startX = (blockIdx.y/bsizeY)*modulesPerBlockX;
	const int startY = (blockIdx.y%bsizeY)*modulesPerBlockY;

    const int  bw = modulesPerBlockX;
    const int  bh = modulesPerBlockY;
    const int  sx = threadIdx.y/modulesPerBlockY;
    const int  sy = threadIdx.y - sx*modulesPerBlockY;

	const int  ix = sx+startX;
	const int  iy = sy+startY;

	const int widthz = numCases;
	const int widthyz = imgSizeY*numCases;

	const int sizeModule2 = sizeModule*sizeModule;
	const int sharedY2 = sharedY*sharedY;

	for(int zind = 0; zind < casePerThread; zind++)
	{
		const int z = threadIdx.x + blockIdx.x*blockDim.x + zind*blockDim.x*gridDim.x;		
	//pragma unroll here

		for(int channelInd = 0; channelInd < channels; channelInd++)
		{
			const int channelOffset = channelInd*imgPixels*numCases;

			float sum = 0;
			for(int filterID = 0; filterID <  numFilters; filterID++)
			{
				const int sOffset = channelInd*numFilters*sharedY2*blockDim.x + filterID*sharedY2*blockDim.x + threadIdx.x*sharedY2;
				const int filterOffset = numFilters*channelOffset + filterID*imgPixels*numCases;

				SHARED_MEM(ix, iy, z, lobe, getValAct, sdata)	
			}
		}

		__syncthreads();

		for(int channelInd = 0; channelInd < channels; channelInd++)
		{
			const int channelOffset = channelInd*imgPixels*numCases;

			float sum = 0;
			for(int filterID = 0; filterID <  numFilters; filterID++)
			{
				const int sOffset = channelInd*numFilters*sharedY2*blockDim.x + filterID*sharedY2*blockDim.x + threadIdx.x*sharedY2;
				const int filterOffset = numFilters*channelOffset + filterID*imgPixels*numCases;
				
				for(int dsx = - lobe; dsx < lobe+1; dsx++)
				for(int dsy = - lobe; dsy <  lobe+1; dsy++)
					sum += sdata[(sx + dsx + lobe)*sharedY+(sy + dsy + lobe)]
							*const_area[filterID*sizeModule2 + (-dsy + lobe)*sizeModule +(-dsx + lobe)];

			}
			target[channelOffset + ix*widthyz + iy*widthz + z] = sum;
		}
	}
}

template <int lobe>
__global__ void kMicroConvWeightGrad(const float* actGrad, const float* input, float** const target,
								const uint target_size, const uint numCases, const uint casePerThread, const uint tagWidth,
								const uint channels, const uint numFilters, 
								const uint modulesPerBlockX, const uint modulesPerBlockY,
								const uint imgSizeX, const uint imgSizeY, const uint imgPixels)
{

//order x>y>z, *not* y>x
	extern __shared__ float sdata[];
	const int imgSize = imgSizeX*imgSizeY;
	const int sharedY = modulesPerBlockY + 2*lobe;
	const int sizeSharedBlock = sharedY*(modulesPerBlockX + 2*lobe);
	float* sdataImg = sdata;
	float* sdataRes = sdata + sizeSharedBlock*blockDim.x;

	const int bsizeX = imgSizeX/modulesPerBlockX;
	const int bsizeY = imgSizeY/modulesPerBlockY;
	const int startX = (blockIdx.y/bsizeY)*modulesPerBlockX;
	const int startY = (blockIdx.y%bsizeY)*modulesPerBlockY;

    const int  bw = modulesPerBlockX;
    const int  bh = modulesPerBlockY;
    const int  sx = threadIdx.y/modulesPerBlockY;
    const int  sy = threadIdx.y - sx*modulesPerBlockY;

	const int  ix = sx+startX;
	const int  iy = sy+startY;

	const int zoff = threadIdx.x + blockIdx.x*blockDim.x;

	const int widthz = numCases;
	const int widthyz = imgSizeY*numCases;

	const int sharedY2 = sharedY*sharedY;

	const int conv_size = 2*lobe+1;
	const int conv2 = conv_size*conv_size;

	int resStride = numFilters*conv2;
	int res_off = resStride*(threadIdx.y*blockDim.x + threadIdx.x);

	const int sOffset = threadIdx.x*sharedY2;

	for(int channelInd = 0; channelInd < channels; channelInd++)
	{
		const int channelOffset = channelInd*imgPixels*numCases;

		memset(sdataRes + res_off, 0, resStride*sizeof(float));

		for(int zind = 0; zind < casePerThread; zind++)
		{

			const int z = zoff + zind*blockDim.x*gridDim.x;		
			for(int filterID = 0; filterID <  numFilters; filterID++)
			{

				SHARED_MEM(ix, iy, z, lobe, getValInput, sdataImg)	

				__syncthreads();

				for(int dsx = - lobe; dsx < lobe+1; dsx++)
				for(int dsy = - lobe; dsy < lobe+1; dsy++)
				{
					int idx = min(max(ix + dsx, 0), imgSizeX-1);
					int idy = min(max(iy + dsy, 0), imgSizeY-1);

					const int filterOffset = numFilters*channelOffset + filterID*imgPixels*numCases;				
					float vact = actGrad[filterOffset + ix*widthyz + iy*widthz + z];
					float vimg = sdataImg[(sx + dsx + lobe)*sharedY+(sy + dsy + lobe) + sOffset];
						//input[channelOffset + idx*widthyz + idy*widthz + z];

					int ind_coeff = filterID*conv2 + (dsy + lobe)*conv_size +(dsx + lobe);
					sdataRes[res_off + ind_coeff] += vact*vimg;


				}//dsx
			}//filter

		}//z

		for(int isx = 0; isx < conv_size; isx++)
		for(int isy = 0; isy < conv_size; isy++)
		{
			for(int filterID = 0; filterID <  numFilters; filterID++)
			{
				int ind_coeff = filterID*conv2 + isy*conv_size + isx;
				int ind_ch = ind_coeff + channelInd*numFilters*conv2;
				target[ind_ch][ix*imgSizeX*tagWidth + tagWidth*iy + zoff] = sdataRes[res_off + ind_coeff];
			}
		}

	}//channel

}

//-------------------------------------------------------------
//VectFunc
//-------------------------------------------------------------
#define SCALE_H 1.

template <int sizeV>
__global__ void kVectFuncAct(const float* input, float* const target,
								const uint numPixelsPerGroup, const uint numCases,
								const uint strideInp, const uint strideTag, int numColors, int sizeH) {

// ix, iy == 0 almost always
	const int bd_off =  (blockDim.y*blockIdx.y + threadIdx.y)*strideInp + blockDim.x*blockIdx.x + threadIdx.x;
	const int pix_stride = numPixelsPerGroup*strideInp;
	const int pix_tag_stride = numPixelsPerGroup*strideTag;

    for (uint iy = 0; iy < numPixelsPerGroup; iy += gridDim.y*blockDim.y) 
	{
        for (uint ix = 0; ix < numCases; ix += gridDim.x*blockDim.x)
		{	

			int xy_off = iy*strideInp +	ix + bd_off;

			for (uint color = 0; color < numColors; color ++) {	

				int color_off =  color*pix_stride;
			
				float inpVal[sizeV];//use shared instead?
	#pragma unroll
				for (uint inp_i = 0; inp_i < sizeV; inp_i++) {					

					int voff = color_off*sizeV + inp_i*pix_stride + xy_off;

					float val = input[voff];

					inpVal[inp_i] = val;
				}

				float vmax= 0;
#pragma unroll	
				for (uint out_i = 0; out_i < sizeH; out_i++) {
					int out_par = out_i*sizeV;

					float output = 0;
#pragma unroll			
					for (uint inp_i = 0; inp_i < sizeV; inp_i++)
					{		
						float param = const_area[out_par + inp_i];
						float val = inpVal[inp_i];
						output += param*val;
					}// inp_i

					//suppression filter

					//output = fmaxf(output, 0);
					vmax = fmaxf(output, vmax);
				}//out_i


				for (uint out_i = 0; out_i < sizeH; out_i++) {
					int out_par = out_i*sizeV;

					float output = 0;
#pragma unroll
					for (uint inp_i = 0; inp_i < sizeV; inp_i++)
					{		
						float param = const_area[out_par + inp_i];
						float val = inpVal[inp_i];
						output += param*val;
					}// inp_i

					//suppression filter
					output = fmaxf(output - SCALE_H*(vmax-output), 0);

					int toffset = color_off*sizeH + out_i*pix_tag_stride +  xy_off;
					target[toffset] = output;
				}//out_i

			}//color
        }
    }

}


template <int sizeV>
__global__ void kVectFuncGrad(const float* actGrad, const float* input, float* const target,
								const uint numPixelsPerGroup, const uint numCases,
								const uint strideInp, const uint strideOut,
								int numColors, int sizeH) {

	const int inStep = strideInp*numPixelsPerGroup;
	const int outStep = strideOut*numPixelsPerGroup;

	const int pix_out_stride = numPixelsPerGroup*strideOut;
	const int pix_in_stride = numPixelsPerGroup*strideInp;

	const int btx = blockDim.x*blockIdx.x + threadIdx.x;
	const int bty = blockDim.y*blockIdx.y + threadIdx.y;

	const int bd_off_in =  bty*strideInp + btx;
	const int bd_off_out = bty*strideOut + btx;

//with no N_SUM ix, iy == 0 almost always
    for (uint iy = 0; iy < numPixelsPerGroup; iy += gridDim.y*blockDim.y) {
        for (uint ix = 0; ix < numCases; ix += gridDim.x*blockDim.x) {	

			int xy_off_in = iy*strideInp +	ix + bd_off_in;
			int xy_off_out = iy*strideOut +	ix + bd_off_out;

			for (uint color = 0; color < numColors; color ++) {	//optimize away

				//Offset out_offset;
				//out_offset 
				//<< Index(color) << sizeH << numPixelsPerGroup << Index(iy) << Index(blockDim.y, blockIdx.y) << Index(threadIdx.y)
				//<< strideOut
				//<< Index(ix ) << Index(blockDim.x, blockIdx.x) << Index(threadIdx.x);

				//Offset v_offset;
				//v_offset 
				//<< Index(color) << sizeV << numPixelsPerGroup << Index(iy) << Index(blockDim.y, blockIdx.y) << Index(threadIdx.y)
				//<< strideInp
				//<< Index(ix ) << Index(blockDim.x, blockIdx.x) << Index(threadIdx.x);

				float vres[sizeV];
				memset(vres, 0, sizeof(vres));

				for (uint out_i = 0; out_i < sizeH; out_i++)
				{
					int out_off = color*pix_out_stride*sizeH + out_i*pix_out_stride + xy_off_out;

					float vsum = 0;
					for (uint inp_i = 0; inp_i < sizeV; inp_i++) {	
					int in_off = color*pix_in_stride*sizeV + inp_i*pix_in_stride + xy_off_in;

						vsum += input[in_off]*const_area[out_i*sizeV + inp_i];
					}

					if(vsum > 0)
					{
						float grad_next = actGrad[out_off];

						for (uint inp_i = 0; inp_i < sizeV; inp_i++)
							vres[inp_i] += grad_next*const_area[out_i*sizeV + inp_i];
					}
				}

				for (uint inp_i = 0; inp_i < sizeV; inp_i++)
				{
					int in_off = color*pix_in_stride*sizeV + inp_i*pix_in_stride + xy_off_in;
					target[in_off] = vres[inp_i];
				}

			}//color
		}//ix
	}//iy
}

template <int sizeV>
__global__ void kVectFuncParamWeightGrad(	const float* actGrad, const float* input, float** const target,
											const uint numColors,
											const uint target_size, const uint numPixelsPerGroup, const uint numCases,
											const uint strideInp, const uint strideOut, const uint strideTag, int sizeH)
{
	extern __shared__ float sh_mem[];
	const int xy_off = threadIdx.y*blockDim.x + threadIdx.x;
	const int res_off = xy_off*sizeV*sizeH;
	float* resh = sh_mem + sizeV*blockDim.x*blockDim.y + res_off;
	float* in_store = sh_mem;
	
	memset(resh, 0, sizeV*sizeH*sizeof(float));

	const int btx = blockDim.x*blockIdx.x + threadIdx.x;
	const int bty = blockDim.y*blockIdx.y + threadIdx.y;

	const int bd_off_in =  bty*strideInp + btx;
	const int bd_off_out = bty*strideOut + btx;
	const int bd_off_tag = bty*strideTag + btx;

	const int pix_out_stride = numPixelsPerGroup*strideOut;
	const int pix_in_stride = numPixelsPerGroup*strideInp;
	
	for (uint iy = 0; iy < numPixelsPerGroup; iy += gridDim.y*blockDim.y) {

	  for (uint ix = 0; ix < numCases; ix += gridDim.x*blockDim.x) {	

		int xy_off_in = iy*strideInp +	ix + bd_off_in;
		int xy_off_out = iy*strideOut +	ix + bd_off_out;

		for (uint color = 0; color < numColors; color ++) {	

			float* inp_val = in_store + xy_off*sizeV;
			//float inp_val[sizeV];

				for (uint pin = 0; pin < sizeV; pin++)
				{
					int in_off = color*pix_in_stride*sizeV + pin*pix_in_stride + xy_off_in;
					inp_val[pin] = input[in_off];
				}

				int kmax= 0;
				float vmax = 0;

				for (uint pout = 0; pout < sizeH; pout++)
				{
					float vsum = 0;
#pragma unroll
					for (uint pin = 0; pin < sizeV; pin++)
					{
						vsum += inp_val[pin]*const_area[pout*sizeV + pin];
					}
					if(vsum > vmax)
					{
						vmax = vsum;
						kmax = pout;
					};
				}//pout

				float vres_max[sizeV];
				memset(vres_max, 0, sizeof(vres_max));

				for (uint pout = 0; pout < sizeH; pout++)
				{
					float* vres =  resh + sizeV*pout;

					int out_off = color*pix_out_stride*sizeH + pout*pix_out_stride + xy_off_out;
					float grad_next = actGrad[out_off];

					float output = 0;
#pragma unroll
					for (uint pin = 0; pin < sizeV; pin++)
					{
						output +=  inp_val[pin]*const_area[pout*sizeV + pin];
					}

					output = fmaxf(output - SCALE_H*(vmax-output), 0);

					if(output > 0)
					{
						for (uint pin = 0; pin < sizeV; pin++)
						{
							vres[pin] += grad_next*(1+SCALE_H)*inp_val[pin];
							vres_max[pin] += - SCALE_H*grad_next*inp_val[pin];
						}
					}//vsum
				}//pout
#pragma unroll
				for (uint pin = 0; pin < sizeV; pin++)
				{
					resh[kmax*sizeV + pin] += vres_max[pin];
				}

			}//color
		}//ix
	}//iy
		
	for (uint pout = 0; pout < sizeH; pout++)
#pragma unroll
	for (uint pin = 0; pin < sizeV; pin++)
	{

		target[pout*sizeV+pin][bd_off_tag] = resh[pout*sizeV+pin];
	}

}

//*************************************************************************************
//-------------------------------------------------------------
//API EltwiseMax
//-------------------------------------------------------------

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

//-------------------------------------------------------------
//API EltwiseFunc
//-------------------------------------------------------------



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

//	float sumt = target.sum();
//	printf("sum_tag %f \n", sumt);


	cutilCheckMsg("computeEltwiseFuncGrad: Kernel execution failed");
};

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
//-------------------------------------------------------------
//API MicroConv
//-------------------------------------------------------------

#include "conv_debug.h"
#define SIZE_CONV 3


void computeMicroConvAct(NVMatrix& input, NVMatrix& target, vector<double>& param, int sizeModuleSide, int channels,
						 int imgSize, int imgPixels, int numFilters)
{
	int out_width = input.getNumCols();
	int out_height = input.getNumRows()*numFilters;

    if (target.getNumCols() != out_width || target.getNumRows() != out_height) {
        target.resize(out_height, out_width);
		//printf("**resize out_height %i out_width %i \n",out_height, out_width);
    }

	int numCases = out_width;

	int imgSizeX = imgSize;
	int imgSizeY = imgSize;

	int img_threads_x = 8;
	int img_threads_y = 8;
	int casePerThread = 16;
	int nblocksx = 2;//~number of blocks x
	int case_threads = DIVUP(numCases, nblocksx*casePerThread); 

	int imgBlocksY = DIVUP(imgSizeY,img_threads_x);
	int imgBlocksX = DIVUP(imgSizeX,img_threads_y);

	int lobe = sizeModuleSide/2;


	int sharedX = lobe*2 + img_threads_x;
	int sharedY = lobe*2 + img_threads_y;
	int shared_size = sharedX*sharedY*channels*case_threads*sizeof(float);

	dim3 threads(case_threads, img_threads_x*img_threads_y);
	dim3 blocks = dim3(DIVUP(numCases, threads.x*casePerThread), imgBlocksY*imgBlocksX);


	float temp[CONST_AREA_SIZE];
	assert(param.size() <= CONST_AREA_SIZE);
	memset(temp, 0, sizeof(temp));
	for(int i = 0; i < param.size(); i++)
		temp[i] = (float)param[i];
	cudaMemcpyToSymbol(const_area, temp, sizeof(float)*CONST_AREA_SIZE, 0, cudaMemcpyHostToDevice);

	//printf("blocks.x %i blocks.y %i threads.x %i threads.y %i shared_size %i casePerThread %i\n",
	//	blocks.x, blocks.y, threads.x, threads.y, shared_size, casePerThread);
	//printf("sharedY %i img_threads_x %i img_threads_y %i sizeModuleSide %i imgSizeX %i imgSizeY %i imgPixels %i numFilters %i numCases %i lobe %i\n",
	//	sharedY,img_threads_x,img_threads_y,sizeModuleSide,imgSizeX,imgSizeY, imgPixels,numFilters,numCases,lobe);


	assert(SIZE_CONV == 3);

	//singletonTempMem.allocFloatElement(input.getNumCols()*input.getNumRows());
	//singletonTempMem.allocFloatElement(out_height*out_width);
	//float* tempHostInput = singletonTempMem.getPtr(0);
	//float* tempHostTarget = singletonTempMem.getPtr(1);
	//int deltan = singletonTempMem._start[1]-singletonTempMem._start[0];
	//printf(" size inp %i singletonTempMem._size %i deltan %i \n",
	//	input.getNumCols()*input.getNumRows(),singletonTempMem._size, deltan);
	//cutilSafeCallNoSync( cudaMemcpy(tempHostInput, input.getDevData(), input.getNumCols()*input.getNumRows()*sizeof(float), cudaMemcpyDeviceToHost) );
	//double sum_host =0;
	//debugMicroConvFilterAct((SIZE_CONV-1)/2, SIZE_CONV, temp, tempHostInput, tempHostTarget,
	//									numCases, channels, numFilters,
	//									sharedY, img_threads_x,  img_threads_y, 
	//									imgSizeX, imgSizeY,
	//									imgPixels);
	// sum_host = Sum(tempHostTarget, out_height*out_width);
	//printf(" debugMicroConvFilterAct sum %f \n", sum_host);


	//emuMicroConvFilterAct(threads.x, threads.y, blocks.x, blocks.y,
	//									(SIZE_CONV-1)/2, SIZE_CONV,
	//									temp, tempHostInput, tempHostTarget,
	//									numCases, channels, numFilters, casePerThread,
	//									sharedY, img_threads_x,  img_threads_y, 
	//									imgSizeX, imgSizeY,
	//									imgPixels);

	//sum_host = Sum(tempHostTarget, out_height*out_width);
	//printf(" emuMicroConvFilterAct sum %f \n", sum_host);


	//singletonTempMem.reset();



	kMicroConvFilterAct<(SIZE_CONV-1)/2, SIZE_CONV><<<blocks, threads, shared_size>>>(input.getDevData(), target.getDevData(),
										numCases, channels, numFilters, casePerThread,
										sharedY, img_threads_x,  img_threads_y, 
										imgSizeX, imgSizeY,
										imgPixels);

//debug
	//printf("kMicroConvAct4Channel end \n");

	//float sum = target.sum();
	//printf(" kMicroConvAct4Channel sum %f \n", sum);

	cutilCheckMsg("computeMicroConvAct: Kernel execution failed");

};

void computeMicroConvActGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& target,
							 vector<double>& param, int sizeModuleSide, int channels,
							int imgSize, int imgPixels, int numFilters)
{


    int inp_width = input.getNumCols(); 
    int inp_height = input.getNumRows();

    if (target.getNumCols() != inp_width || target.getNumRows() != inp_height) {
        target.resize(inp_height, inp_width);
    }

	int numCases = inp_width;

	int imgSizeX = imgSize;
	int imgSizeY = imgSize;

	int img_threads_x = 8;
	int img_threads_y = 8;
	int casePerThread = 16;

	int nblocksx = 2;//~number of blocks x

	int case_threads = DIVUP(numCases, nblocksx*casePerThread); 

	int lobe = sizeModuleSide/2;

	int sharedX = lobe*2 + img_threads_x;
	int sharedY = lobe*2 + img_threads_y;
	int shared_size = sharedX*sharedY*numFilters*channels*case_threads*sizeof(float);

	int imgBlocksY = DIVUP(imgSizeY,img_threads_x);
	int imgBlocksX = DIVUP(imgSizeX,img_threads_y);

	dim3 threads(case_threads, img_threads_x*img_threads_y);
	dim3 blocks = dim3(DIVUP(numCases, threads.x*casePerThread), imgBlocksY*imgBlocksX);
	

	float temp[CONST_AREA_SIZE];
	assert(param.size() <= CONST_AREA_SIZE);
	memset(temp, 0, sizeof(temp));
	for(int i = 0; i < param.size(); i++)
		temp[i] = (float)param[i];
	cudaMemcpyToSymbol(const_area, temp, sizeof(float)*CONST_AREA_SIZE, 0, cudaMemcpyHostToDevice);


	printf("blocks.x %i blocks.y %i threads.x %i threads.y %i shared_size %i casePerThread %i\n",
		blocks.x, blocks.y, threads.x, threads.y, shared_size, casePerThread);
	printf("sharedY %i img_threads_x %i img_threads_y %i sizeModuleSide %i imgSizeX %i imgSizeY %i imgPixels %i numFilters %i numCases %i lobe %i\n",
		sharedY,img_threads_x,img_threads_y,sizeModuleSide,imgSizeX,imgSizeY, imgPixels,numFilters,numCases,lobe);


	//singletonTempMem.allocFloatElement(actGrad.getNumCols()*actGrad.getNumRows());
	//singletonTempMem.allocFloatElement(target.getNumCols()*target.getNumRows());
	//float* tempHostInput = singletonTempMem.getPtr(0);
	//float* tempHostTarget = singletonTempMem.getPtr(1);

	//cutilSafeCallNoSync( cudaMemcpy(tempHostInput, actGrad.getDevData(), actGrad.getNumCols()*actGrad.getNumRows()*sizeof(float),
	//	cudaMemcpyDeviceToHost) );

	//double sum_host =0;
	//debugMicroConvActGrad((SIZE_CONV-1)/2, SIZE_CONV, temp, tempHostInput, tempHostTarget,
	//							numCases, channels, numFilters, casePerThread, 
	//							img_threads_x, img_threads_y,
	//							sharedY, sizeModuleSide, lobe,
	//							imgSizeX, imgSizeY,
	//							imgPixels);
	//sum_host = Sum(tempHostTarget, target.getNumCols()*target.getNumRows());
	//printf(" debugMicroConvFilterAct sum %f \n", sum_host);
	//singletonTempMem.reset();

	kMicroConvActGrad<<<blocks, threads, shared_size>>>(actGrad.getDevData(), target.getDevData(),
								numCases, channels, numFilters, casePerThread, 
								img_threads_x, img_threads_y,
								sharedY, sizeModuleSide, lobe,
								imgSizeX, imgSizeY,
								imgPixels);
//	double sum = target.sum();
//	printf(" kMicroConvGrad sum %f \n", sum);
//	printf("kMicroConvGrad end \n");

	cutilCheckMsg("kMicroConvGrad: Kernel execution failed");
}

void computeMicroConvWeightGrad(NVMatrix& actGrad, NVMatrix& input,
								vector<NVMatrix>& tempMatrix, void* arrayPtr,
								vector<double>& param, int sizeModuleSide, int channels,
								int imgSize, int imgPixels, int numFilters)
{

	int numCases = input.getNumCols();

	int imgSizeX = imgSize;
	int imgSizeY = imgSize;

	int img_threads_x = 8;
	int img_threads_y = 8;
	int casePerThread = 16;

	int nblocksx = 2;//~number of blocks x

	int case_threads = DIVUP(numCases, nblocksx*casePerThread); 

	int lobe = sizeModuleSide/2;

	int sharedX = lobe*2 + img_threads_x;
	int sharedY = lobe*2 + img_threads_y;

	int conv_size = (lobe*2 + 1);
	int conv_size2 = conv_size*conv_size;

	int imgBlocksY = DIVUP(imgSizeY,img_threads_x);
	int imgBlocksX = DIVUP(imgSizeX,img_threads_y);

//for optimization can change both block sizes!
	dim3 threads(case_threads, img_threads_x*img_threads_y);
	dim3 blocks = dim3(DIVUP(numCases, threads.x*casePerThread), imgBlocksY*imgBlocksX);

	int sizeSharedBlock = sharedX*sharedY;
	int shared_size = (sizeSharedBlock*threads.x + threads.x*threads.y*numFilters*conv_size2)*sizeof(float);//looped out - case_threads*imgsPerThread;

    int tag_width = DIVUP(input.getNumCols(), casePerThread) ; //could be reduced
    int tag_height = blocks.y*threads.y;//could be reduced
	int tag_size = tag_width*tag_height;

	float* tempMatrixPtr[CONST_AREA_SIZE];
	for(int i =0; i < tempMatrix.size(); i++)
	{
		if (tempMatrix[i].getNumCols() != tag_width || tempMatrix[i].getNumRows() != tag_height) {
			tempMatrix[i].resize(tag_height, tag_width);
			cudaMemset(tempMatrix[i].getDevData(), 0, tag_size*sizeof(float));
		}

		tempMatrixPtr[i] = tempMatrix[i].getDevData();
	}

	cudaMemcpy(arrayPtr, tempMatrixPtr, sizeof(float*)*tempMatrix.size(), cudaMemcpyHostToDevice);

//	printf("kMicroConvWeightGrad *************** \n");
//	printf("tag_width %i tag_height %i shared_size %i  tempMatrix.size() %i conv_size %i casePerThread %i\n",
//		tag_width, tag_height, shared_size, tempMatrix.size(), conv_size, casePerThread);
//
//	printf("blocks.x %i blocks.y %i threads.x %i threads.y %i shared_size %i \n",
//		blocks.x, blocks.y, threads.x, threads.y, shared_size);
//	printf("sharedY %i img_threads_x %i img_threads_y %i sizeModuleSide %i imgSizeX %i imgSizeY %i imgPixels %i numFilters %i numCases %i lobe %i\n",
//		sharedY,img_threads_x,img_threads_y,sizeModuleSide,imgSizeX,imgSizeY, imgPixels,numFilters,numCases,lobe);
//
//
	//const int sizeConv2 = SIZE_CONV*SIZE_CONV;
	//int filterID = 0;
	//int dsy = 0;
	//int dsx = 1;
	//int channelID = 0;
	//int ind_coeff = filterID*sizeConv2 + (dsy + lobe)*SIZE_CONV +(dsx + lobe);
//
//	singletonTempMem.allocFloatElement(actGrad.getNumCols()*actGrad.getNumRows());
//	singletonTempMem.allocFloatElement(input.getNumCols()*input.getNumRows());
//	singletonTempMem.allocFloatElement(tag_height*tag_width);
//	int out_width = input.getNumCols();
//	int out_height = input.getNumRows()*numFilters;
//	singletonTempMem.allocFloatElement(out_width*out_height);
//
//	float* tempHostAct = singletonTempMem.getPtr(0);
//	float* tempHostInp = singletonTempMem.getPtr(1);
//	float* tempHostTag = singletonTempMem.getPtr(2);
//	float* tempHostTagA = singletonTempMem.getPtr(3);
//
//	cudaMemcpy(tempHostAct, actGrad.getDevData(), actGrad.getNumCols()*actGrad.getNumRows()*sizeof(float),
//		cudaMemcpyDeviceToHost);
//
//	cudaMemcpy(tempHostInp, input.getDevData(), input.getNumCols()*input.getNumRows()*sizeof(float),
//		cudaMemcpyDeviceToHost);
//	//memset(tempHostTagA, 0, tag_height*tag_width*sizeof(float));
//	memset(tempHostTag, 0, tag_height*tag_width*sizeof(float));
//
//	double sum_a = Sum(tempHostAct, actGrad.getNumCols()*actGrad.getNumRows());
//	double sum_i = Sum(tempHostInp, input.getNumCols()*input.getNumRows());
//	 printf(" sum_a %f sum_i %f \n", sum_a, sum_i);
//
//	float temp[CONST_AREA_SIZE];
//	assert(param.size() <= CONST_AREA_SIZE);
//	memset(temp, 0, sizeof(temp));
//	for(int i = 0; i < param.size(); i++)
//		temp[i] = (float)param[i];
//
//	
//	debugMicroConvLinApprox((SIZE_CONV-1)/2, SIZE_CONV, temp, tempHostInp, tempHostAct, tempHostTagA,
//										numCases, channels, numFilters,
//										sharedY, img_threads_x,  img_threads_y, 
//										imgSizeX, imgSizeY,
//										imgPixels);
//	double sum_host0 = Sum(tempHostTagA, out_height*out_width);
//	printf(" debugMicroConvFilterAct sum0 %f \n", sum_host0);
//	double delta = 1e-3;
//	temp[ind_coeff] += delta;
//
//	debugMicroConvLinApprox((SIZE_CONV-1)/2, SIZE_CONV, temp, tempHostInp, tempHostAct, tempHostTagA,
//										numCases, channels, numFilters,
//										sharedY, img_threads_x,  img_threads_y, 
//										imgSizeX, imgSizeY,
//										imgPixels);
//	double sum_host1 = Sum(tempHostTagA, out_height*out_width);
//	printf(" debugMicroConvFilterAct sum1 %f \n", sum_host1);
//
//	printf(" debugMicroConv grad %f \n", (sum_host1-sum_host0)/delta);
//
//
//memset(tempHostTag, 0, tag_height*tag_width*sizeof(float));
//  debugMicroConvWeightGrad(lobe, SIZE_CONV, dsx, dsy, filterID, channelID, tempHostAct, tempHostInp, tempHostTag,
//								tag_size, numCases,
//								channels, numFilters, 
//								img_threads_x, img_threads_y, sharedY,
//								lobe, sizeModuleSide, sizeSharedBlock,
//								imgSizeX, imgSizeY, imgPixels);
//
//  double sum_host = Sum(tempHostTag, tag_height*tag_width);
//  printf(" debugMicroConvWeightGrad sum %f \n", sum_host);
//
//memset(tempHostTag, 0, tag_height*tag_width*sizeof(float));

  //emuMicroConvWeightGrad(threads.x, threads.y, blocks.x, blocks.y,
		//					lobe, SIZE_CONV, dsx, dsy, filterID, channelID, tempHostAct, tempHostInp, tempHostTag,
		//						tag_size, numCases, casePerThread, tag_width,
		//						channels, numFilters, 
		//						img_threads_x, img_threads_y, sharedY,
		//						sizeSharedBlock,
		//						imgSizeX, imgSizeY, imgPixels);

  // double sum_host_emu = Sum(tempHostTag, tag_height*tag_width);
  //printf(" emuMicroConvWeightGrad sum %f \n", sum_host_emu);


	kMicroConvWeightGrad<SIZE_CONV/2><<<blocks, threads, shared_size>>>(actGrad.getDevData(), input.getDevData(), (float**)arrayPtr,
								tag_size, numCases, casePerThread, tag_width,
								channels, numFilters, 
								img_threads_x, img_threads_y,
								imgSizeX, imgSizeY, imgPixels);

//	double sum_ag = actGrad.sum();
//	double sum_ig = input.sum();
//double sum = tempMatrix[ind_coeff].sum();
//printf(" kMicroConvWeightGrad sum %f  \n", sum);
	//printf(" kMicroConvWeightGrad sum %f sum_ag %f sum_ig %f \n", sum, sum_ag, sum_ig);

////debug
//	printf("kMicroConvWeightGrad end \n");

	cutilCheckMsg("kMicroConvWeightGrad: Kernel execution failed");
}

//-------------------------------------------------------------
//API VectFunc
//-------------------------------------------------------------

void computeVectFuncAct(NVMatrix& input, NVMatrix& target, vector<double>& param, int sizeV, int sizeH, int channels)
{
//printf("\n kVectFuncAct start*** \n");

	assert(sizeV <= 4 || sizeV == 6 || sizeV == 8 || sizeV == 12 || sizeV == 16);

    int inp_width = input.getNumCols(); 
    int inp_height = input.getNumRows();

	int out_width = inp_width;
	int out_height = (inp_height*sizeH)/sizeV;

	int numCases = out_width;
	int numPixelsPerGroup = inp_height/channels;

	int numColors = channels/sizeV;


    if (target.getNumCols() != out_width || target.getNumRows() != out_height) {
//		printf("**resize out_height %i out_width %i \n",out_height, out_width);
        target.resize(out_height, out_width);
    }
	float temp[CONST_AREA_SIZE];
	assert(param.size() <= CONST_AREA_SIZE);
	memset(temp, 0, sizeof(temp));
	for(int i = 0; i < param.size(); i++)
		temp[i] = (float)param[i];
	cudaMemcpyToSymbol(const_area, temp, sizeof(float)*CONST_AREA_SIZE, 0, cudaMemcpyHostToDevice);

    dim3 threads(min(ELTWISE_THREADS_X, inp_width), ELTWISE_THREADS_Y);

    dim3 blocks(std::min(NUM_BLOCKS_MAX, (int)DIVUP(inp_width, threads.x)),
                std::min(NUM_BLOCKS_MAX, DIVUP(numPixelsPerGroup, ELTWISE_THREADS_Y)));

	for(int i = 0; i < param.size()/2; i++)
	{
		printf("param %f %f \n",  param[2*i], param[2*i]);
	}

//	float sumi = input.sum();
//	printf("sumi %f \n",  sumi);
//	printf("blocks.x %i blocks.y %i threads.x %i threads.y %i numColors %i \n",blocks.x, blocks.y, threads.x, threads.y, numColors);
//	printf("inp_height %i numPixelsPerGroup %i out_width %i out_height %i sizeV %i \n",inp_height, numPixelsPerGroup,out_width,out_height,sizeV);
//	printf("sizeV %i sizeH %i strides %i %i \n", sizeV, sizeH, input.getStride(), target.getStride());
////debug
//	cudaMemset(target.getDevData(), 0, out_height*out_width*sizeof(float));
//	
//	singletonTempMem.allocFloatElement(input.getNumCols()*input.getNumRows());
//	singletonTempMem.allocFloatElement(out_height*out_width);
//	float* tempHostInput = singletonTempMem.getPtr(0);
//	float* tempHostTarget = singletonTempMem.getPtr(1);
//	cudaMemcpy(tempHostInput, input.getDevData(), input.getNumCols()*input.getNumRows()*sizeof(float), cudaMemcpyDeviceToHost);
//	cudaDeviceSynchronize();
//
//	double sum_inp = Sum(tempHostInput, input.getNumCols()*input.getNumRows());
//	printf("sum_inp %f \n",  sum_inp);
//
//	double sum_host =0;
//	memset(tempHostTarget, 0, out_height*out_width*sizeof(float));
//	debugVectFuncAct(sizeV, temp, tempHostInput, tempHostTarget,
//								numPixelsPerGroup, numCases, input.getStride(), target.getStride(), numColors, sizeH);
//
//	sum_host = Sum(tempHostTarget, out_height*out_width);
//
//	printf(" debugVectFuncAct sum %f \n", sum_host);
//
//	singletonTempMem.reset();

#define ELT_ACT(SIZE_ARR) \
	if(sizeV == SIZE_ARR){\
	cudaFuncSetCacheConfig(kVectFuncAct<SIZE_ARR>, cudaFuncCachePreferL1);\
	kVectFuncAct<SIZE_ARR><<<blocks, threads>>>(input.getDevData(),\
	target.getDevData(), numPixelsPerGroup, numCases, input.getStride(), target.getStride(), numColors, sizeH);};
	ELT_ACT(1)
	ELT_ACT(2)
	ELT_ACT(3)
	ELT_ACT(4)
	ELT_ACT(6)
	ELT_ACT(8)
	ELT_ACT(12)
	ELT_ACT(16)
#undef ELT_ACT

//	float sumt = target.sum();
//	printf("kVectFuncAct sumt %f \n",  sumt);

	//printf("kVectFuncAct end \n");
	cutilCheckMsg("kVectFuncAct: Kernel execution failed");
}


void computeVectFuncGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& target,
								 vector<double>& param,  int sizeV, int sizeH, int channels)
{


	assert(sizeV <= 4 || sizeV == 6 || sizeV == 8 || sizeV == 12 || sizeV == 16);

    int inp_width = input.getNumCols(); 
    int inp_height = input.getNumRows();

    if (target.getNumCols() != inp_width || target.getNumRows() != inp_height) {
        target.resize(inp_height, inp_width);
    }


	int out_width = inp_width;
	int out_height = (inp_height*sizeH)/sizeV;

	int numCases = out_width;
	int numPixelsPerGroup = inp_height/channels;

	int numColors = channels/sizeV;

	float temp[CONST_AREA_SIZE];
	assert(param.size() <= CONST_AREA_SIZE);
	memset(temp, 0, sizeof(temp));
	for(int i = 0; i < param.size(); i++)
		temp[i] = (float)param[i];
	cudaMemcpyToSymbol(const_area, temp, sizeof(float)*CONST_AREA_SIZE, 0, cudaMemcpyHostToDevice);

    dim3 threads(min(ELTWISE_THREADS_X, inp_width), ELTWISE_THREADS_Y);

    dim3 blocks(std::min(NUM_BLOCKS_MAX, (int)DIVUP(inp_width, threads.x)),
                std::min(NUM_BLOCKS_MAX, DIVUP(numPixelsPerGroup, ELTWISE_THREADS_Y)));

	printf("kVectFuncGrad start ************************\n");
	printf("blocks.x %i blocks.y %i threads.x %i threads.y %i \n",
		blocks.x, blocks.y, threads.x, threads.y);
	printf("numPixelsPerGroup %i numCases %i numColors %i out_width %i out_height %i\n",
		numPixelsPerGroup, numCases, numColors, out_width, out_height);

	singletonTempMem.allocFloatElement(input.getNumCols()*input.getNumRows());
	singletonTempMem.allocFloatElement(inp_height*inp_width);
	singletonTempMem.allocFloatElement(actGrad.getNumCols()*actGrad.getNumRows());
	singletonTempMem.allocFloatElement(inp_height*inp_width);
	float* tempHostInput = singletonTempMem.getPtr(0);
	float* tempHostTarget = singletonTempMem.getPtr(1);
	float* tempHostActGrad = singletonTempMem.getPtr(2);
	float* tempHostTarget1 = singletonTempMem.getPtr(1);
	cudaMemcpy(tempHostInput, input.getDevData(), input.getNumCols()*input.getNumRows()*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(tempHostActGrad, actGrad.getDevData(), actGrad.getNumCols()*actGrad.getNumRows()*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	debugVectFuncGrad(sizeV, temp, tempHostActGrad,
				tempHostInput, tempHostTarget, tempHostTarget1, numPixelsPerGroup, numCases,
				input.getStride(), actGrad.getStride(), numColors, sizeH);

	double sum_host = Sum(tempHostTarget, inp_height*inp_width);
	double sum_host1 = Sum(tempHostTarget1, inp_height*inp_width);
	printf(" debugVectFuncAct sum %f sum1 %f \n", sum_host, sum_host1);
	singletonTempMem.reset();

#define ELT_GRAD(SIZE_ARR) \
		if(sizeV == SIZE_ARR){\
			cudaFuncSetCacheConfig(kVectFuncGrad<SIZE_ARR>, cudaFuncCachePreferL1);\
			kVectFuncGrad<SIZE_ARR><<<blocks, threads>>>(actGrad.getDevData(),\
				input.getDevData(), target.getDevData(), numPixelsPerGroup, numCases,\
				input.getStride(), actGrad.getStride(), numColors, sizeH);};
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
	printf("kVectFuncGrad sum_tag %f \n", sumt);


	cutilCheckMsg("kVectFuncGrad: Kernel execution failed");

};

void computeVectFuncWeightGrad(NVMatrix& actGrad, NVMatrix& input,
								vector<NVMatrix>& tempMatrix,
								void* arrayPtr,
								vector<double>& param,  int sizeV, int sizeH, int channels)
{
	assert(sizeV <= 4 || sizeV == 6 || sizeV == 8 || sizeV == 12 || sizeV == 16);

    int inp_width = input.getNumCols(); 
    int inp_height = input.getNumRows();

	int out_width = inp_width;
	int out_height = (inp_height*sizeH)/sizeV;

	int numCases = out_width;
	int numPixelsPerGroup = inp_height/channels;

	int numColors = channels/sizeV;



#define N_SUM 1
    dim3 threads(min(ELTWISE_THREADS_X, inp_width), ELTWISE_THREADS_Y);
    dim3 blocks(std::min(NUM_BLOCKS_MAX, (int)DIVUP(inp_width, threads.x)),//reduce
                std::min(NUM_BLOCKS_MAX, (int)DIVUP(numPixelsPerGroup/N_SUM, ELTWISE_THREADS_Y)));
#undef N_SUM

	int shared_size = sizeV*(sizeH+1)*threads.x*threads.y*sizeof(float);

    int tag_width = blocks.x*threads.x; //could be reduced
    int tag_height = blocks.y*threads.y;//could be reduced
	int tag_size = tag_width*tag_height;

	float temp[CONST_AREA_SIZE];
	assert(param.size() <= CONST_AREA_SIZE);
	memset(temp, 0, sizeof(temp));
	for(int i = 0; i < param.size(); i++)
		temp[i] = (float)param[i];
	cudaMemcpyToSymbol(const_area, temp, sizeof(float)*CONST_AREA_SIZE, 0, cudaMemcpyHostToDevice);

	float* tempMatrixPtr[CONST_AREA_SIZE];
	for(int i =0; i < tempMatrix.size(); i++)
	{
		if (tempMatrix[i].getNumCols() != tag_width || tempMatrix[i].getNumRows() != tag_height) {
			tempMatrix[i].resize(tag_height, tag_width);
		}
		tempMatrixPtr[i] = tempMatrix[i].getDevData();
	}

	cudaMemcpy(arrayPtr, tempMatrixPtr, sizeof(float*)*tempMatrix.size(), cudaMemcpyHostToDevice);

	//for(int i =0; i < tempMatrix.size(); i++)
	//{
	//	cudaMemset(tempMatrix[i].getDevData(), 0, tag_size*sizeof(float));
	//}
//----------
	//printf("kVectFuncParamWeightGrad start ************************\n");

	//printf("blocks.x %i blocks.y %i threads.x %i threads.y %i shared_size %i \n",
	//	blocks.x, blocks.y, threads.x, threads.y, shared_size);
	//printf("numPixelsPerGroup %i numCases %i numColors %i out_width %i out_height %i\n",
	//	numPixelsPerGroup, numCases, numColors, out_width, out_height);

	////float sumi = input.sum();
	////printf("sumi %f \n",  sumi);

	//printf( "tempMatrix.size() %i tag_width %i tag_height %i actGrad %i %i tempMatrix[0].getStride() %i \n",
	//		tempMatrix.size(), tag_width, tag_height, actGrad.getNumCols(), actGrad.getNumRows(), tempMatrix[0].getStride());
	//
	//singletonTempMem.allocFloatElement(input.getNumCols()*input.getNumRows());
	//singletonTempMem.allocFloatElement(max(tag_height*tag_width, out_height*out_width));
	//singletonTempMem.allocFloatElement(actGrad.getNumCols()*actGrad.getNumRows());
	//float* tempHostInput = singletonTempMem.getPtr(0);
	//float* tempHostTarget = singletonTempMem.getPtr(1);
	//float* tempHostActGrad = singletonTempMem.getPtr(2);
	//cudaMemcpy(tempHostInput, input.getDevData(), input.getNumCols()*input.getNumRows()*sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(tempHostActGrad, actGrad.getDevData(), actGrad.getNumCols()*actGrad.getNumRows()*sizeof(float), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();


	//debugVectFuncParamWeightGrad(sizeV,  temp, blocks.y, threads.y, blocks.x, threads.x, 
	//			tempHostActGrad,  tempHostInput, tempHostTarget, numColors, tag_size, numPixelsPerGroup, numCases,
	//			input.getStride(), actGrad.getStride(), tempMatrix[0].getStride(), sizeH);


	//double sum_host = Sum(tempHostTarget, tag_height*tag_width);
	//	double sum_act = Sum(tempHostActGrad, actGrad.getNumCols()*actGrad.getNumRows());
	//singletonTempMem.reset();

	//float suma = actGrad.sum();
	//printf("debugVectFuncParamWeightGrad******* sum_host %f sum_act %f suma %f\n", sum_host, sum_act, suma);


	//debugVectFuncLinApprox(sizeV, temp, tempHostInput,
	//							tempHostActGrad, tempHostTarget,
	//							numPixelsPerGroup, numCases,
	//							input.getStride(), tempMatrix[0].getStride(), numColors, sizeH);
	//float delta = 1e-4;
	//float sumLA0 =  Sum(tempHostTarget, out_height*out_width);
	//temp[1] += delta;
	//debugVectFuncLinApprox(sizeV, temp, tempHostInput,
	//							tempHostActGrad, tempHostTarget,
	//							numPixelsPerGroup, numCases,
	//							input.getStride(), tempMatrix[0].getStride(), numColors, sizeH);

	//float sumLA1 =  Sum(tempHostTarget, out_height*out_width);

	//printf("debugVectFunc * s0 %f s1 %f deriv %f\n", sumLA0, sumLA1, (sumLA1-sumLA0)/delta);

//----------


#define ELT_GRAD(SIZE_ARR) \
		if(sizeV == SIZE_ARR){\
			cudaFuncSetCacheConfig(kVectFuncParamWeightGrad<SIZE_ARR>, cudaFuncCachePreferL1);\
			kVectFuncParamWeightGrad<SIZE_ARR><<<blocks, threads, shared_size>>>(actGrad.getDevData(),\
				input.getDevData(), (float**)arrayPtr, numColors, tag_size, numPixelsPerGroup, numCases,\
				input.getStride(), actGrad.getStride(), tempMatrix[0].getStride(), sizeH);};
		ELT_GRAD(1)
		ELT_GRAD(2)
		ELT_GRAD(3)
		ELT_GRAD(4)
		ELT_GRAD(6)
		ELT_GRAD(8)
		ELT_GRAD(12)
		ELT_GRAD(16)
#undef ELT_GRAD

	float sumt = tempMatrix[1].sum();
	printf("kVectFuncParamWeightGrad sum_tag %f \n", sumt);

		cutilCheckMsg("kVectFuncParamWeightGrad: Kernel execution failed");
}