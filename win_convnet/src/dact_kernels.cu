
#include <assert.h>

#include <layer_kernels.cuh>

#include "tt.h"

#define CONST_AREA_SIZE 256
extern __device__ __constant__ float const_area[CONST_AREA_SIZE];


__global__ void kDShrinkWeightGrad(const float* actGrad, const float* input,
								   const float* pos_bias, const float* neg_bias,
								   float* const target_pos, float* const target_neg,
                                   const uint height, const uint width, uint stride) {
    const uint idxX = blockIdx.x * ELTWISE_THREADS_X + threadIdx.x;
    const uint idxY = blockIdx.y * ELTWISE_THREADS_Y + threadIdx.y;

    for (uint y = idxY; y < height; y += gridDim.y * ELTWISE_THREADS_Y) {
        for (uint x = idxX; x < width; x += gridDim.x * ELTWISE_THREADS_X) {
			uint ind = y * stride + x;
			float inp = input[ind];
			float v_pos = fmaxf(inp + pos_bias[ind], 0);
			float v_neg = fminf(inp + neg_bias[ind], 0);
			float agrad = actGrad[ind];
			float tpos = 0;
			float tneg = 0;

			if(v_pos > -v_neg)
			{
				tpos += agrad;
			}

			if(-v_neg > v_pos)
			{
				tneg += agrad;
			}

			target_pos[ind] = tpos;
			target_pos[ind] = tneg;
        }
    }
}

__global__ void kDShrinkGrad(const float* actGrad, const float* input,
							const float* pos_bias, const float* neg_bias,
							float* const target,
                            const uint height, const uint width, uint stride) {
    const uint idxX = blockIdx.x * ELTWISE_THREADS_X + threadIdx.x;
    const uint idxY = blockIdx.y * ELTWISE_THREADS_Y + threadIdx.y;

    for (uint y = idxY; y < height; y += gridDim.y * ELTWISE_THREADS_Y) {
        for (uint x = idxX; x < width; x += gridDim.x * ELTWISE_THREADS_X) {
			uint ind = y * stride + x;
			float b_pos = pos_bias[ind];
			float b_neg = neg_bias[ind];
			float inp = input[ind];
			float agrad = actGrad[ind];
			
			float grad;
			if(b_pos > b_neg)
			{
				grad = ((inp > b_pos)||(inp < b_neg))*agrad;
			}
			else
			{
				grad = agrad;
			}

			target[ind] = grad;
        }
    }
}

//*************************************************************************************


void dshrinkWeightGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& pos_bias, NVMatrix& neg_bias,
					   NVMatrix& target_pos, NVMatrix& target_neg){
    assert(actGrad.isSameDims(input));
	assert(actGrad.isSameDims(pos_bias));
	assert(actGrad.isSameDims(neg_bias));
	assert(actGrad.isSameDims(target_pos));
	assert(actGrad.isSameDims(target_neg));

    assert(actGrad.isTrans() == input.isTrans());
	assert(actGrad.isTrans() == pos_bias.isTrans());
	assert(actGrad.isTrans() == neg_bias.isTrans());
	assert(actGrad.isTrans() == target_pos.isTrans());
	assert(actGrad.isTrans() == target_neg.isTrans());



    int height = actGrad.getFollowingDim(), width = actGrad.getLeadingDim();
    dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
                std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
    dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);

	kDShrinkWeightGrad<<<blocks, threads>>>(actGrad.getDevData(), input.getDevData(), pos_bias.getDevData(),
											neg_bias.getDevData(), target_pos.getDevData(), target_neg.getDevData(),
											height, width, actGrad.getStride());

    cutilCheckMsg("dshrinkWeightGrad: Kernel execution failed");
}

void dshrinkGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& pos_bias, NVMatrix& neg_bias,
					   NVMatrix& target){
    assert(actGrad.isSameDims(input));
	assert(actGrad.isSameDims(pos_bias));
	assert(actGrad.isSameDims(neg_bias));
	assert(actGrad.isSameDims(target));

    assert(actGrad.isTrans() == input.isTrans());
	assert(actGrad.isTrans() == pos_bias.isTrans());
	assert(actGrad.isTrans() == neg_bias.isTrans());
	assert(actGrad.isTrans() == target.isTrans());

    int height = actGrad.getFollowingDim(), width = actGrad.getLeadingDim();
    dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
                std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
    dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);

	kDShrinkGrad<<<blocks, threads>>>(actGrad.getDevData(), input.getDevData(), pos_bias.getDevData(),
											neg_bias.getDevData(), target.getDevData(),
											height, width, actGrad.getStride());

    cutilCheckMsg("dshrinkGrad: Kernel execution failed");
}
