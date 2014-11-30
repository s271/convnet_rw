
#include <assert.h>

#include <layer_kernels.cuh>

#include "tt.h"

#define CONST_AREA_SIZE 256
__device__ __constant__ float const_area[CONST_AREA_SIZE];

__device__ inline float Switch(float s, float C) 
{
	return fminf(fmaxf(s*C, -.5), .5);
	//return (s>0)*.5 - (s<0)*.5;
}

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

template <int sizeArr>
__global__ void kEltwiseDFuncAct(const float* input, float* const target,
								const uint imgInPixels, const uint numCases,
								const uint strideInp, const uint strideTag,
								const int numPixelsPerChannel,
								const float Csw, const float Bsw,
								const uint sizeIn, const uint sizeOut) {

	const int numPixelsPerGroup = imgInPixels/sizeIn;

//    dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(out_width, ELTWISE_THREADS_X)),
//                std::min(NUM_BLOCKS_MAX, DIVUP(numPixelsPerGroup, ELTWISE_THREADS_Y)));

	const uint idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const uint idxY = blockIdx.y * blockDim.y + threadIdx.y;
#ifdef MIX_F
	const int pixelChannelID = idxY%numPixelsPerChannel;
#endif
	const int sw_len = sizeIn*ELWISE_DFUNC_SEC;

// ix, iy == 0 almost always
   for (uint y = idxY; y < numPixelsPerGroup; y += gridDim.y*blockDim.y) {
#ifdef MIX_F		
		const int hiID = y/numPixelsPerChannel;
#endif
        for (uint x = idxX; x < numCases; x += gridDim.x*blockDim.x) {	
			
			float inpVal[sizeArr];//use shared instead?
			float v_sw =0;

			for (uint inp_i = 0; inp_i < sizeIn; inp_i++) {	

#ifdef MIX_F		
				int inp_off = hiID*sizeIn*numPixelsPerChannel*strideInp
					+ inp_i*numPixelsPerChannel*strideInp + pixelChannelID*strideInp + x;
#else
				int inp_off = inp_i*numPixelsPerGroup*strideInp +  y*strideInp + x;
#endif

				float val = input[inp_off];
				inpVal[inp_i] = val;
				v_sw += val;
			}
			float Sw = Switch(v_sw + Bsw,  Csw);
			//float v_sw = Median3(inpVal[0],inpVal[1],inpVal[2]);
		
			for (uint out_i = 0; out_i < sizeOut; out_i++) {
				int out_par = out_i*EL_SWITCH*sizeIn*ELWISE_DFUNC_SEC;

				float sum = 0;
		
				for (uint inp_i = 0; inp_i < sizeIn; inp_i++)
				{	
					float val = inpVal[inp_i];
					//float Sw = Switch(v_sw + 4*val + Bsw,  Csw);

					float param = const_area[out_par + inp_i];
					float paramMP = const_area[out_par + sizeIn + inp_i];
					float paramMN = const_area[out_par + 2*sizeIn + inp_i];
					float paramBP = const_area[out_par + 3*sizeIn + inp_i];
					float paramBN = const_area[out_par + 4*sizeIn + inp_i];
					float output = param*val + paramMP*fmax(val+paramBP, 0) + paramMN*fmin(val+paramBN, 0);

					float param_1 = const_area[out_par + inp_i+sw_len];
					float paramMP_1 = const_area[out_par + sizeIn + inp_i+sw_len];
					float paramMN_1 = const_area[out_par + 2*sizeIn + inp_i+sw_len];
					float paramBP_1 = const_area[out_par + 3*sizeIn + inp_i+sw_len];
					float paramBN_1 = const_area[out_par + 4*sizeIn + inp_i+sw_len];
					float output_1 = param_1*val + paramMP_1*fmax(val+paramBP_1, 0)  + paramMN_1*fmin(val+paramBN_1, 0);;

					sum += Sw*(output - output_1) + .5*(output + output_1);

				}// inp_i
				
				int tag_off = out_i*numPixelsPerGroup*strideTag +  y*strideTag + x;

				target[tag_off] = sum;
			}//out_i
        }
    }

}

//not optimised, rebuild it
template <int B_X, int B_Y, int sizeOut, int sizeIn>
__global__ void kEltwiseDFuncParamWeightGrad(float* actGrad, float* input, float** target,
								const uint imgInPixels, const uint numCases,
								const uint stride, const uint strideTag,
								const uint numPixelsPerChannel,
								const float Csw, const float Bsw)
{
	const int numPixelsPerGroup = imgInPixels/sizeIn;
	const int groupStride  = numPixelsPerGroup*stride;
	const int sw_len = sizeIn*ELWISE_DFUNC_SEC;
	const int tagOffset = (threadIdx.x + blockIdx.x*blockDim.x) +  (threadIdx.y + blockIdx.y*blockDim.y)*strideTag;

    const uint idxX = blockIdx.x * B_X + threadIdx.x;
    const uint idxY = blockIdx.y * B_Y + threadIdx.y;

	for(int pout = 0; pout < sizeOut; pout++)
	{

		for (uint y = idxY; y < numPixelsPerGroup; y += gridDim.y * B_Y) {
#ifdef MIX_F
			const int hiID = y/numPixelsPerChannel;
			const int pixelChannelID = idxY%numPixelsPerChannel;
#endif
			for (uint x = idxX; x < numCases; x += gridDim.x * B_X) {
				int offset_act = y * stride + x;

#ifdef MIX_F
#define stride_in stride
#else
#define stride_in groupStride
#endif
				float InArr[sizeIn];

				float v_sw = 0;
				for(int pin = 0; pin < sizeIn; pin++)
				{
#ifdef MIX_F
					int offset_in = hiID*sizeIn*numPixelsPerChannel*stride
						+ pin*numPixelsPerChannel*stride + pixelChannelID*stride + x;
#else
					int offset_in = offset_act + pin*groupStride;
#endif
					float val = input[offset_in];
					InArr[pin] = val;
					v_sw += val;
				}

				float Sw = Switch(v_sw+ Bsw, Csw);

				//float v_sw = Median3(InArr[0],InArr[1],InArr[2]);

				float grad_next = actGrad[offset_act + pout*groupStride];

				int out_par = pout*EL_SWITCH*sizeIn*ELWISE_DFUNC_SEC;

				for(int pin = 0; pin < sizeIn; pin++)
				{
					float in_val = InArr[pin];
					int out_ind = out_par + pin;

					//float Sw = Switch(v_sw + 4*in_val + Bsw, Csw);

					float val_p_0 = in_val + const_area[out_ind + 3*sizeIn];
					float val_n_0 = in_val + const_area[out_ind + 4*sizeIn];
					target[out_ind][tagOffset] += (.5+Sw)*grad_next*in_val;
					target[out_ind + sizeIn][tagOffset] += (.5+Sw)*grad_next*(val_p_0 > 0)*in_val;
					target[out_ind + 2*sizeIn][tagOffset] += (.5+Sw)*grad_next*(val_n_0 < 0)*in_val;
					target[out_ind + 3*sizeIn][tagOffset]  += 
						(.5+Sw)*grad_next*const_area[out_ind + sizeIn]*(val_p_0 > 0);
					target[out_ind + 4*sizeIn][tagOffset]  += 
						(.5+Sw)*grad_next*const_area[out_ind + 2*sizeIn]*(val_n_0 < 0);

					int out_ind_sw = out_ind + sw_len;

					float val_p_1 = in_val + const_area[out_ind_sw + 3*sizeIn];
					float val_n_1 = in_val + const_area[out_ind_sw + 4*sizeIn];
					target[out_ind_sw][tagOffset] += (.5-Sw)*grad_next*in_val;
					target[out_ind_sw + sizeIn][tagOffset] += (.5-Sw)*grad_next*(val_p_1 > 0)*in_val;
					target[out_ind_sw + 2*sizeIn][tagOffset] += (.5-Sw)*grad_next*(val_n_1 < 0)*in_val;
					target[out_ind_sw + 3*sizeIn][tagOffset] +=
						(.5-Sw)*grad_next*const_area[out_ind_sw + sizeIn]*(val_p_1 > 0);
					target[out_ind_sw + 4*sizeIn][tagOffset] +=
						(.5-Sw)*grad_next*const_area[out_ind_sw + 2*sizeIn]*grad_next*(val_n_1 < 0);
				}
			}
		}


	}
}

template <int sizeArr>
__global__ void kEltwiseDFuncGrad(const float* actGrad, const float* input, float* const target,
								const uint imgInPixels, const uint numCases,
								const uint strideInp, const uint strideOut,
								const int numPixelsPerChannel,
								const float Csw, const float InvCsw, const float Bsw,
								const uint sizeIn, const uint sizeOut) {


	const int numPixelsPerGroup = imgInPixels/sizeIn;	
	const int outStep = strideOut*numPixelsPerGroup;

	const uint idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const uint idxY = blockIdx.y * blockDim.y + threadIdx.y;
#ifdef MIX_F
	const int pixelChannelID = idxY%numPixelsPerChannel;
#endif
	const int sw_len = sizeIn*ELWISE_DFUNC_SEC;


//with no N_SUM ix, iy == 0 almost always
    for (uint y = idxY; y < numPixelsPerGroup; y += gridDim.y*blockDim.y) {
#ifdef MIX_F
		const int hiID = y/numPixelsPerChannel;
#endif
        for (uint x = idxX; x < numCases; x += gridDim.x*blockDim.x) {	

			float grad_next[sizeArr];
			int act_off = y*strideInp + x;
#ifdef MIX_F
			int inp_off = hiID*sizeIn*numPixelsPerChannel*strideInp
					 + pixelChannelID*strideInp + x;
#define strideInpStep	strideInp*numPixelsPerChannel		
#else
#define inp_off act_off
#define strideInpStep	strideInp*numPixelsPerGroup	
#endif

			for (uint out_i = 0; out_i < sizeOut; out_i++)
			{
				grad_next[out_i] = actGrad[act_off + outStep*out_i];
			}//out_i

//debug
			float inpArr[3];
			float v_sw =0;
			for (uint inp_i = 0; inp_i < sizeIn; inp_i++)
			{
				float val = input[inp_off + inp_i*strideInpStep];
				inpArr[inp_i] = val;
				v_sw += val;
			}
			//float v_sw = Median3(inpArr[0],inpArr[1],inpArr[2]);
			float Sw = Switch(v_sw + Bsw, Csw);
			

			for (uint inp_i = 0; inp_i < sizeIn; inp_i++) {	

				float val = inpArr[inp_i];

				//float Sw = Switch(v_sw + 4*val + Bsw, Csw);
								
				float sum_grad = 0;
				
				for (uint out_i = 0; out_i < sizeOut; out_i++)	
				{

					int out_par = out_i*EL_SWITCH*sizeIn*ELWISE_DFUNC_SEC;
					float vsignp_0 = (val + const_area[out_par + 3*sizeIn + inp_i] > 0);
					float vsignn_0 = (val + const_area[out_par + 4*sizeIn + inp_i] < 0);
					float c_0 = vsignp_0*const_area[out_par + sizeIn + inp_i]
					+  vsignn_0*const_area[out_par + 2*sizeIn + inp_i]
					+ const_area[out_par + inp_i];

					float vsignp_1 = (val + const_area[out_par + sw_len + 3*sizeIn + inp_i] > 0);
					float vsignn_1 = (val + const_area[out_par + sw_len + 4*sizeIn + inp_i] < 0);

					float c_1 = vsignp_1*const_area[out_par + sw_len + sizeIn + inp_i] 
					+ vsignn_1*const_area[out_par + sw_len + 2*sizeIn + inp_i] 
					+ const_area[out_par + sw_len + inp_i];


					sum_grad += grad_next[out_i]*((Sw+.5)*c_0	+ (.5-Sw)*c_1);
						//+(v_sw+Bsw > -InvCsw && v_sw+Bsw < InvCsw)*Csw*(c_0-c_1));
				}
#ifdef MIX_F
				target[inp_off + inp_i*numPixelsPerChannel*strideInp] = sum_grad;
#else
				target[inp_off + inp_i*strideInpStep] = sum_grad;
#endif
			}//inp_i	

		}//ix
	}//iy

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

void computeEltwiseDFuncAct(NVMatrix& input, NVMatrix& target, vector<double>& param, int channels, int size_in, int size_out)
{

	assert(size_in <= 4);

    int inp_width = input.getNumCols(); 
    int inp_height = input.getNumRows();

	int out_width = inp_width;
	int out_height = (inp_height*size_out)/size_in;

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
//el switch 
	float Csw = param[param.size()-2];
	float Bsw = param[param.size()-1];

	int numPixelsPerGroup = inp_height/size_in;
	//int numChannelsPerGroup = channels/size_in;
	int numPixelsPerChannel = inp_height/channels;	

    dim3 threads(min(ELTWISE_THREADS_X, inp_width), ELTWISE_THREADS_Y);
    dim3 blocks(std::min(NUM_BLOCKS_MAX, (int)DIVUP(out_width, threads.x)),
                std::min(NUM_BLOCKS_MAX, DIVUP(numPixelsPerGroup, ELTWISE_THREADS_Y)));

#define ELT_DACT(SIZE_ARR) \
	if(size_in == SIZE_ARR){\
	cudaFuncSetCacheConfig(kEltwiseDFuncAct<SIZE_ARR>, cudaFuncCachePreferL1);\
	kEltwiseDFuncAct<SIZE_ARR><<<blocks, threads>>>(input.getDevData(),\
	target.getDevData(), inp_height, inp_width, input.getStride(), target.getStride(), numPixelsPerChannel,\
	Csw, Bsw, size_in, size_out);};
	ELT_DACT(1)
	ELT_DACT(2)
	ELT_DACT(3)
	ELT_DACT(4)

#undef ELT_ACT

//float sumt = target.sum();
//	printf("kEltwiseFuncAct sumt %f \n", sumt);

	cutilCheckMsg("computeEltwiseDFuncAct: Kernel execution failed");
}


void computeEltwiseDFuncParamWeightGrad(NVMatrix& actGrad, NVMatrix& input,
								 void* arrayPtr, vector<NVMatrix>& tempMatrix,
								 NVMatrix& tempC, NVMatrix& tempB,
								 vector<double>& param, float lim,
								 int channels, int size_in, int size_out)
{

	assert(size_out <= 4 && size_in <= 4);// || size_out == 12 || size_out == 16);

    int inp_width = input.getNumCols(); 
    int inp_height = input.getNumRows();

	assert(input.getStride() == actGrad.getStride());

	float temp[CONST_AREA_SIZE];
	assert(param.size() <= CONST_AREA_SIZE);
	memset(temp, 0, sizeof(temp));
	for(int i = 0; i < param.size(); i++)
		temp[i] = (float)param[i];
	cudaMemcpyToSymbol(const_area, temp, sizeof(float)*CONST_AREA_SIZE, 0, cudaMemcpyHostToDevice);
//el switch 
	float Csw = param[param.size()-2];
	float Bsw = param[param.size()-1];

	int numPixelsPerGroup = inp_height/size_in;
	//int numChannelsPerGroup = channels/size_in;
	int numPixelsPerChannel = inp_height/channels;

//	printf("inp_height %i numPixelsPerGroup %i %i\n", inp_height, numPixelsPerGroup, actGrad.getNumRows()/size_out);
#define N_SUM 1
    dim3 threads(min(ELTWISE_THREADS_X, inp_width), ELTWISE_THREADS_Y);
    dim3 blocks(std::min(NUM_BLOCKS_MAX, (int)DIVUP(inp_width, threads.x)),
                std::min(NUM_BLOCKS_MAX, (int)DIVUP(numPixelsPerGroup/N_SUM, ELTWISE_THREADS_Y)));
#undef N_SUM

	int tag_width = blocks.x*threads.x;
	int tag_height = blocks.y*threads.y;
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

#define DELT_W_GRAD(SIZE_ARR_OUT, SIZE_ARR_IN) \
		if(size_out == SIZE_ARR_OUT && size_in == SIZE_ARR_IN){\
		kEltwiseDFuncParamWeightGrad<ELTWISE_THREADS_X, ELTWISE_THREADS_Y, SIZE_ARR_OUT, SIZE_ARR_IN><<<blocks, threads>>>(actGrad.getDevData(),\
		input.getDevData(), (float**)arrayPtr,\
		inp_height, inp_width,\
		input.getStride(), tempMatrix[0].getStride(), numPixelsPerChannel, Csw, Bsw);};
		DELT_W_GRAD(1,2)
		DELT_W_GRAD(2,2)
		DELT_W_GRAD(3,2)
		DELT_W_GRAD(4,2)
		DELT_W_GRAD(1,3)
		DELT_W_GRAD(2,3)
		DELT_W_GRAD(3,3)
		DELT_W_GRAD(4,3)
		DELT_W_GRAD(1,4)
		DELT_W_GRAD(2,4)
		DELT_W_GRAD(3,4)
		DELT_W_GRAD(4,4)
#undef DELT_W_GRAD

}

void computeEltwiseDFuncGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& target,
								 vector<double>& param, int channels, int size_in, int size_out)
{


	assert(size_out <= 4);


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
//el switch 
	float Csw = param[param.size()-2];
	float Bsw = param[param.size()-1];

	int numPixelsPerGroup = inp_height/size_in;
	//int numChannelsPerGroup = channels/size_in;
	int numPixelsPerChannel = inp_height/channels;
    dim3 threads(min(ELTWISE_THREADS_X, inp_width), ELTWISE_THREADS_Y);

    dim3 blocks(std::min(NUM_BLOCKS_MAX, (int)DIVUP(inp_width, threads.x)),
                std::min(NUM_BLOCKS_MAX, DIVUP(numPixelsPerGroup, ELTWISE_THREADS_Y)));


#define ELT_DGRAD(SIZE_ARR) \
		if(size_out == SIZE_ARR){\
			cudaFuncSetCacheConfig(kEltwiseDFuncGrad<SIZE_ARR>, cudaFuncCachePreferL1);\
			kEltwiseDFuncGrad<SIZE_ARR><<<blocks, threads>>>(actGrad.getDevData(),\
				input.getDevData(), target.getDevData(), inp_height, inp_width,\
				input.getStride(), actGrad.getStride(), numPixelsPerChannel,\
				Csw, 1./Csw, Bsw, size_in, size_out);};
		ELT_DGRAD(1)
		ELT_DGRAD(2)
		ELT_DGRAD(3)
		ELT_DGRAD(4)

#undef ELT_DGRAD

//	float sumt = target.sum();
//	printf("FuncGrad sum_tag %f \n", sumt);


	cutilCheckMsg("computeEltwiseFuncGrad: Kernel execution failed");
};
