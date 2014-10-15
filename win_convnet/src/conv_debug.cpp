#include <matrix.h>
#include <matrix_funcs.h>
#include <math.h> 
#include "conv_debug.h"
#include "tt.h"
#pragma warning( disable : 4018 )


#define max std::max<int>
#define min std::min<int>

TempMem singletonTempMem;

TempMem::TempMem()
{
	_size =0;
	_boundary = 0;
	_floatArea = NULL;
}

TempMem::~TempMem()
{
	if(_size > 0)
		delete [] _floatArea;
}

void TempMem::alloc(int size)
{
	if(size > _size)
	{
		if(_size > 0)
			delete [] _floatArea;

		_floatArea = new float[size];

		_size = size;
	}
};

void TempMem::allocFloatElement(int size)
{	
	int old_boundary = _boundary;
	_boundary += size;
	alloc(_boundary);
	_start.push_back(old_boundary);
};

float* TempMem::getPtr(int ind)
{
	return _floatArea+_start[ind];
}

void TempMem::reset()
{
	_boundary = 0;
	_start.clear();
}

double Sum(const float* input, int size)
{
	double sum =0;
	for(int i = 0; i < size; i++)
		sum += (double)input[i];

	return sum;
}

//-------------------------------------------------------------
//MicroConv
//-------------------------------------------------------------

void debugMicroConvFilterAct(int lobe, int SIZE_CONV, float* filterArea, const float* input, float* const target, 
								const uint numCases, const uint channels, const uint numFilters,
								const uint sharedY, const uint modulesPerBlockX,  const uint modulesPerBlockY, 
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels)
{
	const int widthz = numCases;
	const int widthyz = imgSizeY*numCases;
	int sizeConv2 = SIZE_CONV*SIZE_CONV;

	for(int channelInd = 0; channelInd < channels; channelInd++)
	for(int ix = 0; ix < imgSizeX; ix++)
	for(int iy = 0; iy < imgSizeY; iy++)
	for(int z = 0; z < numCases; z++)
	for(int filterID = 0; filterID <  numFilters; filterID++)
	{

		const int channelOffset = channelInd*imgPixels*numCases;

		float sum = 0;

		for(int dsx = - lobe; dsx < lobe+1; dsx++)
		for(int dsy = - lobe; dsy <  lobe+1; dsy++)
		{
			int idx = min(max(ix + dsx, 0), imgSizeX-1);
			int idy = min(max(iy + dsy, 0), imgSizeY-1);
			float sdata = input[channelOffset + idx*widthyz + idy*widthz + z];
			sum += sdata*filterArea[channelInd*sizeConv2*numFilters + filterID*sizeConv2 + (dsy + lobe)*SIZE_CONV +(dsx + lobe)];
		}

		target[numFilters*channelOffset + filterID*imgPixels*numCases + ix*widthyz + iy*widthz + z] = sum;
	}
}


#define SMEM(X, Y, sdata) sdata[(X)*sharedY+(Y)+sOffset]

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

void emuMicroConvFilterAct(int blockDimx, int blockDimy, int gridDimx, int gridDimy, int LOBE, int SIZE_CONV, float* filterArea, const float* input, float* const target,
								const uint numCases, const uint channels, const uint numFilters, const uint casePerThread,
								const uint sharedY, const uint modulesPerBlockX,  const uint modulesPerBlockY, 
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels)
{
	float sdata[10*10*4*3];
	memset(sdata, 0, sizeof(sdata));

	const int widthz = numCases;
	const int widthyz = imgSizeY*numCases;

	const int sizeConv2 = SIZE_CONV*SIZE_CONV;
	const int sharedY2 = sharedY*sharedY;

	const int  bw = modulesPerBlockX;
	const int  bh = modulesPerBlockY;

			const int bsizeX = imgSizeX/modulesPerBlockX;
			const int bsizeY = imgSizeY/modulesPerBlockY;

	printf("gridDimx %i gridDimy %i blockDimx %i blockDimy %i modulesPerBlockY %i channels %i sharedY %i SIZE_CONV %i \n"
		,gridDimx, gridDimy, blockDimx, blockDimy, modulesPerBlockY, channels, sharedY, SIZE_CONV);

	for(int blockIdxx = 0; blockIdxx < gridDimx; blockIdxx++)
	for(int blockIdxy = 0; blockIdxy < gridDimy; blockIdxy++)
	{
		for(int zind = 0; zind < casePerThread; zind++)
		{
		for(int threadIdxx = 0; threadIdxx < blockDimx; threadIdxx++)
		for(int threadIdxy = 0; threadIdxy < blockDimy; threadIdxy++)
		{

			const int startX = (blockIdxy/bsizeY)*modulesPerBlockX;
			const int startY = (blockIdxy%bsizeY)*modulesPerBlockY;

			const int  sx = threadIdxy/modulesPerBlockY;
			const int  sy = threadIdxy - sx*modulesPerBlockY;

			const int  ix = sx+startX;
			const int  iy = sy+startY;

			const int z = threadIdxx + blockIdxx*blockDimx + zind*blockDimx*gridDimx;	

		//put pragme unroll here	
			for(int channelInd = 0; channelInd < channels; channelInd++)
			{	
					const int sOffset = channelInd*sharedY2*blockDimx + threadIdxx*sharedY2;
					const int channelOffset = channelInd*imgPixels*numCases;

					if(z < numCases)
					{
						for(int filterID = 0; filterID <  numFilters; filterID++)
						{
								
								SHARED_MEM(ix, iy, z, LOBE, getValInput, sdata)
						}//filted
					}//if
				}//z,channel
		}//x thr

		for(int threadIdxx = 0; threadIdxx < blockDimx; threadIdxx++)
		for(int threadIdxy = 0; threadIdxy < blockDimy; threadIdxy++)
		{
		//order x>y>z, *not* y>x
			const int startX = (blockIdxy/bsizeY)*modulesPerBlockX;
			const int startY = (blockIdxy%bsizeY)*modulesPerBlockY;

			const int  sx = threadIdxy/modulesPerBlockY;
			const int  sy = threadIdxy - sx*modulesPerBlockY;

			const int  ix = sx+startX;
			const int  iy = sy+startY;

			const int z = threadIdxx + blockIdxx*blockDimx + zind*blockDimx*gridDimx;	


			for(int channelInd = 0; channelInd < channels; channelInd++)
			{	
					const int sOffset = channelInd*sharedY2*blockDimx + threadIdxx*sharedY2;
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

									sum += sd*filterArea[filterID*sizeConv2 + (-dsy + LOBE)*SIZE_CONV +(-dsx + LOBE)];

								}	
										

								target[numFilters*channelOffset + filterID*imgPixels*numCases + ix*widthyz + iy*widthz + z] = sum;
						}//filterID
					}//if(z < numCases)
				}//channel
		}//thread y
		}//zind

	}// block
	
}

#define getValAct(X, Y, Z) actGrad[filterOffset + (X)*widthyz+(Y)*widthz + (Z)]

void debugMicroConvActGrad(int LOBE, int SIZE_CONV, float* filterArea, const float* actGrad, float* const target,
								const uint numCases, const uint channels, const uint numFilters, const uint casePerThread,
								const uint modulesPerBlockX, const uint modulesPerBlockY,
								const uint sharedY, const uint sizeModule, const uint lobe,
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels)
{


	const int widthz = numCases;
	const int widthyz = imgSizeY*numCases;
	int sizeConv2 = SIZE_CONV*SIZE_CONV;

	for(int z = 0; z < numCases; z++)
	{

		for(int channelInd = 0; channelInd < channels; channelInd++)
		{
			const int channelOffset = channelInd*imgPixels*numCases;
			for(int ix = 0; ix < imgSizeX; ix++)
			for(int iy = 0; iy < imgSizeY; iy++)
			{
				float sum = 0;
				for(int filterID = 0; filterID <  numFilters; filterID++)
				{
					const int filterOffset = numFilters*channelOffset + filterID*imgPixels*numCases;
					
					for(int dsx = - lobe; dsx < lobe+1; dsx++)
					for(int dsy = - lobe; dsy <  lobe+1; dsy++)
					{

						int idx = min(max(ix + dsx, 0), imgSizeX-1);
						int idy = min(max(iy + dsy, 0), imgSizeY-1);

						float inpd = actGrad[filterOffset + idx*widthyz + idy*widthz + z];

						sum += inpd*filterArea[filterID*sizeConv2 + (-dsy + lobe)*sizeModule +(-dsx + lobe)];
					}
				}
				target[channelOffset + ix*widthyz + iy*widthz + z] = sum;
			}
		}
	}
}

void debugMicroConvLinApprox(int lobe, int SIZE_CONV, float* filterArea, const float* input, const float* actGrad, float* const target, 
								const uint numCases, const uint channels, const uint numFilters,
								const uint sharedY, const uint modulesPerBlockX,  const uint modulesPerBlockY, 
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels)
{
	const int widthz = numCases;
	const int widthyz = imgSizeY*numCases;
	int sizeConv2 = SIZE_CONV*SIZE_CONV;

	for(int channelInd = 0; channelInd < channels; channelInd++)
	for(int ix = 0; ix < imgSizeX; ix++)
	for(int iy = 0; iy < imgSizeY; iy++)
	for(int z = 0; z < numCases; z++)
	for(int filterID = 0; filterID <  numFilters; filterID++)
	{

		const int channelOffset = channelInd*imgPixels*numCases;

		float sum = 0;


		for(int dsx = - lobe; dsx < lobe+1; dsx++)
		for(int dsy = - lobe; dsy <  lobe+1; dsy++)
		{
			int idx = min(max(ix + dsx, 0), imgSizeX-1);
			int idy = min(max(iy + dsy, 0), imgSizeY-1);
			float sdata = input[channelOffset + idx*widthyz + idy*widthz + z];


			sum += sdata*filterArea[channelInd*sizeConv2*numFilters + filterID*sizeConv2 + (dsy + lobe)*SIZE_CONV +(dsx + lobe)];
		}

		float act = actGrad[numFilters*channelOffset + filterID*imgPixels*numCases + ix*widthyz + iy*widthz + z];

		target[numFilters*channelOffset + filterID*imgPixels*numCases + ix*widthyz + iy*widthz + z] = act*sum;
	}
}

 void debugMicroConvWeightGrad(int LOBE, int SIZE_CONV, int dsx, int dsy, int filterID, int channelInd,
								const float* actGrad, const float* input, float* const target_,
								const uint target_size, const uint numCases,
								const uint channels, const uint numFilters, 
								const uint modulesPerBlockX, const uint modulesPerBlockY, const uint sharedY,
								const uint lobe, const uint sizeModule, const uint sizeShared,
								const uint imgSizeX, const uint imgSizeY, const uint imgPixels)
{

//order x>y>z, *not* y>x

	const int widthz = numCases;
	const int widthyz = imgSizeY*numCases;

	const int sizeModule2 = sizeModule*sizeModule;
	const int sharedY2 = sharedY*sharedY;
	const int imgSize = imgSizeX*imgSizeY;
	const int channelOffset = channelInd*imgSize*numCases;

	for(int ix = 0; ix < imgSizeX; ix++)
	for(int iy = 0; iy < imgSizeY; iy++)
	{

//pragma unroll here
		float sum = 0;

			const int filterOffset = numFilters*channelOffset + filterID*imgSize*numCases;

			for(int z = 0; z < numCases; z++)
			{	
					int idx = min(max(ix + dsx, 0), imgSizeX-1);
					int idy = min(max(iy + dsy, 0), imgSizeY-1);

					float actd = actGrad[filterOffset + ix*widthyz + iy*widthz + z];
					float imgd = input[channelOffset + idx*widthyz + idy*widthz + z];							
					sum += actd*imgd;
			}//z
			target_[ix*imgSizeX + iy] = sum;
	}//ix
}



float sdata_mc[20032/4];
void emuMicroConvWeightGrad(int blockDimx, int blockDimy, int gridDimx, int gridDimy, 
							int lobe, int SIZE_CONV, int dsx, int dsy, int filterID, int channelInd,
							const float* actGrad, const float* input, float* const target,
							const uint target_size, const uint numCases, const uint casePerThread, const uint tagWidth,
							const uint channels, const uint numFilters, 
							const uint modulesPerBlockX, const uint modulesPerBlockY, const uint sharedY,
							const uint sizeSharedBlock,
							const uint imgSizeX, const uint imgSizeY, const uint imgPixels)
{


//order x>y>z, *not* y>x
	float* sdataImg = sdata_mc;
	float* sdataRes = sdata_mc + sizeSharedBlock*blockDimx;
	const int imgSize = imgSizeX*imgSizeY;

	const int bsizeX = imgSizeX/modulesPerBlockX;
	const int bsizeY = imgSizeY/modulesPerBlockY;

    const int  bw = modulesPerBlockX;
    const int  bh = modulesPerBlockY;


	const int sharedY2 = sharedY*sharedY;

	const int conv_size = 2*lobe+1;
	const int conv2 = conv_size*conv_size;
	int resStride = numFilters*conv2;

	for(int blockIdxx = 0; blockIdxx < gridDimx; blockIdxx++)
	for(int blockIdxy = 0; blockIdxy < gridDimy; blockIdxy++)
	{

		for(int threadIdxx = 0; threadIdxx < blockDimx; threadIdxx++)
		for(int threadIdxy = 0; threadIdxy < blockDimy; threadIdxy++)
		{
			int res_off = resStride*(threadIdxy*blockDimx + threadIdxx);
			memset(sdataRes, 0, blockDimx*blockDimy*resStride*sizeof(float));
		}

		for(int zind = 0; zind < casePerThread; zind++)
		{


			for(int threadIdxx = 0; threadIdxx < blockDimx; threadIdxx++)
			for(int threadIdxy = 0; threadIdxy < blockDimy; threadIdxy++)
			{
				const int startX = (blockIdxy/bsizeY)*modulesPerBlockX;
				const int startY = (blockIdxy%bsizeY)*modulesPerBlockY;

				const int  bw = modulesPerBlockX;
				const int  bh = modulesPerBlockY;
				const int  sx = threadIdxy/modulesPerBlockY;
				const int  sy = threadIdxy - sx*modulesPerBlockY;

				const int  ix = sx+startX;
				const int  iy = sy+startY;

				const int zoff = threadIdxx + blockIdxx*blockDimx;
				const int widthz = numCases;
				const int widthyz = imgSizeY*numCases;

				const int sharedY2 = sharedY*sharedY;
				const int sOffset = threadIdxx*sharedY2;

				const int channelOffset = channelInd*imgPixels*numCases;
				const int z = zoff + zind*blockDimx*gridDimx;
				SHARED_MEM(ix, iy, z, lobe, getValInput, sdataImg)	

			}


			for(int threadIdxx = 0; threadIdxx < blockDimx; threadIdxx++)
			for(int threadIdxy = 0; threadIdxy < blockDimy; threadIdxy++)
			{

				const int startX = (blockIdxy/bsizeY)*modulesPerBlockX;
				const int startY = (blockIdxy%bsizeY)*modulesPerBlockY;


				const int  sx = threadIdxy/modulesPerBlockY;
				const int  sy = threadIdxy - sx*modulesPerBlockY;

				const int  ix = sx+startX;
				const int  iy = sy+startY;

				const int zoff = threadIdxx + blockIdxx*blockDimx;

				const int widthz = numCases;
				const int widthyz = imgSizeY*numCases;

				int res_off = resStride*(threadIdxy*blockDimx + threadIdxx);

			//	memset(sdataRes + res_off, 0, resStride*sizeof(float));
				const int sOffset = threadIdxx*sharedY2;

				const int channelOffset = channelInd*imgPixels*numCases;

				const int z = zoff + zind*blockDimx*gridDimx;		

				{
					int idx = min(max(ix + dsx, 0), imgSizeX-1);
					int idy = min(max(iy + dsy, 0), imgSizeY-1);

					{

						const int filterOffset = numFilters*channelOffset + filterID*imgPixels*numCases;				
						float vact = actGrad[filterOffset + ix*widthyz + iy*widthz + z];
						float vimg = sdataImg[(sx + dsx + lobe)*sharedY+(sy + dsy + lobe) + sOffset];
							//input[channelOffset + idx*widthyz + idy*widthz + z];

						int ind_coeff = filterID*conv2 + (dsy + lobe)*conv_size +(dsx + lobe);
						sdataRes[res_off + ind_coeff] += vact*vimg;
						//target[ix*imgSizeX*tagWidth + tagWidth*iy + zoff] += vact*vimg;

					}//filter

				}//dsx

			}//threads
		}//zind

		for(int threadIdxx = 0; threadIdxx < blockDimx; threadIdxx++)
		for(int threadIdxy = 0; threadIdxy < blockDimy; threadIdxy++)
		{
			const int startX = (blockIdxy/bsizeY)*modulesPerBlockX;
			const int startY = (blockIdxy%bsizeY)*modulesPerBlockY;


			const int  sx = threadIdxy/modulesPerBlockY;
			const int  sy = threadIdxy - sx*modulesPerBlockY;

			const int  ix = sx+startX;
			const int  iy = sy+startY;
			const int zoff = threadIdxx + blockIdxx*blockDimx;
			int res_off = resStride*(threadIdxy*blockDimx + threadIdxx);


			int isx = dsx+lobe;
			int isy = dsy+lobe;

			{
				int ind_coeff = filterID*conv2 + isy*conv_size + isx;
				int ind_ch = ind_coeff + channelInd*numFilters*conv2;
				target[ix*imgSizeX*tagWidth + tagWidth*iy + zoff] = sdataRes[res_off + ind_coeff];
			}

		}//threads

	}//blocks

}

void debugVectFuncLinApprox(int sizeV, float* filterArea, const float* input,
								const float* actGrad, float* const target,
								const uint numPixelsPerGroup, const uint numCases,
								const uint strideInp, const uint strideTag, int numColors, int sizeH)
{
    for (int iy = 0; iy < numPixelsPerGroup; iy ++) {

        for (uint ix = 0; ix < numCases; ix ++) {	

			for (uint color = 0; color < numColors; color ++) {	
			
				for (uint out_i = 0; out_i < sizeH; out_i++) {

					float grad_next = actGrad[color*sizeH*numPixelsPerGroup*numCases +  
					out_i*numPixelsPerGroup*numCases + iy*numCases + ix];
					
					float output = 0;
			
					for (uint inp_i = 0; inp_i < sizeV; inp_i++)
					{		
						float param = filterArea[out_i*sizeV + inp_i];
					    float val =
							  input[color*sizeV*numPixelsPerGroup*strideInp + inp_i*numPixelsPerGroup*strideInp +  iy*strideInp + ix];
	
						output += param*val;
					}

					//suppression filter could be here

					output = grad_next*_max(output, 0);

					target[color*sizeH*numPixelsPerGroup*strideInp + out_i*numPixelsPerGroup*strideInp +  iy*strideInp + ix]
						= output;
				}//out_i
			}
        }
    }
}

void debugVectFuncAct(int sizeV, float* filterArea, const float* input, float* const target,
								const uint numPixelsPerGroup, const uint numCases,
								const uint strideInp, const uint strideTag, int numColors, int sizeH)
{
    for (int iy = 0; iy < numPixelsPerGroup; iy ++) {

        for (uint ix = 0; ix < numCases; ix ++) {	

			for (uint color = 0; color < numColors; color ++) {	
			
				for (uint out_i = 0; out_i < sizeH; out_i++) {

					float output = 0;
			
					for (uint inp_i = 0; inp_i < sizeV; inp_i++)
					{		
						float param = filterArea[out_i*sizeV + inp_i];
					    float val =
							  input[color*sizeV*numPixelsPerGroup*strideInp + inp_i*numPixelsPerGroup*strideInp +  iy*strideInp + ix];
	
						output += param*val;
					}

					//suppression filter could be here

					output = _max(output, 0);

					target[color*sizeH*numPixelsPerGroup*strideInp + out_i*numPixelsPerGroup*strideInp +  iy*strideInp + ix]
						= output;
				}//out_i
			}
        }
    }
}

void emuVectFuncAct(int sizeV, float* filterArea, int gridDimy, int blockDimy, int gridDimx, int blockDimx,
					float* input, float* const target,
					const uint numPixelsPerGroup, const uint numCases,
					const uint strideInp, const uint strideTag, int numColors, int sizeH) {



	for(int blockIdxx = 0; blockIdxx < gridDimx; blockIdxx++)
	for(int blockIdxy = 0; blockIdxy < gridDimy; blockIdxy++)
	{

	for(int threadIdxx = 0; threadIdxx < blockDimx; threadIdxx++)
	for(int threadIdxy = 0; threadIdxy < blockDimy; threadIdxy++)
	{

		for (uint iy = 0; iy < numPixelsPerGroup; iy += gridDimy*blockDimy) {

			for (uint ix = 0; ix < numCases; ix += gridDimx*blockDimx) {	

			for (uint color = 0; color < numColors; color ++) {	
			
				float inpVal[2];//use shared instead?

				for (uint inp_i = 0; inp_i < sizeV; inp_i++) {	
					Offset inpOffset;
					inpOffset << Index(color) << sizeV
					<< Index(inp_i)
					<< numPixelsPerGroup
					<< Index(iy) << Index(blockDimy, blockIdxy) << Index(threadIdxy)
					<< strideInp
					<< Index(ix ) << Index(blockDimx, blockIdxx) << Index(threadIdxx);
					float val = input[inpOffset._offset];

					//int voff = color*sizeV*numPixelsPerGroup*strideInp  +
					//	inp_i*numPixelsPerGroup*strideInp +
					//	(iy + blockDimy*blockIdxy + threadIdxy)*strideInp+
					//	ix + blockDimx*blockIdxx + threadIdxx;

					//float val = input[voff];

					inpVal[inp_i] = val;
				}
	
				for (uint out_i = 0; out_i < sizeH; out_i++) {
					int out_par = out_i*sizeV;

					float output = 0;
			
					for (uint inp_i = 0; inp_i < sizeV; inp_i++)
					{		
						float param = filterArea[out_par + inp_i];
						float val =	inpVal[inp_i];
						output += param*val;
					}// inp_i

					//suppression filter could be here

					//output = output >0?output: 0;
					output = _max(output, 0);

					Offset tagOffset;
					tagOffset << Index(color) << sizeH
					<<Index(out_i)
					<< numPixelsPerGroup
					<< Index(iy) << Index(blockDimy, blockIdxy) << Index(threadIdxy)
					<< strideTag
					<< Index(ix ) << Index(blockDimx, blockIdxx) << Index(threadIdxx);
					target[tagOffset._offset] = output;
					//int toffset = color*sizeH*numPixelsPerGroup*strideInp + out_i*numPixelsPerGroup*strideInp
					//	+  (iy + blockDimy*blockIdxy +threadIdxy)*strideInp + ix + blockDimx*blockIdxx + threadIdxx;
					//target[toffset] = output;
				}//out_i
			}//color
        }//ix
    }//iy

	}//threads
	}//blocks


}

void debugVectFuncParamWeightGrad(int sizeV, float* filterArea,	int gridDimy, int blockDimy, int gridDimx, int blockDimx,
							  const float* actGrad, const float* input, float* const target_,
											const uint numColors,
											const uint target_size, const uint numPixelsPerGroup, const uint numCases,
											const uint strideInp, const uint strideOut, const uint strideTag, int sizeH)
{


	int pout = 0;
	int pin_t = 1;


//	for (uint pout = 0; pout < sizeH; pout++)
//	{
	
		for (uint iy = 0; iy < numPixelsPerGroup; iy ++) {
		  for (uint ix = 0; ix < numCases; ix ++) {	

			 float vres[256];
			  memset(vres, 0, sizeof(vres));

			  for (uint color = 0; color < numColors; color ++) {	//optimize away				

					float grad_next = actGrad[color*sizeH*numPixelsPerGroup*numCases +  
					pout*numPixelsPerGroup*numCases + iy*numCases + ix];

					float in_val[256];
					float vsum = 0;
					for (uint pin = 0; pin < sizeV; pin++)
					{
						
						in_val[pin] = input[color*sizeV*numPixelsPerGroup*numCases +  
										pin*numPixelsPerGroup*numCases + iy*numCases + ix];

						vsum += in_val[pin]*filterArea[pout*sizeV + pin];
					}

					if(vsum > 0)
					{
						for (uint pin = 0; pin < sizeV; pin++)
						{		
							vres[pin] += grad_next*in_val[pin];
						}
					}//if
				}//color

  				target_[iy*strideTag + ix] = vres[pin_t];

			}//ix

		}//iy

//	}// pout

}

void debugVectFuncGrad(int sizeV, float* filterArea, const float* actGrad, const float* input, float* const target,
								const uint numPixelsPerGroup, const uint numCases,
								const uint strideInp, const uint strideOut,
								int numColors, int sizeH) {

	const int inStep = strideInp*numPixelsPerGroup;
	const int outStep = strideOut*numPixelsPerGroup;

//with no N_SUM ix, iy == 0 almost always
    for (uint iy = 0; iy < numPixelsPerGroup; iy ++) {
        for (uint ix = 0; ix < numCases; ix ++) {	
			for (uint color = 0; color < numColors; color ++) {	


				int out_off = color*sizeH*numPixelsPerGroup*strideOut + iy*strideOut +ix;
				int v_off = color*sizeV*numPixelsPerGroup*strideOut + iy*strideOut +ix;

				float vres[256];
				memset(vres, 0, sizeof(vres));

				for (uint out_i = 0; out_i < sizeH; out_i++)
				{
					float vsum = 0;
					for (uint inp_i = 0; inp_i < sizeV; inp_i++) {	
						int inp_offset = v_off + inp_i*inStep;

							//v_offset._offset + inp_i*inStep;

						vsum += input[inp_offset]*filterArea[out_i*sizeV + inp_i];
					}

					if(vsum > 0)
					{
						float grad_next = actGrad[out_off + outStep*out_i];
							//actGrad[out_offset._offset + outStep*out_i];

						for (uint inp_i = 0; inp_i < sizeV; inp_i++)
							vres[inp_i] += grad_next*filterArea[out_i*sizeV + inp_i];
					}
				}

				for (uint inp_i = 0; inp_i < sizeV; inp_i++)
				{
					int inp_offset = v_off + inp_i*inStep;
						//v_offset._offset + inp_i*inStep;
					target[inp_offset] = vres[inp_i];
				}

			}//color
		}//ix
	}//iy
}



