#include <matrix.h>
#include <matrix_funcs.h>
#include "conv_debug.h"


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

void debugMicroConvFilterAct(int lobe, int SIZE_MODULE, float* filterArea, const float* input, float* const target, 
								const uint numCases, const uint channels, const uint numFilters,
								const uint sharedY, const uint modulesPerBlockX,  const uint modulesPerBlockY, 
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels)
{
	const int widthz = numCases;
	const int widthyz = imgSizeY*numCases;
	int sizeModule2 = SIZE_MODULE*SIZE_MODULE;

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
			sum += sdata*filterArea[filterID*sizeModule2 + (-dsy + lobe)*SIZE_MODULE +(-dsx + lobe)];
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
        SMEM((LOBE) + bw + sx, sy, sdata) = getVal(min(x + bh, imgSizeX-1), max(y - (LOBE), 0), z);\
        SMEM((LOBE) + bw + sx, (LOBE) + bh + sy, sdata) = getVal(min(x + bw, imgSizeX-1), min(y + bh, imgSizeY-1), z);\
    }

#define getValInput(X, Y, Z) input[channelOffset + (X)*widthyz+(Y)*widthz + (Z)]

void emuMicroConvFilterAct(int blockDimx, int blockDimy, int gridDimx, int gridDimy, int LOBE, int SIZE_MODULE, float* filterArea, const float* input, float* const target,
								const uint numCases, const uint channels, const uint numFilters, const uint casePerThread,
								const uint sharedY, const uint modulesPerBlockX,  const uint modulesPerBlockY, 
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels)
{
	float sdata[10*10*128*3*2];
	memset(sdata, 0, sizeof(sdata));

	const int widthz = numCases;
	const int widthyz = imgSizeY*numCases;

	const int sizeModule2 = SIZE_MODULE*SIZE_MODULE;
	const int sharedY2 = sharedY*sharedY;

	const int  bw = modulesPerBlockX;
	const int  bh = modulesPerBlockY;

	printf("gridDimx %i gridDimy %i blockDimx %i blockDimy %i modulesPerBlockY %i channels %i sharedY %i SIZE_MODULE %i \n"
		,gridDimx, gridDimy, blockDimx, blockDimy, modulesPerBlockY, channels, sharedY, SIZE_MODULE);


	for(int blockIdxx = 0; blockIdxx < gridDimx; blockIdxx++)
	for(int blockIdxy = 0; blockIdxy < gridDimy; blockIdxy++)
	{

	for(int zind = 0; zind < casePerThread; zind++)
	{

		for(int threadIdxx = 0; threadIdxx < blockDimx; threadIdxx++)
		{
			const int z = threadIdxx + blockIdxx*blockDimx + zind*blockDimx*gridDimx;

			for(int threadIdxy = 0; threadIdxy < blockDimy; threadIdxy++)
			{


			const int  sx = threadIdxy/modulesPerBlockY;
			const int  sy = threadIdxy - sx*modulesPerBlockY;

			const int bsizeX = imgSizeX/modulesPerBlockX;
			const int bsizeY = imgSizeY/modulesPerBlockY;
			const int startX = (blockIdxy/bsizeY)*modulesPerBlockX;
			const int startY = (blockIdxy%bsizeY)*modulesPerBlockY;

			const int  ix = sx+startX;
			const int  iy = sy+startY;

			const int widthz = numCases;
			const int widthyz = imgSizeY*numCases;

			const int sizeModule2 = SIZE_MODULE*SIZE_MODULE;

			const int  bw = modulesPerBlockX;
			const int  bh = modulesPerBlockY;

		//put pragme unroll here	
			for(int channelInd = 0; channelInd < channels; channelInd++)
			{
				const int sOffset = channelInd*sharedY2*casePerThread + zind*sharedY2;
				const int channelOffset = channelInd*imgPixels*numCases;

				if(z < numCases)
				{
					for(int filterID = 0; filterID <  numFilters; filterID++)
					{
							
							SHARED_MEM(ix, iy, z, LOBE, getValInput, sdata)

					}//filted
				}//if
			}//channel
		}// y thr
		}//X THR

		for(int threadIdxx = 0; threadIdxx < blockDimx; threadIdxx++)
		{
			const int z = threadIdxx + blockIdxx*blockDimx + zind*blockDimx*gridDimx;

		for(int threadIdxy = 0; threadIdxy < blockDimy; threadIdxy++)
		{
		//order x>y>z, *not* y>x
			const int  sx = threadIdxy/modulesPerBlockY;
			const int  sy = threadIdxy - sx*modulesPerBlockY;

			const int bsizeX = imgSizeX/modulesPerBlockX;
			const int bsizeY = imgSizeY/modulesPerBlockY;
			const int startX = (blockIdxy/bsizeY)*modulesPerBlockX;
			const int startY = (blockIdxy%bsizeY)*modulesPerBlockY;

			const int  ix = sx+startX;
			const int  iy = sy+startY;

			const int widthz = numCases;
			const int widthyz = imgSizeY*numCases;

			const int sizeModule2 = SIZE_MODULE*SIZE_MODULE;

			const int  bw = modulesPerBlockX;
			const int  bh = modulesPerBlockY;

			for(int channelInd = 0; channelInd < channels; channelInd++)
			{	
				const int sOffset = channelInd*sharedY2*casePerThread + zind*sharedY2;
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

								sum += sd*filterArea[filterID*sizeModule2 + (-dsy + LOBE)*SIZE_MODULE +(-dsx + LOBE)];

							}											
							target[numFilters*channelOffset + filterID*imgPixels*numCases + ix*widthyz + iy*widthz + z] = sum;
					}//filterID
				}//if(z < numCases)
			}//channel

		}//thread x
		}//thread y
		}//zind
	}// block
	
}
