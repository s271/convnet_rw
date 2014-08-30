#include <matrix.h>
#include <matrix_funcs.h>
#include "conv_debug.h"

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

float* TempMem::allocFloatElement(int size)
{	
	int old_boundary = _boundary;
	_boundary += size;
	alloc(_boundary);
	float* start_element = _floatArea + old_boundary;
	_start.push_back(start_element);
	return start_element;
};

float* TempMem::getPtr(int ind)
{
	return _start[ind];
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

		int sx= ix;
		int sy =iy;


		for(int dsx = - lobe; dsx < lobe+1; dsx++)
		for(int dsy = - lobe; dsy <  lobe+1; dsy++)
		{
			float sdata = input[channelOffset + (sx + dsx + lobe)*widthyz + (sy + dsy + lobe)*widthz + z];
			sum += sdata*filterArea[filterID*sizeModule2 + (-dsy + lobe)*SIZE_MODULE +(-dsx + lobe)];
		}

		target[numFilters*channelOffset + filterID*imgPixels*numCases + ix*widthyz + iy*widthz + z] = sum;
	}
}