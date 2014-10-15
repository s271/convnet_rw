#include <assert.h>
#include <vector>
using namespace std;

class TempMem
{
public:
	int _size;
	float* _floatArea;
	vector<int> _start;
	int _boundary;

	TempMem();
	~TempMem();
	void alloc(int size);
	void allocFloatElement(int size);
	float* getPtr(int ind);
	void reset();

};

#define _max(a,b) (a>b?a:b)

extern TempMem singletonTempMem;

double Sum(const float* input, int size);
void debugMicroConvFilterAct(int lobe, int SIZE_CONV,  float* filterArea, const float* input, float* const target,
								const uint numCases, const uint channels, const uint numFilters,
								const uint sharedY, const uint modulesPerBlockX,  const uint modulesPerBlockY, 
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels);

void debugMicroConvLinApprox(int lobe, int SIZE_CONV,  float* filterArea, const float* input, const float* actGrad, float* const target,
								const uint numCases, const uint channels, const uint numFilters,
								const uint sharedY, const uint modulesPerBlockX,  const uint modulesPerBlockY, 
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels);

void emuMicroConvFilterAct(int blockDimx, int blockDimy, int gridDimx, int gridDimy, int LOBE, int SIZE_CONV, float* filterArea, const float* input, float* const target,
								const uint numCases, const uint channels, const uint numFilters, const uint casePerThread,
								const uint sharedY, const uint modulesPerBlockX,  const uint modulesPerBlockY, 
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels);

void debugMicroConvActGrad(int LOBE, int SIZE_CONV, float* filterArea, const float* actGrad, float* const target,
								const uint numCases, const uint channels, const uint numFilters, const uint casePerThread,
								const uint modulesPerBlockX, const uint modulesPerBlockY,
								const uint sharedY, const uint sizeModule, const uint lobe,
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels);

 void debugMicroConvWeightGrad(int LOBE, int SIZE_CONV, int dsx, int dsy, int  filterID, int channelInd,
								const float* actGrad, const float* input, float* const target_,
								const uint target_size, const uint numCases,
								const uint channels, const uint numFilters, 
								const uint modulesPerBlockX, const uint modulesPerBlockY, const uint sharedY,
								const uint lobe, const uint sizeModule, const uint sizeShared,
								const uint imgSizeX, const uint imgSizeY, const uint imgPixels);
void emuMicroConvWeightGrad(int blockDimx, int blockDimy, int gridDimx, int gridDimy, 
							int lobe, int SIZE_CONV, int dsx, int dsy, int filterID, int channelInd,
							const float* actGrad, const float* input, float* const target,
							const uint target_size, const uint numCases, const uint casePerThread, const uint tagWidth,
							const uint channels, const uint numFilters, 
							const uint modulesPerBlockX, const uint modulesPerBlockY, const uint sharedY,
							const uint sizeSharedBlock,
							const uint imgSizeX, const uint imgSizeY, const uint imgPixels);


void debugVectFuncAct(int sizeV, float* filterArea, const float* input, float* const target,
								const uint imgInPixels, const uint numCases,
								const uint strideInp, const uint strideTag, int numColors, int sizeH);

void emuVectFuncAct(int sizeV, float* filterArea, int gridDimy, int blockDimy, int gridDimx, int blockDimx,
					float* input, float* const target,
					const uint imgInPixels, const uint numCases,
					const uint strideInp, const uint strideTag, int numColors, int sizeH);

void debugVectFuncLinApprox(int sizeV, float* filterArea, const float* input,
								const float* actGrad, float* const target,
								const uint numPixelsPerGroup, const uint numCases,
								const uint strideInp, const uint strideTag, int numColors, int sizeH);

void debugVectFuncParamWeightGrad(int sizeV, float* filterArea,	int gridDimy, int blockDimy, int gridDimx, int blockDimx,
							  const float* actGrad, const float* input, float* const target_,
											const uint numColors,
											const uint target_size, const uint numPixelsPerGroup, const uint numCases,
											const uint strideInp, const uint strideOut, const uint strideTag, int sizeH);

void debugVectFuncGrad(int sizeV, float* filterArea, const float* actGrad, const float* input, float* const target,
								const uint numPixelsPerGroup, const uint numCases,
								const uint strideInp, const uint strideOut,
								int numColors, int sizeH);

