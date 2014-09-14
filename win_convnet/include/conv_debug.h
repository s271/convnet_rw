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

extern TempMem singletonTempMem;

double Sum(const float* input, int size);
void debugMicroConvFilterAct(int lobe, int SIZE_MODULE,  float* filterArea, const float* input, float* const target,
								const uint numCases, const uint channels, const uint numFilters,
								const uint sharedY, const uint modulesPerBlockX,  const uint modulesPerBlockY, 
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels);

void emuMicroConvFilterAct(int blockDimx, int blockDimy, int gridDimx, int gridDimy, int LOBE, int SIZE_MODULE, float* filterArea, const float* input, float* const target,
								const uint numCases, const uint channels, const uint numFilters, const uint casePerThread,
								const uint sharedY, const uint modulesPerBlockX,  const uint modulesPerBlockY, 
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels);

void debugMicroConvActGrad(int LOBE, int SIZE_MODULE, float* filterArea, const float* actGrad, float* const target,
								const uint numCases, const uint channels, const uint numFilters, const uint casePerThread,
								const uint modulesPerBlockX, const uint modulesPerBlockY,
								const uint sharedY, const uint sizeModule, const uint lobe,
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels);

 void debugMicroConvWeightGrad(int LOBE, int SIZE_MODULE, int dsx, int dsy, const float* actGrad, const float* input, float* const target_,
								const uint target_size, const uint numCases,
								const uint channels, const uint numFilters, 
								const uint modulesPerBlockX, const uint modulesPerBlockY, const uint sharedY,
								const uint lobe, const uint sizeModule, const uint sizeShared,
								const uint imgSizeX, const uint imgSizeY, const uint imgPixels);

 void debugVectFuncAct(int sizeV, float* filterArea, const float* input, float* const target,
								const uint imgInPixels, const uint numCases,
								const uint strideInp, const uint strideTag, int numColors, int sizeH);


