#include <assert.h>
#include <vector>
using namespace std;
double Sum(const float* input, int size);
void debugMicroConvFilterAct(int lobe, int SIZE_MODULE,  float* filterArea, const float* input, float* const target,
								const uint numCases, const uint channels, const uint numFilters,
								const uint sharedY, const uint modulesPerBlockX,  const uint modulesPerBlockY, 
								const uint imgSizeX, const uint imgSizeY,
								const uint imgPixels);

class TempMem
{
	int _size;
	float* _floatArea;
	vector<float*> _start;
	int _boundary;
public:
	TempMem();
	~TempMem();
	void alloc(int size);
	float* allocFloatElement(int size);
	float* getPtr(int ind);
	void reset();

};

extern TempMem singletonTempMem;