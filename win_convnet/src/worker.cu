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

#include <algorithm>
#include <util.cuh>
#include <worker.cuh>

using namespace std;

/* 
 * ====================
 * WorkResult
 * ====================
 */
WorkResult::WorkResult(WorkResult::RESULTS resultType, Cost& results) : _resultType(resultType), _results(&results) {
}

WorkResult::WorkResult(WorkResult::RESULTS resultType) : _resultType(resultType), _results(NULL) {
}

WorkResult::~WorkResult() {
    delete _results; // delete NULL is ok
}

Cost& WorkResult::getResults() const {
    return *_results;
}

WorkResult::RESULTS WorkResult::getResultType() const {
    return _resultType;
}

/* 
 * ====================
 * Worker
 * ====================
 */
Worker::Worker(ConvNet& convNet) : _convNet(&convNet) {
}

/* 
 * ====================
 * DataWorker
 * ====================
 */
DataWorker::DataWorker(ConvNet& convNet, CPUData& data) : Worker(convNet), _data(&data) {
    _dp = &convNet.getDataProvider();
}

DataWorker::~DataWorker() {
    _dp->clearData();
}

/* 
 * ====================
 * TrainingWorker
 * ====================
 */
TrainingWorker::TrainingWorker(ConvNet& convNet, CPUData& data, bool test, int epoch, float eps_scale) 
    : DataWorker(convNet, data), _test(test), _epoch(epoch), _eps_scale(eps_scale) {
}

//debug
int minibatch=0;
int gepoch = 0;

// Need to setData here (as opposed to the constructor) because the constructor executes in
// the original CPU thread, which is not the one with GPU access.
void TrainingWorker::run() {
    _dp->setData(*_data);
	_convNet->setEpoch(_epoch);

//debug
	gepoch = _epoch;
	size_t free_mem;
	size_t total_mem;
	cudaError_t  err = cudaMemGetInfo(&free_mem, &total_mem);
	printf(" free memory  %f \n", free_mem/1e6);

//	auxPass();

    Cost& batchCost = *new Cost(0);

	trainingPass(batchCost);

    cudaThreadSynchronize();
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
}

void TrainingWorker::auxPass() {
//choose subset 
	//int subset_size = .3*_dp->getNumMinibatches();
	//vector<int> subset;
	//vector<int> mask;

	//for (int i = 0; i < _dp->getNumMinibatches(); i++)
	//	mask.push_back(0);

	//for (int i=0; i< subset_size; i++)
	//{
	//	int r = rand()%(_dp->getNumMinibatches()-i);
	//	int count = 0;
	//	for (int j = 0; j < _dp->getNumMinibatches(); j++)
	//	{
	//		if(mask[j] == 0)
	//		{
	//			if(count == r)
	//			{
	//				mask[j] = 1;
	//				subset.push_back(j);
	//				break;
	//			}
	//			count++;
	//		}
	//	}
	//}

	//assert(subset.size() == subset_size);
	// 
	//_convNet->zeroAuxWeights();

	//int rndGradInd = rand()%subset_size;

	//for (int ki = 0; ki < subset_size; ki++) {

	//	int mb_ind=subset[ki];

	//	float scale = 1./subset_size;
	//	if(ki == rndGradInd)
	//		scale = 1./subset_size - 1;

	//	_convNet->fpropRnd(mb_ind, _epoch, PASS_AUX);
 //       _convNet->bprop(PASS_AUX);
	//	_convNet->procAuxWeights(scale);
 //   }

}

void TrainingWorker::trainingPass(Cost& batchCost) {

bool useAux  = false;//true;

int err_size = 128;
static int error_upd = 0;
static vector<float> test_error;
static float prev_err = 2;
int check_num = 0;
int failure_num = 0;

//int epoch_switch = 90;
	//for (int ki = 0; ki < 1; ki++) {

	for (int ki = 0; ki < _dp->getNumMinibatches(); ki++) {
//		int mini_ind = shaffle[ki];
//debug
minibatch=ki;
//printf("minibatch %i \n", ki);

		_convNet->setParam(_eps_scale);

       //_convNet->fprop(mini_ind, _test ? PASS_TEST : PASS_TRAIN);
		_convNet->fpropRnd(ki, _epoch, _test ? PASS_TEST : PASS_TRAIN);
        _convNet->getCost(batchCost);

		bool successs = true;

		float err = _convNet->getErrorNum()/ _convNet->getNumCases();

		float avg_neg_delta  =0;
		float avg_delta = 0;

		if(test_error.size() >= err_size)
		{
			for(int i = 0; i<test_error.size() ; i++)
			{
				avg_delta += test_error[i];
				if(test_error[i] < 0)
				{
					avg_neg_delta += fabs(test_error[i]);
				}
			}
			avg_delta *= 1./test_error.size();
			avg_neg_delta *= 1./test_error.size();
		}

		if(prev_err <= 1)
		{
			if(test_error.size() < err_size)
				test_error.push_back(prev_err - err);
			else
				test_error[error_upd] = prev_err - err;		
		}

		float scale_rollback_stage0 = 0;
		//if( err-prev_err > avg_neg_delta/2)// && gepoch < epoch_switch)
		//{
		//	successs = false;		
		//	scale_rollback_stage0 =  .25 + .5*fmax(1-( err-prev_err)/avg_neg_delta,0);
		//	failure_num++;
		//}

		if( err-prev_err > 0)// && gepoch < epoch_switch)
		{
			successs = false;		
			scale_rollback_stage0 =  .2 + .5*fmax(1-(err-prev_err)/avg_neg_delta,0);
			failure_num++;
		}


		error_upd = (error_upd+1)%err_size;
		prev_err = err;
       
        if (!_test) {
            _convNet->bprop(PASS_TRAIN);

			if(!successs)
				_convNet->rollbackWeights(scale_rollback_stage0);

            _convNet->updateWeights(useAux);

			//if(useAux)
			//	_convNet->procAuxWeights();

			//if(gepoch >= epoch_switch)
			//{
			//	_convNet->fpropRnd(ki, _epoch, _test ? PASS_TEST : PASS_TRAIN);
			//	_convNet->getCost(batchCost);
			//	float errNew = _convNet->getErrorNum()/ _convNet->getNumCases();

			//	if(errNew-err > 0)
			//	{
			//		float scale_rollback = .2 + .8*fmax(1-(errNew-err)/avg_neg_delta,0);
			//		_convNet->rollbackWeights(scale_rollback);
			//		failure_num++;
			//	}
			//}

        }

//debug aux
		//if(ki > 60)
		//	exit(-1);
    }
	printf("***failures %f \n", 1.*failure_num/ _dp->getNumMinibatches());
}

/*
 * ====================
 * SyncWorker
 * ====================
 */
SyncWorker::SyncWorker(ConvNet& convNet) : Worker(convNet) {
}

void SyncWorker::run() {
    _convNet->copyToCPU();
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::SYNC_DONE));
}

/* 
 * ====================
 * GradCheckWorker
 * ====================
 */
GradCheckWorker::GradCheckWorker(ConvNet& convNet, CPUData& data) 
    : DataWorker(convNet, data) {
}

void GradCheckWorker::run() {
    _dp->setData(*_data);
    _convNet->checkGradients();
    exit(0);
}

/* 
 * ====================
 * MultiviewTestWorker
 * ====================
 */
MultiviewTestWorker::MultiviewTestWorker(ConvNet& convNet, CPUData& data, int numViews, int logregIdx) 
    : DataWorker(convNet, data), _numViews(numViews), _logregIdx(logregIdx) {
    assert(_data->getNumCases() % _numViews == 0);
}

void MultiviewTestWorker::run() {
    _dp->setData(*_data);
    Layer& logregLayer = _convNet->getLayer(_logregIdx);

    int numCasesReal = _dp->getNumCases() / _numViews;
    int numMiniReal = DIVUP(numCasesReal, _dp->getMinibatchSize());
    
    Cost& batchCost = *new Cost(0);
    for (int i = 0; i < numMiniReal; i++) {
        NVMatrix softmaxActs;
        for (int v = 0; v < _numViews; v++) {
            GPUData& mini = _dp->getDataSlice(v * numCasesReal + i * _dp->getMinibatchSize(),
                                              min((v + 1) * numCasesReal, v * numCasesReal + (i + 1) * _dp->getMinibatchSize()));
            _convNet->fprop(mini, PASS_TEST);
            if (v == 0) {
                logregLayer.getPrev()[1]->getActs().copy(softmaxActs);
            } else {
                softmaxActs.add(logregLayer.getPrev()[1]->getActs());
            }
        }
        softmaxActs.scale(1.0 / _numViews);
        NVMatrixV logregInput;
        logregInput.push_back(&logregLayer.getPrev()[0]->getActs());
        logregInput.push_back(&softmaxActs);
        
        logregLayer.fprop(logregInput, PASS_TEST);
        
        _convNet->getCost(batchCost);
    }
    cudaThreadSynchronize();

    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
}

/* 
 * ====================
 * FeatureWorker
 * ====================
 */
FeatureWorker::FeatureWorker(ConvNet& convNet, CPUData& data, Matrix& ftrs, int layerIdx)
    : DataWorker(convNet, data), _ftrs(&ftrs), _layerIdx(layerIdx) {
    assert(ftrs.getNumRows() == data.getNumCases());
    assert(!ftrs.isTrans());
}

FeatureWorker::~FeatureWorker() {
    delete _ftrs;
}

void FeatureWorker::run() {
    _dp->setData(*_data);
    Layer& ftrLayer = _convNet->getLayer(_layerIdx);
    Cost& batchCost = *new Cost(0);
    for (int i = 0; i < _dp->getNumMinibatches(); i++) {

        _convNet->fprop(i, PASS_TEST);
        _convNet->getCost(batchCost);
        Matrix& miniFtrs = _ftrs->sliceRows(i * _dp->getMinibatchSize(),
                                            min(_dp->getNumCases(), (i + 1) * _dp->getMinibatchSize()));
        NVMatrix& acts = ftrLayer.getActs();
        NVMatrix acts_T;
        if (acts.isTrans()) {
            NVMatrix& soft_T = acts.getTranspose();
            soft_T.transpose(acts_T);
            delete &soft_T;
        } else {
            acts.transpose(acts_T);
        }
        acts_T.copyToHost(miniFtrs);
        delete &miniFtrs;
    }
    cudaThreadSynchronize();
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
}