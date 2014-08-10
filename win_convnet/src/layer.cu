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
#include <cutil_inline.h>
#include <iostream>

#include <layer_kernels.cuh>
#include <layer.cuh>
#include <data.cuh>
#include <util.cuh>
#include <cudaconv2.cuh>
#include <matrix.h>

using namespace std;

/* 
 * =======================
 * Layer
 * =======================
 */

Layer::Layer(ConvNet* convNet, PyObject* paramsDict, bool trans) : 
             _convNet(convNet),  _trans(trans), _dropout(0){
    _name = pyDictGetString(paramsDict, "name");
    _type = pyDictGetString(paramsDict, "type");
    
    _numGradProducersNext = 0;
    _foundGradConsumers = false;
    _gradConsumer = pyDictGetInt(paramsDict, "gradConsumer");
    _actsTarget = pyDictGetInt(paramsDict, "actsTarget");
    _actsGradTarget = pyDictGetInt(paramsDict, "actsGradTarget");
    _conserveMem = pyDictGetInt(paramsDict, "conserveMem");
    _outputs = _actsTarget < 0 ? new NVMatrix() : NULL;
    _actsGrad = _actsGradTarget < 0 ? new NVMatrix() : NULL;

    _dropout = pyDictGetFloat(paramsDict, "dropout");
    //_dropout_mask = new NVMatrix(); error?

	_nan2Zero = false;
}

void Layer::fpropNext(PASS_TYPE passType) {
    for (int i = 0; i < _next.size(); i++) {
        _next[i]->fprop(passType);
    }
}

void Layer::truncBwdActs() {
    // Only truncate actsGrad if I own it
    if (_conserveMem && _actsGradTarget < 0) { 
        getActsGrad().truncate();
    }
    if (_conserveMem) {
        getActs().truncate();
    }
}

void Layer::fprop(PASS_TYPE passType) {
    _rcvdFInputs += 1;
    if (_rcvdFInputs == _prev.size()) {
        NVMatrixV v;
        for (int i = 0; i < _prev.size(); i++) {
            v.push_back(&_prev[i]->getActs());
        }
        fprop(v, passType);
    }
}

//void Layer::fprop(NVMatrix& v, PASS_TYPE passType) {
//    NVMatrixV vl;
//    vl.push_back(&v);
//    fprop(vl, passType);
//}
void Layer::setParam(float eps_scale)
{

	setCommon(eps_scale);

	setParamNext(eps_scale);
}

void Layer::setParamNext(float eps_scale) {
    for (int i = 0; i < _next.size(); i++) {
        _next[i]->setParam(eps_scale);
    }
}

void Layer::fprop(NVMatrixV& v, PASS_TYPE passType) {
    assert(v.size() == _prev.size());
    _inputs.clear();
    _inputs.insert(_inputs.begin(), v.begin(), v.end());
    _outputs = _actsTarget < 0 ? _outputs : _inputs[_actsTarget];
    _rcvdFInputs = _prev.size();
    for (NVMatrixV::iterator it = v.begin(); it != v.end(); ++it) {
        (*it)->transpose(_trans);
    }
    getActs().transpose(_trans);

    // First do fprop on the input whose acts matrix I'm sharing, if any
    if (_actsTarget >= 0) {
        fpropActs(_actsTarget, 0, passType);
    }
    // Then add the rest of the inputs to that
    for (int i = 0; i < _prev.size(); i++) {
        if (i != _actsTarget) {
            fpropActs(i, _actsTarget >= 0 || i > 0, passType);
        }
    }

    if (passType != PASS_TEST && _dropout > 0.0) {
        _dropout_mask.resize(getActs().getNumRows(), getActs().getNumCols());
        _dropout_mask.randomizeUniform();
        _dropout_mask.biggerThanScalar(_dropout);
        getActs().eltwiseMult(_dropout_mask);
    }
      
    if (passType == PASS_TEST && _dropout > 0.0) {
        getActs().scale(1.0 - _dropout);
    }

    fpropNext(passType);
}

void Layer::bprop(PASS_TYPE passType) {
    if (_rcvdBInputs == _numGradProducersNext) {
        _rcvdBInputs++; // avoid doing bprop computation twice
        bprop(getActsGrad(), passType);
    }
}

void Layer::bprop(NVMatrix& v, PASS_TYPE passType) {
	//v = getActsGrad() from next
    v.transpose(_trans);
    for (int i = 0; i < _prev.size(); i++) {
        _prev[i]->getActs().transpose(_trans);
        _prev[i]->getActsGrad().transpose(_trans);
    }
    getActs().transpose(_trans);
    
    if (_dropout > 0.0) {
      v.eltwiseMult(_dropout_mask);
    }

    bpropCommon(v, passType);
    
    if (isGradProducer()) {
        // First propagate activity gradient to all layers whose activity
        // gradient matrix I'm definitely not sharing.
        for (int i = 0; i < _prev.size(); i++) {
            if (_prev[i]->isGradConsumer() && _actsGradTarget != i) {
                bpropActs(v, i, _prev[i]->getRcvdBInputs() > 0 ? 1 : 0, passType);
                _prev[i]->incRcvdBInputs();
            }
        }
        // Then propagate activity gradient to the layer whose activity gradient
        // matrix I'm sharing, if any.
        if (_actsGradTarget >= 0 && _prev[_actsGradTarget]->isGradConsumer()) {
            bpropActs(v, _actsGradTarget, _prev[_actsGradTarget]->getRcvdBInputs() > 0 ? 1 : 0, passType);
            _prev[_actsGradTarget]->incRcvdBInputs();
        }
    }
    truncBwdActs();
    
    if (isGradProducer()) {
        for (int i = 0; i < _prev.size(); i++) {
            if (_prev[i]->isGradConsumer()) {
                _prev[i]->bprop(passType);
            }
        }
    }
}

void Layer::reset() {
    _rcvdFInputs = 0;
    _rcvdBInputs = 0;
}

string& Layer::getName() {
    return _name;
}

string& Layer::getType() {
    return _type;
}

int Layer::getRcvdFInputs() {
    return _rcvdFInputs;
}

int Layer::getRcvdBInputs() {
    return _rcvdBInputs;
}

int Layer::incRcvdBInputs() {
    return ++_rcvdBInputs;
}

void Layer::addNext(Layer* l) {
    _next.push_back(l);
    _numGradProducersNext += l->isGradProducer();
}

void Layer::addPrev(Layer* l) {
    _prev.push_back(l);
}

void Layer::postInit() {
//    _outputs = _actsTarget < 0 ? new NVMatrix() : &_prev[_actsTarget]->getActs();
    _actsGrad = _actsGradTarget < 0 ? new NVMatrix() : &_prev[_actsGradTarget]->getActsGrad();
}

// Does this layer, or some layer below it, need the gradient
// for parameter updates?
// Only weight layers should be grad consumers themselves.
bool Layer::isGradConsumer() {
    if (!_foundGradConsumers) {
        for (int i = 0; i < _prev.size(); i++) {
            _gradConsumer |= _prev[i]->isGradConsumer();
        }
        _foundGradConsumers = true;
    }
    return _gradConsumer;
}

// Does this layer produce gradient for layers below?
bool Layer::isGradProducer() {
    return true;
}

vector<Layer*>& Layer::getPrev() {
    return _prev;
}

vector<Layer*>& Layer::getNext() {
    return _next;
}

NVMatrix& Layer::getActs() {
    assert(_outputs != NULL);
    return *_outputs;
}

NVMatrix& Layer::getActsGrad() {
    assert(_actsGrad != NULL);
    return *_actsGrad;
}

/* 
 * =======================
 * NeuronLayer
 * =======================
 */
NeuronLayer::NeuronLayer(ConvNet* convNet, PyObject* paramsDict) 
    : Layer(convNet, paramsDict, true) {
    _neuron = &Neuron::makeNeuron(PyDict_GetItemString(paramsDict, "neuron"));
}

void NeuronLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _neuron->computeInputGrad(v, _prev[0]->getActsGrad(), scaleTargets > 0);
}

void NeuronLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _neuron->activate(*_inputs[0], getActs());
}

/* 
 * =======================
 * WeightLayer
 * =======================
 */
WeightLayer::WeightLayer(ConvNet* convNet, PyObject* paramsDict, bool trans, bool useGrad) : 
    Layer(convNet, paramsDict, trans) {
 
    MatrixV& hWeights = *pyDictGetMatrixV(paramsDict, "weights");
    MatrixV& hWeightsInc = *pyDictGetMatrixV(paramsDict, "weightsInc");
    Matrix& hBiases = *pyDictGetMatrix(paramsDict, "biases");
    Matrix& hBiasesInc = *pyDictGetMatrix(paramsDict, "biasesInc");

	string layerType = pyDictGetString(paramsDict, "type");
	//printf("WeightLayer %s \n", _name.c_str());


	MatrixV *phBregman_b_weights = NULL;
	Matrix *phBregman_b_bias = NULL;

//bregman
	if(layerType == "fc")
	{
		phBregman_b_weights = pyDictGetMatrixV(paramsDict, "b_weight_bregman");
	    phBregman_b_bias = pyDictGetMatrix(paramsDict, "b_bias_bregman");
	}
    
    floatv& momW = *pyDictGetFloatV(paramsDict, "momW");
    float momB = pyDictGetFloat(paramsDict, "momB");
    floatv& epsW = *pyDictGetFloatV(paramsDict, "epsW");
    float epsB = pyDictGetFloat(paramsDict, "epsB");
    floatv& wc = *pyDictGetFloatV(paramsDict, "wc");

	float muL1 = pyDictGetFloat(paramsDict, "muL1");

	_renorm = pyDictGetFloat(paramsDict, "renorm");
    
    // Source layers for shared weights
    intv& weightSourceLayerIndices = *pyDictGetIntV(paramsDict, "weightSourceLayerIndices");
    // Weight matrix indices (inside the above source layers) for shared weights
    intv& weightSourceMatrixIndices = *pyDictGetIntV(paramsDict, "weightSourceMatrixIndices");
    
    for (int i = 0; i < weightSourceLayerIndices.size(); i++) {
        int srcLayerIdx = weightSourceLayerIndices[i];
        int matrixIdx = weightSourceMatrixIndices[i];
        if (srcLayerIdx == convNet->getNumLayers()) { // Current layer
            _weights.addWeights(*new Weights(_weights[matrixIdx], epsW[i]));
        } else if (srcLayerIdx >= 0) {
            WeightLayer& srcLayer = *static_cast<WeightLayer*>(&convNet->getLayer(srcLayerIdx));
            Weights* srcWeights = &srcLayer.getWeights(matrixIdx);
            _weights.addWeights(*new Weights(*srcWeights, epsW[i]));

        } else if(layerType != "fc"){
            _weights.addWeights(*new Weights(*hWeights[i], *hWeightsInc[i], epsW[i], wc[i], momW[i], muL1, _renorm, useGrad));
        } else
			_weights.addWeights(*new Weights(*hWeights[i], *hWeightsInc[i],
			*((*phBregman_b_weights)[i]),
			epsW[i], wc[i], momW[i], muL1, _renorm, useGrad));

    }
    if(layerType != "fc")
		_biases = new Weights(hBiases, hBiasesInc, epsB, 0, momB, 0, 0, true);
	else
		_biases = new Weights(hBiases, hBiasesInc,
		*phBregman_b_bias,
		epsB, 0, momB, 0, 0, true);

    // Epsilons for finite-difference gradient checking operation
    _wStep = 0.001;
    _bStep = 0.002;
    
    delete &weightSourceLayerIndices;
    delete &weightSourceMatrixIndices;
    delete &hWeights;
    delete &hWeightsInc;
    delete &momW;
    delete &epsW;
    delete &wc;
}
void WeightLayer::setCommon(float eps_scale) {
    for (int i = 0; i < _weights.getSize(); i++) {
		if(eps_scale > 0)
			_weights[i].setEps(_weights[i].getEpsInit()*eps_scale);
	}
}

void WeightLayer::bpropCommon(NVMatrix& v, PASS_TYPE passType) {
    if (_biases->getEps() > 0) {
        bpropBiases(v, passType);
    }
    for (int i = 0; i < _weights.getSize(); i++) {
        if (_weights[i].getEps() > 0) {
            bpropWeights(v, i, passType);
            // Increment its number of updates
            _weights[i].incNumUpdates();
        }
    }
}

void WeightLayer::updateWeights() {

	bool use_inc_drop = false;
	//if(_type == "conv")
	//	use_inc_drop = true;

//debug shrink
//	if(_name == "fc128_1" || _name == "fc128_2" || _name == "fc128_3")
//		_weights.shrink(1000*1000);

    _weights.update(use_inc_drop);
    _biases->update(false);
    
}

void WeightLayer::copyToCPU() {
    _weights.copyToCPU();
    _biases->copyToCPU();
}

void WeightLayer::copyToGPU() {
    _weights.copyToGPU();
    _biases->copyToGPU();
}

void WeightLayer::checkGradients() {
    for (int i = 0; i < _weights.getSize(); i++) {
        _convNet->checkGradient(_name + " weights[" + tostr(i) + "]", _wStep, _weights[i]);
    }
    _convNet->checkGradient(_name + " biases", _bStep, *_biases);
}

Weights& WeightLayer::getWeights(int idx) {
    return _weights[idx];
}

/* 
 * =======================
 * FCLayer
 * =======================
 */
extern int train;//temp
extern  int gmini;//temp
extern  int gmini_max;//temp
//#define show_mini gmini
#define show_mini (gmini_max-1)
//#define show_mini 1

FCLayer::FCLayer(ConvNet* convNet, PyObject* paramsDict) : WeightLayer(convNet, paramsDict, true, false) {
    _wStep = 0.1;
    _bStep = 0.01;
}

void FCLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    getActs().addProduct(*_inputs[inpIdx], *_weights[inpIdx], scaleTargets, 1);
    if (scaleTargets == 0) {
        getActs().addVector(_biases->getW());
    }
}

void FCLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	//v = getActsGrad() from next
    NVMatrix& weights_T = _weights[inpIdx].getW().getTranspose();
    _prev[inpIdx]->getActsGrad().addProduct(v, weights_T, scaleTargets, 1);

    delete &weights_T;
}

void FCLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    int numCases = v.getNumRows();
    float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
    _biases->getGrad().addSum(v, 0, 0, scaleBGrad);
}

void FCLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
	//v = getActsGrad() from next

    int numCases = v.getNumRows();

    NVMatrix& prevActs_T = _prev[inpIdx]->getActs().getTranspose();
    float scaleInc = (_weights[inpIdx].getNumUpdates() == 0 && passType != PASS_GC) * _weights[inpIdx].getMom();
    float scaleGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
    
	// Df_next(sum f_prew_k*w_k)/Dw_i = (Df_next(x)/Dx)*f_prev_i
    _weights[inpIdx].getInc().addProduct(prevActs_T, v, scaleInc, scaleGrad);
    
    delete &prevActs_T;
}

/* 
 * =======================
 * LocalLayer
 * =======================
 */
LocalLayer::LocalLayer(ConvNet* convNet, PyObject* paramsDict, bool useGrad) 
    : WeightLayer(convNet, paramsDict, false, useGrad) {
    _padding = pyDictGetIntV(paramsDict, "padding");
    _stride = pyDictGetIntV(paramsDict, "stride");
    _filterSize = pyDictGetIntV(paramsDict, "filterSize");
    _channels = pyDictGetIntV(paramsDict, "channels");
    _imgSize = pyDictGetIntV(paramsDict, "imgSize");
    _numFilters = pyDictGetInt(paramsDict, "filters");
    _groups = pyDictGetIntV(paramsDict, "groups");
    _filterChannels = pyDictGetIntV(paramsDict, "filterChannels");
    _randSparse = pyDictGetIntV(paramsDict, "randSparse");
    _overSample = pyDictGetIntV(paramsDict, "overSample");
    _filterPixels = pyDictGetIntV(paramsDict, "filterPixels");
    _imgPixels = pyDictGetIntV(paramsDict, "imgPixels");
    
    _modulesX = pyDictGetInt(paramsDict, "modulesX");
    _modules = pyDictGetInt(paramsDict, "modules");

    // It's a vector on the heap to be consistent with all the others...
    _filterConns = new vector<FilterConns>();
    PyObject* pyFilterConns = PyDict_GetItemString(paramsDict, "filterConns");
    for (int i = 0; i < _randSparse->size(); i++) {
        FilterConns fc;
        if (_randSparse->at(i)) {
            fc.hFilterConns = getIntA(PyList_GET_ITEM(pyFilterConns, i));
        }
        _filterConns->push_back(fc);
    }
}

void LocalLayer::copyToGPU() {
    WeightLayer::copyToGPU();
    for  (int i = 0; i < _prev.size(); i++) {
        if (_randSparse->at(i)) { // Copy to GPU vector that describes sparse random connectivity
            cudaMalloc(&_filterConns->at(i).dFilterConns, sizeof(int) * _groups->at(i) * _filterChannels->at(i));
            cudaMemcpy(_filterConns->at(i).dFilterConns, _filterConns->at(i).hFilterConns,
                       sizeof(int) * _groups->at(i) * _filterChannels->at(i), cudaMemcpyHostToDevice);
            cutilCheckMsg("cudaMemcpy: failed");
        }
    }
}

/* 
 * =======================
 * ConvLayer
 * =======================
 */
ConvLayer::ConvLayer(ConvNet* convNet, PyObject* paramsDict) : LocalLayer(convNet, paramsDict, true) {
    _partialSum = pyDictGetInt(paramsDict, "partialSum");
    _sharedBiases = pyDictGetInt(paramsDict, "sharedBiases");
}

void ConvLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {

//debug
	//printf(" conv fprop name %s \n", _name.c_str());


    if (_randSparse->at(inpIdx)) {
        convFilterActsSparse(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _filterConns->at(inpIdx).dFilterConns,
                             _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                             _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        convFilterActs(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx),
                       _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }
//debug
//getActs().nan2zero();
   
    if (scaleTargets == 0) {
        if (_sharedBiases) {
            getActs().reshape(_numFilters, getActs().getNumElements() / _numFilters);
            getActs().addVector(_biases->getW());
            getActs().reshape(_numFilters * _modules, getActs().getNumElements() / (_numFilters * _modules));
        } else {
            getActs().addVector(_biases->getW());
        }
    }
}

void ConvLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    int numCases = v.getNumCols();
    float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
    if (_sharedBiases) {
        v.reshape(_numFilters, v.getNumElements() / _numFilters);
        _biases->getGrad().addSum(v, 1, 0, scaleBGrad);
        v.reshape(_numFilters * _modules, v.getNumElements() / (_numFilters * _modules));
    } else {
        _biases->getGrad().addSum(v, 1, 0, scaleBGrad);
    }
}

void ConvLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumCols();

    NVMatrix& tgt = _partialSum > 0 ? _weightGradTmp : _weights[inpIdx].getGrad();
    float scaleWGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
    float scaleTargets = _weights[inpIdx].getNumUpdates() > 0 && _partialSum == 0; // ? 1 : 0;
    if (_randSparse->at(inpIdx)) {
        convWeightActsSparse(_prev[inpIdx]->getActs(), v, tgt, _filterConns->at(inpIdx).dFilterConns, _imgSize->at(inpIdx), _modulesX, _modulesX,
                             _filterSize->at(inpIdx), _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                             _filterChannels->at(inpIdx), _groups->at(inpIdx), _partialSum, scaleTargets, scaleWGrad);
    } else {
        convWeightActs(_prev[inpIdx]->getActs(), v, tgt, _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx), _padding->at(inpIdx),
                       _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), _partialSum, scaleTargets, scaleWGrad);
    }
    if (_partialSum > 0) {
        scaleTargets = _weights[inpIdx].getNumUpdates() > 0;
        _weightGradTmp.reshape(_modules / _partialSum, _filterChannels->at(inpIdx) * _filterPixels->at(inpIdx) * _numFilters);
        _weights[inpIdx].getGrad().addSum(_weightGradTmp, 0, scaleTargets, 1);
        _weights[inpIdx].getGrad().reshape(_filterChannels->at(inpIdx) * _filterPixels->at(inpIdx), _numFilters);
    }

	if(_nan2Zero) {
		_weights[inpIdx].getGrad().nan2zero();//nan fix  
	}

}

void ConvLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {

//debug
	//printf(" conv bpropActs name %s \n", _name.c_str());

    if (_randSparse->at(inpIdx)) {
        NVMatrix& tgt = _overSample->at(inpIdx) > 1 ? _actGradTmp : _prev[inpIdx]->getActsGrad();
        convImgActsSparse(v, *_weights[inpIdx], tgt, _filterConns->at(inpIdx).dFilterConns,
                          _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX, _padding->at(inpIdx), _stride->at(inpIdx),
                          _channels->at(inpIdx), _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
        if (_overSample->at(inpIdx) > 1) {
            _actGradTmp.reshape(_overSample->at(inpIdx), _actGradTmp.getNumElements() / _overSample->at(inpIdx));
            _actGradTmp.sum(0, _prev[inpIdx]->getActsGrad());
            _prev[inpIdx]->getActsGrad().reshape(_prev[inpIdx]->getActsGrad().getNumElements() / v.getNumCols(), v.getNumCols());
        }
    } else {
        convImgActs(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(), _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
                    _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }
}

void ConvLayer::truncBwdActs() {
    LocalLayer::truncBwdActs();
    if (_conserveMem) {
        _weightGradTmp.truncate();
        _actGradTmp.truncate();
    }
}
/* 
 * =======================
 * LocalUnsharedLayer
 * =======================
 */
LocalUnsharedLayer::LocalUnsharedLayer(ConvNet* convNet, PyObject* paramsDict) : LocalLayer(convNet, paramsDict, false) {
}

void LocalUnsharedLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        localFilterActsSparse(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _filterConns->at(inpIdx).dFilterConns,
                              _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                              _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        localFilterActs(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx),
                        _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);

    }  
    if (scaleTargets == 0) {
        getActs().addVector(_biases->getW());
    }
}

void LocalUnsharedLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    int numCases = v.getNumCols();
    float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
    _biases->getGrad().addSum(v, 1, 0, scaleBGrad);
}

void LocalUnsharedLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumCols();
    
    float scaleInc = (passType != PASS_GC && _weights[inpIdx].getNumUpdates() == 0) * _weights[inpIdx].getMom(); // momentum
    float scaleWGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases; // eps / numCases
    if (_randSparse->at(inpIdx)) {
        localWeightActsSparse(_prev[inpIdx]->getActs(), v, _weights[inpIdx].getInc(), _filterConns->at(inpIdx).dFilterConns,
                              _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx), _padding->at(inpIdx), _stride->at(inpIdx),
                              _channels->at(inpIdx), _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleInc, scaleWGrad);
    } else {
        localWeightActs(_prev[inpIdx]->getActs(), v, _weights[inpIdx].getInc(), _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx),
                        _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleInc, scaleWGrad);
    }
}

void LocalUnsharedLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        localImgActsSparse(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(), _filterConns->at(inpIdx).dFilterConns,
                           _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                           _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        localImgActs(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(),_imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
                    _padding->at(inpIdx),  _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }
}

/* 
 * =======================
 * SoftmaxLayer
 * =======================
 */
SoftmaxLayer::SoftmaxLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, true) {
}

void SoftmaxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& input = *_inputs[0];
    NVMatrix& max = input.max(1);
    input.addVector(max, -1, getActs());
    getActs().apply(NVMatrixOps::Exp());
    NVMatrix& sum = getActs().sum(1);
    getActs().eltwiseDivideByVector(sum);

    delete &max;
    delete &sum;
}

void SoftmaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 0);
    bool doLogregGrad = _next.size() == 1 && ( _next[0]->getType() == "cost.logreg");
	bool doRLogGrad = _next.size() == 1 && ( _next[0]->getType() == "cost.rlog" );
    if (doLogregGrad) {
        NVMatrix& labels = _next[0]->getPrev()[0]->getActs();
        float gradCoeff = dynamic_cast<CostLayer*>(_next[0])->getCoeff();
        computeLogregSoftmaxGrad(labels, getActs(), _prev[0]->getActsGrad(), scaleTargets == 1, gradCoeff);
    } else if (doRLogGrad) {
        NVMatrix& labels = _next[0]->getPrev()[0]->getActs();
        float gradCoeff = dynamic_cast<CostLayer*>(_next[0])->getCoeff();
		NVMatrix* probWeights = dynamic_cast<RLogCostLayer*>(_next[0])->GetProbWeights();

        computeRLogSoftmaxGrad(labels, getActs(), _prev[0]->getActsGrad(), *probWeights, scaleTargets == 1, gradCoeff);

	} else {
        computeSoftmaxGrad(getActs(), v, _prev[0]->getActsGrad(), scaleTargets == 1);
    }
}

/* 
 * =======================
 * L2SVMLayer
 * =======================
 */

L2SVMLayer::L2SVMLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, true) {
}

void L2SVMLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& input = *_inputs[0];
    input.copy(getActs());
}


void L2SVMLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 0);

    NVMatrix& labels = _next[0]->getPrev()[0]->getActs();
    float gradCoeff = dynamic_cast<CostLayer*>(_next[0])->getCoeff();

    computeL2SVMGrad(labels, getActs(), _prev[0]->getActsGrad(), scaleTargets == 1, gradCoeff);
}

/* 
 * =======================
 * EltwiseSumLayer
 * =======================
 */
EltwiseSumLayer::EltwiseSumLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _coeffs = pyDictGetFloatV(paramsDict, "coeffs");
}

void EltwiseSumLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (scaleTargets == 0) {
        _inputs[inpIdx]->scale(_coeffs->at(inpIdx), getActs());
    } else {
        getActs().add(*_inputs[inpIdx], _coeffs->at(inpIdx));
    }
}

void EltwiseSumLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (scaleTargets == 0 ) {
        v.scale(_coeffs->at(inpIdx), _prev[inpIdx]->getActsGrad());
    } else {
        assert(&_prev[inpIdx]->getActsGrad() != &v);
        _prev[inpIdx]->getActsGrad().add(v, scaleTargets, _coeffs->at(inpIdx));
    }
}

/* 
 * =======================
 * EltwiseMaxLayer
 * =======================
 */
EltwiseMaxLayer::EltwiseMaxLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
}

void EltwiseMaxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (inpIdx == 1) { // First input, do nothing
        _inputs[inpIdx]->applyBinary(NVMatrixAggs::Max(), *_inputs[0], getActs());
    } else if (inpIdx > 1) {
        getActs().applyBinary(NVMatrixAggs::Max(), *_inputs[inpIdx]);
    }
}

void EltwiseMaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    computeEltwiseMaxGrad(v, *_inputs[inpIdx], getActs(), _prev[inpIdx]->getActsGrad(), scaleTargets != 0);
}
/* 
 * =======================
 * MicroConvLayer
 * =======================
 */
MicroConvLayer::MicroConvLayer(ConvNet* convNet, PyObject* paramsDict): Layer(convNet, paramsDict, false) 
{
	hParamList = PyDict_GetItemString(paramsDict, "meta_param");
	_param = getVectorDouble(hParamList);

	hParamListInc = PyDict_GetItemString(paramsDict, "meta_param_inc");
	_param_inc = getVectorDouble(hParamListInc);

	 _size = pyDictGetInt(paramsDict, "size");
    _channels = pyDictGetInt(paramsDict, "channels");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
    _numFilters = pyDictGetInt(paramsDict, "filters");
    _imgPixels = pyDictGetInt(paramsDict, "imgPixels");
    
    _mom = pyDictGetFloat(paramsDict, "mom");
    _epsP = pyDictGetFloat(paramsDict, "epsP");
    _wc = pyDictGetFloat(paramsDict, "wc");
//debug
	memset(_nstore_count, 0, sizeof(_nstore_count));
	for (int i =0; i < NSTORE; i++)
	for (int j =0; j < _param.size(); j++)
		_grad_store[i].push_back(0);
	//printf(" _param init  %f %f %f \n", _param[2] , _param[_sizeIn + 0] , _param[_sizeIn + 1]);
	//printf(" size_in %i size_out %i  updates %i \n",_sizeIn, _sizeOut, _updates);

};

void MicroConvLayer::copyToCPU()
{
	for(int i = 0; i < _param.size(); i++)
	{
		PyList_SetItem(hParamList, i,  PyFloat_FromDouble(_param[i]));
		PyList_SetItem(hParamListInc, i,  PyFloat_FromDouble(_param_inc[i]));
	}
};
//debug
extern int minibatch;
void MicroConvLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
	//printf(" MicroConvLayer fpropActs minibatch %i \n", minibatch);

	computeMicroConvAct(*_inputs[inpIdx],  getActs(), _param, _size, _channels, _imgSize, _imgPixels, _numFilters);

	//printf(" MicroConvLayer fpropActs end\n");
}

/* 
 * =======================
 * EltwiseFuncLayer
 * =======================
 */
EltwiseFuncLayer::EltwiseFuncLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {

	hParamList = PyDict_GetItemString(paramsDict, "meta_param");
	_param = getVectorDouble(hParamList);

	hParamListInc = PyDict_GetItemString(paramsDict, "meta_param_inc");
	_param_inc = getVectorDouble(hParamListInc);

    _sizeIn = pyDictGetInt(paramsDict, "size_in");
	_sizeOut = pyDictGetInt(paramsDict, "size_out");
	_updates = pyDictGetInt(paramsDict, "updates");
    
    _mom = pyDictGetFloat(paramsDict, "mom");
    _epsP = pyDictGetFloat(paramsDict, "epsP");
    _wc = pyDictGetFloat(paramsDict, "wc");
//debug
	memset(_nstore_count, 0, sizeof(_nstore_count));
	for (int i =0; i < NSTORE; i++)
	for (int j =0; j < _param.size(); j++)
		_grad_store[i].push_back(0);
	//printf(" _param init  %f %f %f \n", _param[2] , _param[_sizeIn + 0] , _param[_sizeIn + 1]);
	//printf(" size_in %i size_out %i  updates %i \n",_sizeIn, _sizeOut, _updates);

}

void EltwiseFuncLayer::copyToCPU()
{
	for(int i = 0; i < _param.size(); i++)
	{
		PyList_SetItem(hParamList, i,  PyFloat_FromDouble(_param[i]));
		PyList_SetItem(hParamListInc, i,  PyFloat_FromDouble(_param_inc[i]));
	}
};


void EltwiseFuncLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
	//printf(" EltwiseFuncLayer fpropActs minibatch %i \n", minibatch);

	computeEltwiseFuncAct(*_inputs[inpIdx],  getActs(), _param, _sizeIn, _sizeOut);

	//printf(" EltwiseFuncLayer fpropActs end\n");
}

//debug
extern int gepoch;

void EltwiseFuncLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {

	static int pin_prev = 0;
	static int pout_prev = 0;

	int pin = (pin_prev + 1 + rand()%2)%_sizeIn;
	int pout = rand()%_sizeOut;

	pin_prev = pin;
	pout_prev = pout;
	//printf(" bpropActs \n");
	//if(gepoch >= 60)
	{
		computeEltwiseFuncParamGradSingle(v, *_inputs[inpIdx],
									  _temp, _temp_m,
									 pin, pout,  _sizeIn, _sizeOut);

//	printf(" temp %i pin %i pout %i  \n", temp.getNumElements(), pin, pout);
	

//	printf(" temp_m %i  \n", temp_m.getNumElements());

		double grad = _temp.sum();
		double grad_m = _temp_m.sum();
		int ind_p = pin + pout*2*_sizeIn;
		int ind_p_m = ind_p + _sizeIn;

	//printf(" grad %f grad_m %f _epsP %f   \n", grad, grad_m, _epsP);
//debug


		double sum = 0, ge =0, ge_m =0;
		for(int k = 0; k < NSTORE; k++)
			sum += _grad_store[k][ind_p]*_grad_store[k][ind_p];

		double sum_m = 0;
		for(int k = 0; k < NSTORE; k++)
			sum_m += _grad_store[k][ind_p_m]*_grad_store[k][ind_p_m];

		if(sum > 0)
			ge = grad*sqrt(NSTORE)/sqrt(sum);
		if(sum_m > 0)
			ge_m = grad_m*sqrt(NSTORE)/sqrt(sum_m);

		_grad_store[_nstore_count[ind_p]][ind_p] = grad;
		_grad_store[_nstore_count[ind_p_m]][ind_p_m] = grad_m;
		_nstore_count[ind_p] = (_nstore_count[ind_p]+1)%NSTORE;
		_nstore_count[ind_p_m] = (_nstore_count[ind_p_m]+1)%NSTORE;

		_param_inc[ind_p] = _mom*_param_inc[ind_p] + _epsP*ge - _wc*_param[ind_p];
		_param_inc[ind_p_m] = _mom*_param_inc[ind_p_m] + _epsP*ge_m - _wc*_param[ind_p_m];

		//printf(" _param_inc %f _param_inc_M %f   \n", _param_inc[ind_p], _param_inc[ind_p_m]);

		_param[ind_p] += _param_inc[ind_p];
		_param[ind_p_m] += _param_inc[ind_p_m];


	//renormalize
		double sumScale = 3;
		double l1sum = 0;
		for(int i =0; i < _param.size(); i++)
			l1sum += fabs(_param[i]);

		for(int i =0; i < _param.size(); i++)
			_param[i] *= sumScale/l1sum;
		////debug
		if(minibatch == 0)
		{
			int numl = (_param.size()+8)/8;
			printf("** params *** \n");
			for (int nk = 0; nk < numl; nk++)
			{
				for (int k = 0; k < 8; k++)
					if(k + nk*8 < _param.size())
					printf("%f ", _param[k + nk*8]);
				printf("\n");
			}
		}
	}



//
//
//		NVMatrix temp0, temp1, temp2, temp3, temp4, temp5;
//		computeEltwiseFuncParamGrad(v, *_inputs[0], *_inputs[1], *_inputs[2],
//		 temp0,  temp1,  temp2, temp3,  temp4, temp5);
//
//		double grad[6];
//		grad[0] = temp0.sum();
//		grad[1] = temp1.sum();
//		grad[2] = temp2.sum();
//		grad[3] = temp3.sum();
//		grad[4] = temp4.sum();
//		grad[5] = temp5.sum();
//		double mom = .9;
//		double eps = 1e-5;
//		double wc = 1e-5;
//
//		for(int i = 0; i < 6; i++)
//		{
//			if(rand()%6 == 0)
//				continue;
//			_param_inc[i] = _mom*_param_inc[i] + _epsP*grad[i] - _wc*_param[i];
//		}
//
////debug
//		//if(minibatch==0)
//		//{
//		//	printf(" _param_inc %f %f %f %f %f \n", _param_inc[0] , _param_inc[1] , _param_inc[2]  , _param_inc[3]  , _param_inc[4]);
//		//}
//
//		for(int i = 0; i < 6; i++)
//		{
//			_param[i] += _param_inc[i];
//			_param[i] = fmin(fmax(_param[i], -1), 1);
//		}
//
//		if(minibatch==0)
//		{
//			//printf(" grads %f %f %f %f %f \n", grad[0] , grad[1] , grad[2], grad[3], grad[4]);
//			printf(" _param after %f %f %f  %f %f %f\n", _param[0] , _param[1] , _param[2] , _param[3] , _param[4], _param[5]);
//
//		}


    int inp_width = ( *_inputs[inpIdx]).getNumCols(); 
    int inp_height = ( *_inputs[inpIdx]).getNumRows();

NVMatrix& target = _prev[inpIdx]->getActsGrad();
    if (target.getNumCols() != inp_width || target.getNumRows() != inp_height) {
        target.resize(inp_height, inp_width);
    }

		computeEltwiseFuncGrad(v, *_inputs[inpIdx], _prev[inpIdx]->getActsGrad(), _param, _sizeIn, _sizeOut);

		//printf(" bpropActs end\n");
}


/* 
 * =======================
 * DataLayer
 * =======================
 */
DataLayer::DataLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _dataIdx = pyDictGetInt(paramsDict, "dataIdx");
}

void DataLayer::fprop(PASS_TYPE passType) {
    throw string("No dava given!");
}

void DataLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
}

void DataLayer::fprop(NVMatrixV& data, PASS_TYPE passType) {
    _outputs = data[_dataIdx];
    fpropNext(passType);
}

bool DataLayer::isGradProducer() {
    return false;
}

/* 
 * =====================
 * PoolLayer
 * =====================
 */
PoolLayer::PoolLayer(ConvNet* convNet, PyObject* paramsDict, bool trans) 
    : Layer(convNet, paramsDict, trans) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _sizeX = pyDictGetInt(paramsDict, "sizeX");
    _start = pyDictGetInt(paramsDict, "start");
    _stride = pyDictGetInt(paramsDict, "stride");
    _outputsX = pyDictGetInt(paramsDict, "outputsX");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
    _pool = pyDictGetString(paramsDict, "pool");
}

PoolLayer& PoolLayer::makePoolLayer(ConvNet* convNet, PyObject* paramsDict) {
    string _pool = pyDictGetString(paramsDict, "pool");
    if (_pool == "max") {
        return *new MaxPoolLayer(convNet, paramsDict);
    } else if(_pool == "avg") {
        return *new AvgPoolLayer(convNet, paramsDict);
    }
    throw string("Unknown pooling layer type ") + _pool;
}

/* 
 * =====================
 * AvgPoolLayer
 * =====================
 */
AvgPoolLayer::AvgPoolLayer(ConvNet* convNet, PyObject* paramsDict) : PoolLayer(convNet, paramsDict, false) {
}

void AvgPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, AvgPooler());
}

void AvgPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalAvgUndo(v, _prev[0]->getActsGrad(), _sizeX, _start, _stride, _outputsX, _imgSize, scaleTargets, 1);
}

/* 
 * =====================
 * MaxPoolLayer
 * =====================
 */
MaxPoolLayer::MaxPoolLayer(ConvNet* convNet, PyObject* paramsDict) : PoolLayer(convNet, paramsDict, false) {
}

void MaxPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, MaxPooler());
}

void MaxPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalMaxUndo(_prev[0]->getActs(), v, getActs(), _prev[inpIdx]->getActsGrad(), _sizeX, _start, _stride, _outputsX, scaleTargets, 1);
}

/* 
 * =====================
 * NailbedLayer
 * =====================
 */
NailbedLayer::NailbedLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _start = pyDictGetInt(paramsDict, "start");
    _stride = pyDictGetInt(paramsDict, "stride");
    _outputsX = pyDictGetInt(paramsDict, "outputsX");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
}

void NailbedLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convBedOfNails(*_inputs[0], getActs(), _channels, _imgSize, _start, _stride, 0, 1);
}

void NailbedLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convBedOfNailsUndo(v, _prev[0]->getActsGrad(), _channels, _imgSize, _start, _stride, scaleTargets, 1);
}

/* 
 * =====================
 * GaussianBlurLayer
 * =====================
 */
GaussianBlurLayer::GaussianBlurLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _hFilter = pyDictGetMatrix(paramsDict, "filter");
}

void GaussianBlurLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convGaussianBlur(*_inputs[0], _filter, getActs(), true, _channels, 0, 1);
    convGaussianBlur(getActs(), _filter, getActs(), false, _channels, 0, 1);
}

// This is here just for completeness' sake. Why would you backpropagate
// through a blur filter?
void GaussianBlurLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& tgt1 = _prev[0]->getRcvdBInputs() > 0 ? _actGradsTmp : _prev[0]->getActsGrad();
    convGaussianBlur(v, _filter, tgt1, true, _channels, 0, 1);
    convGaussianBlur(tgt1, _filter, _prev[0]->getActsGrad(), false, _channels, scaleTargets, 1);
}

void GaussianBlurLayer::copyToGPU() {
    _filter.copyFromHost(*_hFilter, true);
}

/* 
 * =====================
 * ResizeLayer
 * =====================
 */
ResizeLayer::ResizeLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
    _tgtSize = pyDictGetInt(paramsDict, "tgtSize");
    _scale = pyDictGetFloat(paramsDict, "scale");
}

void ResizeLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResizeBilinear(*_inputs[0], getActs(), _imgSize, _tgtSize, _scale);
}

// Can't do this
void ResizeLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * RGBToYUVLayer
 * =====================
 */
RGBToYUVLayer::RGBToYUVLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
}

void RGBToYUVLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convRGBToYUV(*_inputs[0], getActs());
}

// Can't do this
void RGBToYUVLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * RGBToLABLayer
 * =====================
 */
RGBToLABLayer::RGBToLABLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _center = pyDictGetInt(paramsDict, "center");
}

void RGBToLABLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convRGBToLAB(*_inputs[0], getActs(), _center);
}

// Can't do this
void RGBToLABLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * ResponseNormLayer
 * =====================
 */
ResponseNormLayer::ResponseNormLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _size = pyDictGetInt(paramsDict, "size");

    _scale = pyDictGetFloat(paramsDict, "scale");
    _pow = pyDictGetFloat(paramsDict, "pow");
}

void ResponseNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNorm(*_inputs[0], _denoms, getActs(), _channels, _size, _scale, _pow);
}

void ResponseNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormUndo(v, _denoms, _prev[0]->getActs(), getActs(), _prev[0]->getActsGrad(), _channels, _size, _scale, _pow, scaleTargets, 1);
}

void ResponseNormLayer::truncBwdActs() {
    Layer::truncBwdActs();
    if (_conserveMem) {
        _denoms.truncate();
    }
}

/* 
 * =====================
 * CrossMapResponseNormLayer
 * =====================
 */
CrossMapResponseNormLayer::CrossMapResponseNormLayer(ConvNet* convNet, PyObject* paramsDict) : ResponseNormLayer(convNet, paramsDict) {
    _blocked = pyDictGetInt(paramsDict, "blocked");
}

void CrossMapResponseNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormCrossMap(*_inputs[0], _denoms, getActs(), _channels, _size, _scale, _pow, _blocked);
}

void CrossMapResponseNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormCrossMapUndo(v, _denoms, _prev[0]->getActs(), getActs(), _prev[0]->getActsGrad(), _channels, _size, _scale, _pow, _blocked, scaleTargets, 1);
}


/* 
 * =====================
 * ContrastNormLayer
 * =====================
 */
ContrastNormLayer::ContrastNormLayer(ConvNet* convNet, PyObject* paramsDict) : ResponseNormLayer(convNet, paramsDict) {
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
}

void ContrastNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& images = *_inputs[0];
    convLocalPool(images, _meanDiffs, _channels, _size, -_size/2, 1, _imgSize, AvgPooler());
    _meanDiffs.add(images, -1, 1);
    convContrastNorm(images, _meanDiffs, _denoms, getActs(), _channels, _size, _scale, _pow);
}

void ContrastNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convContrastNormUndo(v, _denoms, _meanDiffs, getActs(), _prev[inpIdx]->getActsGrad(), _channels, _size, _scale, _pow, scaleTargets, 1);
}

void ContrastNormLayer::truncBwdActs() {
    ResponseNormLayer::truncBwdActs();
    if (_conserveMem) {
        _meanDiffs.truncate();
    }
}

/* 
 * =====================
 * CostLayer
 * =====================
 */
CostLayer::CostLayer(ConvNet* convNet, PyObject* paramsDict, bool trans) 
    : Layer(convNet, paramsDict, trans) {
    _coeff = pyDictGetFloat(paramsDict, "coeff");
}

float CostLayer::getCoeff() {
    return _coeff;
}

void CostLayer::bprop(PASS_TYPE passType) {
    if (_coeff != 0) {
        Layer::bprop(passType);
    }
}

bool CostLayer::isGradProducer() {
    return _coeff != 0;
}

doublev& CostLayer::getCost() {
    doublev& v = *new doublev();
    v.insert(v.begin(), _costv.begin(), _costv.end());
    return v;
}

CostLayer& CostLayer::makeCostLayer(ConvNet* convNet, string& type, PyObject* paramsDict) {
    if (type == "cost.logreg") {
        return *new LogregCostLayer(convNet, paramsDict);
    } else if (type == "cost.sum2") {
        return *new SumOfSquaresCostLayer(convNet, paramsDict);
    } else if (type == "cost.rlog") {
        return *new RLogCostLayer(convNet, paramsDict);
    } else if (type == "cost.l2svm") {
        return *new L2SVMCostLayer(convNet, paramsDict);
    }
    throw string("Unknown cost layer type ") + type;
}

/* 
 * =====================
 * LogregCostLayer
 * =====================
 */
LogregCostLayer::LogregCostLayer(ConvNet* convNet, PyObject* paramsDict) : CostLayer(convNet, paramsDict, false) {
}

void LogregCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& probs = *_inputs[1];
        int numCases = labels.getNumElements();
        NVMatrix& trueLabelLogProbs = getActs(), correctProbs;

        computeLogregCost(labels, probs, trueLabelLogProbs, correctProbs);
        _costv.clear();
        _costv.push_back(-trueLabelLogProbs.sum());
        _costv.push_back(numCases - correctProbs.sum());
    }
}

void LogregCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 1);
    NVMatrix& labels = _prev[0]->getActs();
    NVMatrix& probs = _prev[1]->getActs();
    NVMatrix& target = _prev[1]->getActsGrad();
    // Numerical stability optimization: if the layer below me is a softmax layer, let it handle
    // the entire gradient computation to avoid multiplying and dividing by a near-zero quantity.
    bool doWork = _prev[1]->getNext().size() > 1 || _prev[1]->getType() != "softmax";
    if (doWork) {
        computeLogregGrad(labels, probs, target, scaleTargets == 1, _coeff);
    }
}


/* 
 * =====================
 * L2SVMCostLayer
 * =====================
 */
L2SVMCostLayer::L2SVMCostLayer(ConvNet* convNet, PyObject* paramsDict) : CostLayer(convNet, paramsDict, false) {
}

void L2SVMCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& acts_prev = *_inputs[1];
        int numCases = labels.getNumElements();
        NVMatrix& acts = getActs(), correctPreds;

		computeL2SVMCost(labels, acts_prev, acts, correctPreds);
        _costv.clear();
        _costv.push_back(acts.sum());//should be max(1-t*act_prev, 0) instead
		_costv.push_back(numCases - correctPreds.sum());
    }

}

void L2SVMCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	//v = getActsGrad() from next - not needed, no next
    assert(inpIdx == 1);
}

/* 
 * =====================
 * RLogCostLayer
 * =====================
 */
RLogCostLayer::RLogCostLayer(ConvNet* convNet, PyObject* paramsDict) : CostLayer(convNet, paramsDict, false),
_avg_log(2.3), _wScale(1) {
	_lp_norm = pyDictGetFloat(paramsDict, "lp_norm");
	_l_decay = pyDictGetFloat(paramsDict, "l_decay");
	_init_coeff = _coeff;
}

void RLogCostLayer::SetCoeff(float newCoeff) {
	_coeff = newCoeff;
}

NVMatrix* RLogCostLayer::GetProbWeights(){
	return &_probWeights;
};

void RLogCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& probs = *_inputs[1];
        int numCases = labels.getNumElements();
        NVMatrix& trueLabelLogProbs = getActs(), correctProbs;

		_probWeights.resize(labels);

		float p_pow = _lp_norm-1;

        computeRLogCost(labels, probs, trueLabelLogProbs, correctProbs, _probWeights, p_pow);
        _costv.clear();
		float sum = -trueLabelLogProbs.sum();
        _costv.push_back(sum);
        _costv.push_back(numCases - correctProbs.sum());

		_avg_log = sum/numCases;

		float step = fminf(pow(_avg_log, _l_decay), _init_coeff);
		SetCoeff(step);

    }
}

void RLogCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 1);
    NVMatrix& labels = _prev[0]->getActs();
    NVMatrix& probs = _prev[1]->getActs();
    NVMatrix& target = _prev[1]->getActsGrad();

    // Numerical stability optimization: if the layer below me is a softmax layer, let it handle
    // the entire gradient computation to avoid multiplying and dividing by a near-zero quantity.
    bool doWork = _prev[1]->getNext().size() > 1 || _prev[1]->getType() != "softmax";
    if (doWork) {
       computeRLogGrad(labels, probs, target, _probWeights, scaleTargets == 1, _coeff);
    }
}

/* 
 * =====================
 * SumOfSquaresCostLayer
 * =====================
 */
SumOfSquaresCostLayer::SumOfSquaresCostLayer(ConvNet* convNet, PyObject* paramsDict) : CostLayer(convNet, paramsDict, false) {
}

void SumOfSquaresCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _inputs[0]->apply(NVMatrixOps::Square(), getActs());
    _costv.clear();
    _costv.push_back(getActs().sum());
}

void SumOfSquaresCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _prev[inpIdx]->getActsGrad().add(*_inputs[0], scaleTargets, -2 * _coeff);
}
