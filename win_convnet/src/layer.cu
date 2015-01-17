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

//debug
extern int gepoch;
extern int minibatch;
FILE* deb_file = NULL;
int deb_out_start = 0;
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
	_dropout_init = _dropout;

	_nan2Zero = false;
	_no_update = false;
}

void Layer::setDropout(float dropout)
{
	_dropout = dropout;
};

float Layer::getDropoutInit()
{
	return _dropout_init;
}; 

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

	if(eps_scale > 0)
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

    if (passType != PASS_TEST && passType != PASS_AUX && _dropout > 0.0) {
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
    
    if (_dropout > 0.0 && passType != PASS_AUX) {
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
	_svrg =  pyDictGetInt(paramsDict, "svrg");

	MatrixV *phAux_weights = NULL;
	Matrix *phAux_bias = NULL;

//bregman
	if(_svrg)
	{
		phAux_weights = pyDictGetMatrixV(paramsDict, "aux_weight");
	    phAux_bias = pyDictGetMatrix(paramsDict, "aux_bias");
	}
    
    floatv& momW = *pyDictGetFloatV(paramsDict, "momW");
    float momB = pyDictGetFloat(paramsDict, "momB");
    floatv& epsW = *pyDictGetFloatV(paramsDict, "epsW");
    float epsB = pyDictGetFloat(paramsDict, "epsB");
    floatv& wc = *pyDictGetFloatV(paramsDict, "wc");

	float muL1 = pyDictGetFloat(paramsDict, "muL1");

	_renorm = pyDictGetFloat(paramsDict, "renorm");
	float rmsW = pyDictGetFloat(paramsDict, "rmsW");
	float rmsB = pyDictGetFloat(paramsDict, "rmsB");

	//debug
	printf("layer %s use rmsW %f rmsB %f \n", _name.c_str(), rmsW, rmsB);
    
    // Source layers for shared weights
    intv& weightSourceLayerIndices = *pyDictGetIntV(paramsDict, "weightSourceLayerIndices");
    // Weight matrix indices (inside the above source layers) for shared weights
    intv& weightSourceMatrixIndices = *pyDictGetIntV(paramsDict, "weightSourceMatrixIndices");


	int aux_store_size = 0;
	bool activeAux = false;
	if(_svrg)
	{
		activeAux = true;
		aux_store_size = AUX_STORAGE;
		printf("layer %s use _svrg \n", _name.c_str());
	}
   
    for (int i = 0; i < weightSourceLayerIndices.size(); i++) {

        int srcLayerIdx = weightSourceLayerIndices[i];
        int matrixIdx = weightSourceMatrixIndices[i];

        if (srcLayerIdx == convNet->getNumLayers()) { // Current layer
            _weights.addWeights(*new Weights(_weights[matrixIdx], epsW[i], rmsW));
        } else if (srcLayerIdx >= 0) {
            WeightLayer& srcLayer = *static_cast<WeightLayer*>(&convNet->getLayer(srcLayerIdx));
            Weights* srcWeights = &srcLayer.getWeights(matrixIdx);
            _weights.addWeights(*new Weights(*srcWeights, epsW[i], rmsW));

        } else if(_svrg == 0){
            _weights.addWeights(*new Weights(*hWeights[i], *hWeightsInc[i], epsW[i], rmsW, wc[i], momW[i], muL1, _renorm, useGrad));
        } else
			_weights.addWeights(*new Weights(*hWeights[i], *hWeightsInc[i],
			*((*phAux_weights)[i]), activeAux, aux_store_size,
			epsW[i], rmsW, wc[i], momW[i], muL1, _renorm, useGrad));

    }
    if(_svrg == 0)
		_biases = new Weights(hBiases, hBiasesInc, epsB, rmsB, 0, momB, 0, 0, true);
	else
		_biases = new Weights(hBiases, hBiasesInc,
		*phAux_bias, activeAux, aux_store_size,
		epsB, rmsB, 0, momB, 0, 0, true);


    // Epsilons for finite-difference gradient checking operation
    _wStep = 0.001;
    _bStep = 0.002;
	_notUseBias = false;
    
    delete &weightSourceLayerIndices;
    delete &weightSourceMatrixIndices;
    delete &hWeights;
    delete &hWeightsInc;
    delete &momW;
    delete &epsW;
    delete &wc;
}

#define MOM_MULTIPLYER 5

void WeightLayer::setCommon(float eps_scale) {
	if(eps_scale > 0)
	{
		for (int i = 0; i < _weights.getSize(); i++) {

			_weights[i].setEps(_weights[i].getEpsInit()*eps_scale);
			//_weights[i].setWc(_weights[i].getWcInit()*eps_scale);
			//float dm = 1 - _weights[i].getMomInit();
			//dm *= eps_scale*MOM_MULTIPLYER;
			//_weights[i].setMom(min(1 - dm, .9999));
		}
		_biases->setEps(_biases->getEpsInit()*eps_scale);

	}
}

void WeightLayer::bpropCommon(NVMatrix& v, PASS_TYPE passType) {
    if (_biases->getEps() > 0 && !_notUseBias) {

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

void WeightLayer::rollbackWeights(float reduceScale) {
    _weights.rollback(reduceScale);
	_biases->rollback(reduceScale);
}

void WeightLayer::updateWeights(bool useAux) {

//	printf(" layer name %s \n", _name.c_str());


    _weights.update(useAux);
	_biases->update(useAux);

//debug
	//if(gepoch%5==1 && minibatch==0)
	//{
	//	if(deb_out_start == 0)
	//	{
	//		deb_file = fopen("deb_output.txt", "wt");
	//		deb_out_start++;
	//	}
	//	else
	//	{
	//		deb_file = fopen("deb_output.txt", "at");
	//	}

	//	fprintf(deb_file, " layer %s avg_grad %f bias%f \n", _name.c_str(),  _weights[0].getNormL2Avg(), _biases->getNormL2Avg());
	//printf(" layer %s avg_grad %f bias%f \n", _name.c_str(),  _weights[0].getNormL2Avg(), _biases->getNormL2Avg());


	//	fclose(deb_file);
	//}	
	//    
}

void WeightLayer::procAuxWeights() {
    _weights.procAux();
	_biases->procAux();   
}

void WeightLayer::zeroAuxWeights() {
    _weights.zeroAux();
	_biases->zeroAux();   
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

Weights* WeightLayer::getBiases() {
    return _biases;
}

void WeightLayer::postInit() {
	Layer::postInit();
	if(_next[0]->getType() == "dshrink")
	{
		printf("next dshrink detected in layer %s, bias off\n", _name.c_str());
		_notUseBias = true;
	}

}


/* 
 * =======================
 * BiasLayer
 * =======================
 */
BiasLayer::BiasLayer(ConvNet* convNet, PyObject* paramsDict, bool trans, bool useGrad) : 
    Layer(convNet, paramsDict, trans) {

    Matrix& hBiases = *pyDictGetMatrix(paramsDict, "biases");
    Matrix& hBiasesInc = *pyDictGetMatrix(paramsDict, "biasesInc");

	string layerType = pyDictGetString(paramsDict, "type");

    float mom = pyDictGetFloat(paramsDict, "mom");
    float eps = pyDictGetFloat(paramsDict, "eps");
    float wc = pyDictGetFloat(paramsDict, "wc"); 

	_biases = new Weights(hBiases, hBiasesInc, eps, wc, 1, mom, 0, 0, true);
}

void BiasLayer::bpropCommon(NVMatrix& v, PASS_TYPE passType) {
    if (_biases->getEps() > 0) {
        bpropBiases(v, passType);
    }
}

void BiasLayer::setCommon(float eps_scale) {
	if(eps_scale > 0)
	{
		_biases->setEps(_biases->getEpsInit()*eps_scale);
		_biases->setWc(_biases->getWcInit()*eps_scale);
		//float dm = 1 - _biases->getMomInit();
		//dm *= eps_scale*2;
		//_biases->setMom(1 - dm);
	}
}

void BiasLayer::updateBiases() {
    _biases->update(false);    
}

void BiasLayer::rollbackWeights(float reduceScale) {
	_biases->rollback(reduceScale);
}

//procAux here?

void BiasLayer::copyToCPU() {
    _biases->copyToCPU();
}

void BiasLayer::copyToGPU() {
    _biases->copyToGPU();
}

Weights* BiasLayer::getBiases() {
    return _biases;
}

/* 
 * =======================
 * LeakReLuLayer
 * =======================
 */
LeakReLuLayer::LeakReLuLayer(ConvNet* convNet, PyObject* paramsDict) :
    BiasLayer(convNet, paramsDict, false, true) {
}

void LeakReLuLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType)
{
    if (!getActs().isSameDims(*_inputs[inpIdx])) {
        getActs().resize(*_inputs[inpIdx]);
    }

//	printf("name %s \n", _name.c_str());
//	printf("getActs() tran %i rc %i %i w %i\n", getActs().isTrans(),
//		getActs().getNumRows(), getActs().getNumCols(), _biases->getW().getNumElements() );
	_inputs[inpIdx]->applyBinaryV(LeakReLuOperator(), _biases->getW(), getActs());

};

void LeakReLuLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType)
{

	NVMatrix& target = _prev[0]->getActsGrad();
	NVMatrix& actsGrad = v;
	NVMatrix& input = *_inputs[0];

    if (scaleTargets == 0) {
		if (!target.isSameDims(actsGrad))
			target.resize(actsGrad);
        actsGrad.applyDTernaryV(LeakReLuGradOperator(), input, _biases->getW(), target);
    } else {
		actsGrad.addDTernaryV(LeakReLuGradOperator(), input, _biases->getW(), target);
	}
};

void LeakReLuLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType)
{

	int numCases = getActs().getNumCols(); 

    float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;

    if (_tempMult.isSameDims(getActs())) {
        _tempMult.resize(getActs());
    }

	NVMatrix& input = *_inputs[0];

	input.eltwiseMult(v, _tempMult);

	_biases->getGrad().addSum(_tempMult, 1, 0, scaleBGrad);

};

void LeakReLuLayer::updateWeights(bool useAux) {
	_biases->update(useAux);
	//_biases->getW().absMinWithScalar(.1);
	_biases->getW().minWithScalar(.5);
	_biases->getW().maxWithScalar(0);
//debug
if(minibatch == 0)
{
float avgNorm = _biases->getW().norm()/sqrtf(_biases->getW().getNumElements());
float avg = _biases->getW().sum()/_biases->getW().getNumElements();
printf("%s avg %f avg_norm %f \n", _name.c_str(), avg, avgNorm);
}

}

Weights* LeakReLuLayer::getLeak() {
    return _biases;
}


extern int train;//temp
extern  int gmini;//temp
extern  int gmini_max;//temp
//#define show_mini gmini
#define show_mini (gmini_max-1)
//#define show_mini 1


/* 
 * =======================
 * DShrinkLayer
 * =======================
 */

DShrinkLayer::DShrinkLayer(ConvNet* convNet, PyObject* paramsDict) : BiasLayer(convNet, paramsDict, false, true) {
}

void DShrinkLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType)
{
	assert(_prev[inpIdx]->getType() == "conv");

	WeightLayer* prevLayer = (WeightLayer*)_prev[inpIdx];
	Weights* prevBias = prevLayer->getBiases();

	ConvLayer* convLayer = (ConvLayer*)_prev[0];
	assert(convLayer->isSharedBiases());

	int numFilters = convLayer->getNumFilters();
	int modules = convLayer->getNumModules();

    if (!getActs().isSameDims(*_inputs[inpIdx])) {
        getActs().resize(*_inputs[inpIdx]);
    }

//	printf("getActs() tran %i rc %i %i \n", getActs().isTrans(),
//		getActs().getNumRows(), getActs().getNumCols());

    (*_inputs[inpIdx]).reshape(numFilters, (*_inputs[inpIdx]).getNumElements() / numFilters);
     getActs().reshape(numFilters, getActs().getNumElements() / numFilters);

//	printf(" fpropActs %s prev bias  cont %i bias cont %i \n", _name.c_str(), prevBias->getW().isContiguous(), _biases->getW().isContiguous());  

	//printf("inp tran %i rc %i %i pbias %i %i %i biases %i %i %i \n", 
	//	(*_inputs[inpIdx]).isTrans(),
	//	(*_inputs[inpIdx]).getNumRows(), (*_inputs[inpIdx]).getNumCols(),
	//	prevBias->getW().isTrans(),
	//	prevBias->getW().getNumRows(), prevBias->getW().getNumCols(),
	//	_biases->getW().isTrans(),
	//	_biases->getW().getNumRows(), _biases->getW().getNumCols());


	_inputs[inpIdx]->applyTernaryV(NVMatrixTernaryOps::DShrink(), prevBias->getW(), _biases->getW(), getActs());

     (*_inputs[inpIdx]).reshape(numFilters * modules, (*_inputs[inpIdx]).getNumElements() / (numFilters * modules));
	 getActs().reshape(numFilters * modules, getActs().getNumElements() / (numFilters * modules));

};

void DShrinkLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
//printf("start bpropActs \n");
	assert(_prev[inpIdx]->getType() == "conv");

	WeightLayer* prevLayer = (WeightLayer*)_prev[inpIdx];
	Weights* prevBias = prevLayer->getBiases();

	//printf("tran %i inp %i %i v %i %i pbias %i %i biases %i %i prevaccgrad %i %i \n", 
	//	v.isTrans(),
	//	(*_inputs[inpIdx]).getNumRows(), (*_inputs[inpIdx]).getNumCols(),
	//	v.getNumRows(), v.getNumCols(),
	//	prevBias->getW().getNumRows(), prevBias->getW().getNumCols(),
	//	_biases->getW().getNumRows(), _biases->getW().getNumCols(),
	//	_prev[inpIdx]->getActsGrad().getNumRows(), _prev[inpIdx]->getActsGrad().getNumCols());

	ConvLayer* convLayer = (ConvLayer*)_prev[0];
	assert(convLayer->isSharedBiases());

	int numFilters = convLayer->getNumFilters();
	int modules = convLayer->getNumModules();


	 _prev[inpIdx]->getActsGrad().resize(v); // target must be same orientation as me for now

	_inputs[0]->reshape(numFilters, _temp_neg.getNumElements() / numFilters);
	v.reshape(numFilters, _temp_neg.getNumElements() / numFilters);
	_prev[inpIdx]->getActsGrad().reshape(numFilters, _temp_neg.getNumElements() / numFilters);

//no reshape inside dshrinkGrad
	dshrinkGrad(v, *_inputs[inpIdx], prevBias->getW(), _biases->getW(),
					   _prev[inpIdx]->getActsGrad());

	 _inputs[0]->reshape(numFilters * modules, _temp_pos.getNumElements() / (numFilters * modules));
	 v.reshape(numFilters * modules, _temp_pos.getNumElements() / (numFilters * modules));
	 _prev[inpIdx]->getActsGrad().reshape(numFilters * modules, _temp_pos.getNumElements() / (numFilters * modules));


}

void DShrinkLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType)
{
	assert(_prev.size() == 1);
	assert(_prev[0]->getType() == "conv");

	WeightLayer* prevLayer = (WeightLayer*)_prev[0];
	Weights* prevBias = prevLayer->getBiases();

	assert(prevBias->getW().getNumRows() == _biases->getW().getNumRows()
		&& prevBias->getW().getNumCols() == _biases->getW().getNumCols());


	_temp_pos.resizeUp(v.getNumRows(), v.getNumCols());
	_temp_neg.resizeUp(v.getNumRows(), v.getNumCols());

	 int numCases = v.getNumCols();


	float scaleBGrad = _biases->getEps() / numCases;

	if(_prev[0]->getType() == "conv" )
	{
		ConvLayer* convLayer = (ConvLayer*)_prev[0];
		assert(convLayer->isSharedBiases());

		int numFilters = convLayer->getNumFilters();
		int modules = convLayer->getNumModules();

		_inputs[0]->reshape(numFilters, _temp_neg.getNumElements() / numFilters);
		v.reshape(numFilters, _temp_neg.getNumElements() / numFilters);

		if (!_temp_pos.isSameDims(v))
			_temp_pos.reshape(numFilters, _temp_pos.getNumElements() / numFilters);

		if (!_temp_neg.isSameDims(v))
			_temp_neg.reshape(numFilters, _temp_neg.getNumElements() / numFilters);


		dshrinkWeightGrad(v, *_inputs[0], prevBias->getW(), _biases->getW(),
					   _temp_pos, _temp_neg);

         prevBias->getGrad().addSum(_temp_pos, 1, 0, scaleBGrad);
        _biases->getGrad().addSum(_temp_neg, 1, 0, scaleBGrad);

  		 _inputs[0]->reshape(numFilters * modules, _temp_pos.getNumElements() / (numFilters * modules));
		 v.reshape(numFilters * modules, _temp_pos.getNumElements() / (numFilters * modules));

	}
	//else //fc layer
	//{
//check sizes
//	dshrinkWeightGrad(v, *_inputs[0], prevBias->getW(), _biases->getW(),
//					   _temp_pos, _temp_neg);

 //       prevBias->getGrad().addSum(_temp_pos, 1, 0, scaleBGrad);
	//	_biases->getGrad().addSum(_temp_neg, 1, 0, scaleBGrad);
	//}

};

/* 
 * =======================
 * FCLayer
 * =======================
 */

FCLayer::FCLayer(ConvNet* convNet, PyObject* paramsDict) : WeightLayer(convNet, paramsDict, true, true) {
    _wStep = 0.1;
    _bStep = 0.01;
}

void FCLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    getActs().addProduct(*_inputs[inpIdx], *_weights[inpIdx], scaleTargets, 1);
    if (scaleTargets == 0 && !_notUseBias) {
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
  
//  _weights[inpIdx].getInc().addProduct(prevActs_T, v, scaleInc, scaleGrad);
	prevActs_T.rightMult(v, scaleGrad, _weights[inpIdx].getGrad());
    
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

int LocalLayer::getNumFilters()
{
	return _numFilters;
};

int LocalLayer::getNumModules()
{
	return _modules;
};


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

//	printf(" conv fprop name %s \n", _name.c_str());


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

	if (scaleTargets == 0 && !_notUseBias) {
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

bool ConvLayer::isSharedBiases()
{
	return _sharedBiases;
};
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
 * EltwiseAbsMaxLayer
 * =======================
 */
EltwiseAbsMaxLayer::EltwiseAbsMaxLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
}

void EltwiseAbsMaxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (inpIdx == 1) { // First input, do nothing
        _inputs[inpIdx]->applyBinary(NVMatrixAggs::AbsMax(), *_inputs[0], getActs());
    } else if (inpIdx > 1) {
        getActs().applyBinary(NVMatrixAggs::AbsMax(), *_inputs[inpIdx]);
    }
}

void EltwiseAbsMaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    computeEltwiseMaxGrad(v, *_inputs[inpIdx], getActs(), _prev[inpIdx]->getActsGrad(), scaleTargets != 0);
}


/* 
 * =====================
 * MAvgPoolLayer
 * =====================
 */
MAvgPoolLayer::MAvgPoolLayer(ConvNet* convNet, PyObject* paramsDict) :Layer(convNet, paramsDict, false) {

    _channels = pyDictGetInt(paramsDict, "channels");
    _size = pyDictGetInt(paramsDict, "size");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
	_imgPixels = pyDictGetInt(paramsDict, "imgPixels");
}

void MAvgPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {

		computeMAvgAct(*_inputs[inpIdx],  getActs(),  _size, _channels, _imgSize, _imgPixels);

}

void MAvgPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {

	computeMAvgGrad(v, _prev[inpIdx]->getActsGrad(),
							 _size, _channels,
							_imgSize, _imgPixels);
}

/* 
 * =====================
 * MMaxPoolLayer
 * =====================
 */
MMaxPoolLayer::MMaxPoolLayer(ConvNet* convNet, PyObject* paramsDict) :Layer(convNet, paramsDict, false) {

    _channels = pyDictGetInt(paramsDict, "channels");
    _size = pyDictGetInt(paramsDict, "size");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
	_imgPixels = pyDictGetInt(paramsDict, "imgPixels");
}

void MMaxPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {

		computeMMaxAct(*_inputs[inpIdx], getActs(),  _size, _channels, _imgSize, _imgPixels);

}

void MMaxPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {

	computeMMaxGrad(v, _prev[0]->getActs(), getActs(), _prev[inpIdx]->getActsGrad(),
							 _size, _channels,
							_imgSize, _imgPixels);

}

/* 
 * =======================
 * VectFuncLayer
 * =======================
 */

VectFuncLayer::VectFuncLayer(ConvNet* convNet, PyObject* paramsDict): Layer(convNet, paramsDict, false) 
{
	hParamList = PyDict_GetItemString(paramsDict, "meta_param");
	_param = getVectorDouble(hParamList);

	hParamListInc = PyDict_GetItemString(paramsDict, "meta_param_inc");
	_param_inc = getVectorDouble(hParamListInc);

	 _sizeV = pyDictGetInt(paramsDict, "sizeV");
	 _sizeH = pyDictGetInt(paramsDict, "sizeH");
	 _channels = pyDictGetInt(paramsDict, "channels");
    
    _mom = pyDictGetFloat(paramsDict, "mom");
    _epsP = pyDictGetFloat(paramsDict, "epsP");
    _wc = pyDictGetFloat(paramsDict, "wc");

	assert(_sizeV*_sizeH == _param.size());

	for (int j =0; j < _param.size(); j++)
		_nstore_count.push_back(0);

	for (int i =0; i < NSTORE; i++)
	for (int j =0; j < _param.size(); j++)
		_grad_store[i].push_back(0);

	for (int j =0; j < _param.size(); j++)
		_tempMatrixArray.push_back(NVMatrix());

	int size_arr = (_param.size()+8)/8;
	size_arr *= 8;
	cudaMalloc(&_arrayPtr, sizeof(float*)*size_arr);
};

VectFuncLayer::~VectFuncLayer()
{
	cudaFree(_arrayPtr);
}

void VectFuncLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType)
{

	computeVectFuncAct(*_inputs[inpIdx],  getActs(), _param,  _sizeV, _sizeH, _channels);

	//printf(" VectFuncLayer fpropActs end\n");
}

void VectFuncLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType)
{

//	printf(" VectFuncLayer bpropActs start\n");


	computeVectFuncGrad(v, *_inputs[inpIdx], _prev[inpIdx]->getActsGrad(),
							 _param, _sizeV, _sizeH, _channels);

//	printf(" VectFuncLayer bpropActs end\n");
}

void VectFuncLayer::bpropCommon(NVMatrix& v, PASS_TYPE passType)
{
//coordinate descent
	for (int i = 0; i < _prev.size(); i++)
		bpropWeights(v, i, passType);

}

void VectFuncLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType)
{
//printf(" VectFuncLayer bpropWeights start\n");


	computeVectFuncWeightGrad(v, *_inputs[inpIdx],
								_tempMatrixArray,
								_arrayPtr,
								_param,  _sizeV, _sizeH, _channels);

	
	_tempMatrixArray[0].ResizeAggStorage(_aggStorage._aggMatrix, _aggStorage._srcCPU);

	int paramSize = _param.size();


	for(int kp = 0; kp < paramSize; kp++)
	{
		//double grad = _tempMatrixArray[kp].sum();
		double grad = _tempMatrixArray[kp].sum_fast(_aggStorage._aggMatrix, _aggStorage._srcCPU);

		double sum_grad = 0;
		for(int k = 0; k < NSTORE; k++)
			sum_grad += _grad_store[k][kp]*_grad_store[k][kp];


		_grad_store[_nstore_count[kp]][kp] = grad;
		_nstore_count[kp] = (_nstore_count[kp]+1)%NSTORE;

		if(sum_grad > 0)
			grad = grad*sqrt(NSTORE)/sqrt(sum_grad);

		_param_inc[kp] = _mom*_param_inc[kp] + _epsP*grad - _wc*_param[kp];
		_param[kp] += _param_inc[kp];


	}

	double sumScale = 5;
	double l1sum = 0;
	for(int i =0; i < _param.size(); i++)
		l1sum += fabs(_param[i]);

	for(int i =0; i < _param.size(); i++)
		_param[i] *= sumScale/l1sum;

	//debug
	if(minibatch == 0)
	{
		printf("vectf %f %f  %f %f  %f %f  %f %f \n",  _param[0], _param[1], _param[2], _param[3], _param[4], _param[5], _param[6], _param[7]);
		printf("      %f %f  %f %f  %f %f  %f %f \n",  _param[8], _param[9], _param[10], _param[11], _param[12], _param[13], _param[14], _param[15]);
	}

//	printf(" VectFuncLayer bpropWeights end\n");
}

void VectFuncLayer::copyToCPU()
{
	for(int i = 0; i < _param.size(); i++)
	{
		PyList_SetItem(hParamList, i,  PyFloat_FromDouble(_param[i]));
		PyList_SetItem(hParamListInc, i,  PyFloat_FromDouble(_param_inc[i]));
	}
};

/* 
 * =======================
 * MicroConvLayer
 * =======================
 */
MicroConvLayer::MicroConvLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {

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


//temporary - one filter
	//_param.erase(_param.begin() + _param.size()/_numFilters, _param.end());
	//_param_inc.erase(_param_inc.begin() + _param_inc.size()/_numFilters, _param_inc.end());
	//assert(_size*_size*_channels == _param.size());
//
//	assert(_size*_size*_channels*_numFilters == _param.size());
//debug

	{
		int numl = (_param.size()+9)/9;
		printf("** MicroConvLayer params *** \n");
		for (int nk = 0; nk < numl; nk++)
		{
			for (int k = 0; k < 9; k++)
				if(k + nk*9 < _param.size())
				printf("%f ", _param[k + nk*9]);
			printf("\n");
		}
	}

	for (int j =0; j < _param.size(); j++)
		_nstore_count.push_back(0);

	for (int i =0; i < NSTORE; i++)
	for (int j =0; j < _param.size(); j++)
		_grad_store[i].push_back(0);
	//printf(" _param init  %f %f %f \n", _param[2] , _param[_sizeIn + 0] , _param[_sizeIn + 1]);
	//printf(" size_in %i size_out %i  updates %i \n",_sizeIn, _sizeOut, _updates);


	for (int j =0; j < _param.size(); j++)
		_tempMatrixArray.push_back(NVMatrix());

	int size_arr = (_param.size()+8)/8;
	size_arr *= 8;
	cudaMalloc(&_arrayPtr, sizeof(float*)*size_arr);
};

MicroConvLayer::~MicroConvLayer()
{
	cudaFree(_arrayPtr);	
}

void MicroConvLayer::copyToCPU()
{
	for(int i = 0; i < _param.size(); i++)
	{
		PyList_SetItem(hParamList, i,  PyFloat_FromDouble(_param[i]));
		PyList_SetItem(hParamListInc, i,  PyFloat_FromDouble(_param_inc[i]));
	}
};

void MicroConvLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
	//printf(" MicroConvLayer fpropActs minibatch %i \n", minibatch);

	computeMicroConvAct(*_inputs[inpIdx],  getActs(), _param, _size, _channels, _imgSize, _imgPixels, _numFilters);

	//printf(" MicroConvLayer fpropActs end\n");
}

void MicroConvLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType)
{
//weight grad

//	printf(" MicroConvLayer bpropActs start\n");

	computeMicroConvActGrad(v, *_inputs[inpIdx], _prev[inpIdx]->getActsGrad(),
							 _param, _size, _channels,
							_imgSize, _imgPixels, _numFilters);

//	printf(" MicroConvLayer bpropActs end\n");

};

void MicroConvLayer::bpropCommon(NVMatrix& v, PASS_TYPE passType)
{
//coordinate descent
	for (int i = 0; i < _prev.size(); i++)
		bpropWeights(v, i, passType);

}



void MicroConvLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType)
{
//	printf(" MicroConvLayer bpropWeights start\n");

	computeMicroConvWeightGrad(v, *_inputs[inpIdx],
								_tempMatrixArray,
								_arrayPtr,
								_param, _size, _channels,
								_imgSize,_imgPixels, _numFilters);

	int paramSize = _param.size();


	_tempMatrixArray[0].ResizeAggStorage(_aggStorage._aggMatrix, _aggStorage._srcCPU);

	for(int kp = 0; kp < paramSize; kp++)
	{
		double grad = _tempMatrixArray[kp].sum_fast(_aggStorage._aggMatrix, _aggStorage._srcCPU);

		double sum_grad = 0;
		for(int k = 0; k < NSTORE; k++)
		{
			sum_grad += _grad_store[k][kp]*_grad_store[k][kp];
		}

		_grad_store[_nstore_count[kp]][kp] = grad;
		_nstore_count[kp] = (_nstore_count[kp]+1)%NSTORE;

		if(sum_grad > 0)
			grad = grad*sqrt(NSTORE)/sqrt(sum_grad);

		_param_inc[kp] = _mom*_param_inc[kp] + _epsP*grad - _wc*_param[kp];
		_param[kp] += _param_inc[kp];

	}

	double sumScale = 3;
	double l1sum = 0;
	for(int i =0; i < _param.size(); i++)
		l1sum += fabs(_param[i]);

	for(int i =0; i < _param.size(); i++)
		_param[i] *= sumScale/l1sum;

	if(minibatch == 0)
	{
		int numl = (_param.size()+9)/9;
		printf("** params *** \n");
		for (int nk = 0; nk < numl; nk++)
		{
			for (int k = 0; k < 9; k++)
				if(k + nk*9 < _param.size())
				printf("%f ", _param[k + nk*9]);
			printf("\n");
		}
	}

//renormalize here possibly
//	printf(" MicroConvLayer bpropWeights end\n");
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
	_channels = pyDictGetInt(paramsDict, "channels");
	_updates = pyDictGetInt(paramsDict, "updates");
    _nstore = pyDictGetInt(paramsDict, "nstore");
    _mom = pyDictGetFloat(paramsDict, "mom");
    _epsP = pyDictGetFloat(paramsDict, "epsP");
    _wc = pyDictGetFloat(paramsDict, "wc");

	_epsPInit =_epsP;
	_wcInit = _wc;
	_momInit = _mom;

	_nstore_count = 0;

	hParamListGrad = PyDict_GetItemString(paramsDict, "meta_param_nstore");
	_grad_store = getVectorDouble(hParamListGrad);

	int nump = _sizeIn*2;
	int numl = (_param.size()+nump-1)/nump;
	printf("** params *** \n");
	for (int nk = 0; nk < numl; nk++)
	{
		for (int k = 0; k < nump; k++)
			if(k + nk*nump < _param.size())
			printf("%f ", _param[k + nk*nump]);
		printf("\n");
	}

	//printf(" size_in %i size_out %i  _channels %i \n",_sizeIn, _sizeOut, _channels);

	for (int j =0; j < _param.size(); j++)
		_tempMatrixArray.push_back(NVMatrix());

	int size_arr = (_param.size()+8)/8;
	size_arr *= 8;
	cudaMalloc(&_arrayPtr, sizeof(float*)*size_arr);

	_aux_store_size = AUX_STORAGE;
	_aux_filled = 0;
	_aux_update = 0;
	for (int i =0; i < _param.size(); i++)
	{
		_aux_sum.push_back(0);
		_aux_corr.push_back(0);
		_grad.push_back(0);
	}

	for (int k =0; k < _aux_store_size; k++)
	for (int j =0; j < _param.size(); j++)
	{
		_aux_storage.push_back(0);
	}
}

EltwiseFuncLayer::~EltwiseFuncLayer()
{
	cudaFree(_arrayPtr);
}

void EltwiseFuncLayer::copyToCPU()
{
	for(int i = 0; i < _param.size(); i++)
	{
		PyList_SetItem(hParamList, i,  PyFloat_FromDouble(_param[i]));
		PyList_SetItem(hParamListInc, i,  PyFloat_FromDouble(_param_inc[i]));
	}

	int offset = _nstore_count;

	for(int j = 0; j < _nstore; j++)
	for(int i = 0; i < _param.size(); i++)
	{
		int out = j*_param.size() + i;
		int in = ((j+offset)%_nstore)*_param.size() + i;
		PyList_SetItem(hParamListGrad, out,  PyFloat_FromDouble(_grad_store[in]));
	}


};

void EltwiseFuncLayer::MakeAuxParams()
{

	if(_aux_filled >= _aux_store_size)
	{

//make sum
		for (int i =0; i < _param.size(); i++)
		{
			_aux_sum[i] = 0;
			for (int k =0; k < _aux_store_size; k++)
				_aux_sum[i] += _aux_storage[i + k*_param.size()];			
		}

		int rnd_aux = rand()%AUX_STORAGE;

		for (int i =0; i < _param.size(); i++)
			_aux_corr[i] = 1./_aux_filled*_aux_sum[i] -  _aux_storage[i + rnd_aux*_param.size()];

		for (int i =0; i < _param.size(); i++)
			_grad[i] = _aux_corr[i];
	}

//fill new
	for (int i =0; i < _param.size(); i++)
		_aux_storage[i + _aux_update*_param.size()] = _grad[i];


	_aux_update = (_aux_update+1)%_aux_store_size;
	_aux_filled = min(_aux_filled+1, _aux_store_size);
};


void EltwiseFuncLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
	//printf(" EltwiseFuncLayer fpropActs minibatch %i \n", minibatch);

	computeEltwiseFuncAct(*_inputs[inpIdx],  getActs(), _param, _channels, _sizeIn, _sizeOut);

//	printf("fprop size_in %i size_out %i  inp %f act %f \n", _sizeIn, _sizeOut,(*_inputs[inpIdx]).sum(),  getActs().sum());

	//printf(" EltwiseFuncLayer fpropActs end\n");
}

void EltwiseFuncLayer::setCommon(float eps_scale) {
	if(eps_scale > 0)
	{
		//float dm = 1 - _momInit;
		//dm *= eps_scale*MOM_MULTIPLYER;
		//_mom = min(1 - dm, .9999);

		_epsP = eps_scale*_epsPInit;
		//_wc = eps_scale*_wcInit;	
	}
}

void EltwiseFuncLayer::rollbackWeights(float reduceScale) {

	for(int kp = 0; kp < _param.size()-2; kp++)//paramSize-2, B, C off
	{
		_param[kp] -= (1-reduceScale)*_param_inc[kp];
	}
}

void EltwiseFuncLayer::updateWeights(bool useAux)
{

	int paramSize = _param.size();

	for(int kp = 0; kp < paramSize-2; kp++)//paramSize-2, B, C off
	{

		int out_len = EL_SWITCH*ELWISE_FUNC_SEC*_sizeIn;
		int k_out = kp/out_len;
		int sw_len = ELWISE_FUNC_SEC*_sizeIn;
		int k_ws = (kp - k_out*out_len)/sw_len;
		int k_v = kp - k_out*out_len - k_ws*sw_len;

		if(k_v > 2*_sizeIn)
			continue;

		double grad = _grad[kp];
		
//should make orthognal projection to equal vector(sizeIn)

		double sum_grad = 0;
		int nsum = 0;
		for(int k = 0; k < _nstore; k++)
		{
			double g_stored = _grad_store[k*_param.size() + kp];

			if(g_stored > 0)
				nsum++;

			sum_grad += g_stored*g_stored;
		}

		_grad_store[_nstore_count*_param.size() + kp] = grad;

		if(sum_grad > 0)
			grad = grad*sqrt(nsum)/sqrt(sum_grad);
	
		double eps = _epsP;
		double wc = _wc;

		//_param_inc[kp] = _mom*_param_inc[kp] + eps*grad;
		//float r =_param_inc[kp] - wc*_param[kp];
		//if(_param_inc[kp]*r >= 0)
		//	_param_inc[kp] = r;
		_param_inc[kp] = _mom*_param_inc[kp] + eps*grad - wc*_param[kp];

			
//debug
		if(kp != paramSize-2)
			_param[kp] += _param_inc[kp];
	}

	_nstore_count = (_nstore_count+1)%_nstore;

	if(minibatch == 0)
	{
		int nump = _sizeIn*ELWISE_FUNC_SEC;
		int numl = (_param.size()+nump-1)/nump;
		printf("** params *** \n");
		for (int nk = 0; nk < numl; nk++)
		{
			for (int k = 0; k < nump; k++)
				if(k + nk*nump < _param.size())
				printf("%f ", _param[k + nk*nump]);
			printf("\n");
		}
	}

};

void EltwiseFuncLayer::l1normalize()
{
	int out_len = EL_SWITCH*ELWISE_FUNC_SEC*_sizeIn;
	int vect_len = _sizeIn*ELWISE_FUNC_SEC;
	int vnorm_len = _sizeIn*2;

	double sumScale = _sizeIn*_sizeOut;

	for(int k_sw = 0; k_sw < EL_SWITCH; k_sw++)
	{
		double l1sum = 0;
		for(int k_out = 0; k_out < _sizeOut; k_out++)
		{
			for(int kinp = 0; kinp < vnorm_len; kinp++)
			{
				double pv = _param[k_out*out_len + k_sw*vect_len + kinp];
				l1sum += fabs(pv);
			}
		}
		
		assert(l1sum>0);

		for(int k_out = 0; k_out < _sizeOut; k_out++)
		{
			for(int kinp = 0; kinp < vnorm_len; kinp++)
				_param[k_out*out_len + k_sw*vect_len + kinp] *= sumScale/l1sum;
		}
	}
}

void EltwiseFuncLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {

	l1normalize();

	int paramSize = _param.size();

	float lim = 1./_param[paramSize-1];

	computeEltwiseFuncParamWeightGrad(v, *_inputs[inpIdx],
								 _arrayPtr, _tempMatrixArray,
								 _tempC, _tempB, _param, lim,
								 _channels, _sizeIn, _sizeOut);

	_tempMatrixArray[0].ResizeAggStorage(_aggStorage._aggMatrix, _aggStorage._srcCPU);

	//_tempB.ResizeAggStorage(_aggStorageC._aggMatrix, _aggStorageC._srcCPU);B, C off

	int out_len = EL_SWITCH*ELWISE_FUNC_SEC*_sizeIn;
	int vect_len = _sizeIn*ELWISE_FUNC_SEC;
	int vnorm_len = _sizeIn*2;

	if(passType == PASS_TRAIN)
	{
		for(int kp = 0; kp < paramSize-2; kp++)//paramSize-2, B, C off
		{

			int k_out = kp/out_len;
			int sw_len = ELWISE_FUNC_SEC*_sizeIn;
			int k_ws = (kp - k_out*out_len)/sw_len;
			int k_v = kp - k_out*out_len - k_ws*sw_len;

			if(k_v > 2*_sizeIn)
				continue;

			double grad = 0;
			if(kp < paramSize-2)
				 grad = _tempMatrixArray[kp].sum_fast(_aggStorage._aggMatrix, _aggStorage._srcCPU);
			else if(kp == paramSize-2)
				grad =0;// _tempC.sum_fast(_aggStorageC._aggMatrix, _aggStorageC._srcCPU);
			else if(kp == paramSize-1)
				grad = _tempB.sum_fast(_aggStorageC._aggMatrix, _aggStorageC._srcCPU);
				//grad = -6*(*_inputs[inpIdx]).sum()/(*_inputs[inpIdx]).getNumElements() - _param[paramSize-1];
			_grad[kp] = grad;
		}

		//MakeAuxParams();

	//test
		//if(0)
		if(minibatch == 0)
		{

		//debug
	//printf(" meta name %s \n", _name.c_str());
	//NVMatrix temp;
	//_inputs[inpIdx]->apply(NVMatrixOps::Abs(), temp);
	//float favg = temp.sum()/(*_inputs[inpIdx]).getNumElements();
	//printf("inp avg %f \n", favg);

			int numPixelPerGroup =  v.getNumElements()/_sizeOut;

			testGroupsEltwiseFunc(v, *_inputs[inpIdx],
									 _arrayPtr, _tempMatrixArray, _param,
									 _sizeIn, _sizeOut, _channels, 0);
			double gr0 = _tempMatrixArray[0].sum_fast(_aggStorage._aggMatrix, _aggStorage._srcCPU);
			double diff1 = _tempMatrixArray[1].sum_fast(_aggStorage._aggMatrix, _aggStorage._srcCPU);
			double diff2 = _tempMatrixArray[2].sum_fast(_aggStorage._aggMatrix, _aggStorage._srcCPU);
			printf("***EltwiseFunc %s group test gr0 %f diff1_rel %f diff2_rel %f \n",_name.c_str(), gr0/numPixelPerGroup, diff1/gr0, diff2/gr0);

			//double gr_s0 = _tempMatrixArray[3].sum_fast(_aggStorage._aggMatrix, _aggStorage._srcCPU)/numPixelPerGroup;
			//double gr_s1 = _tempMatrixArray[4].sum_fast(_aggStorage._aggMatrix, _aggStorage._srcCPU)/numPixelPerGroup;
			//double gr_s2 = _tempMatrixArray[5].sum_fast(_aggStorage._aggMatrix, _aggStorage._srcCPU)/numPixelPerGroup;
			//printf("gr_s0 %f gr_s1 %f gr_s2 %f\n", gr_s0, gr_s1, gr_s2);

	//end test
		}
	}

	computeEltwiseFuncGrad(v, *_inputs[inpIdx], _prev[inpIdx]->getActsGrad(), _param, _channels, _sizeIn, _sizeOut);

//		printf("EltwiseFuncLayer bpropActs end\n");
}

/* 
 * =======================
 * EltwiseDFuncLayer
 * =======================
 */
EltwiseDFuncLayer::EltwiseDFuncLayer(ConvNet* convNet, PyObject* paramsDict) : EltwiseFuncLayer(convNet, paramsDict){
//debug
	int paramSizeC = _param.size()-2;
	//for(int kp = paramSizeC/2; kp < paramSizeC; kp++)
	//	_param[kp] = 0;

}

void EltwiseDFuncLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {

	computeEltwiseDFuncAct(*_inputs[inpIdx],  getActs(), _param, _channels, _sizeIn, _sizeOut);
}

//static int debug_count = 0;

void EltwiseDFuncLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
//debug
//debug_count++;

	int paramSize = _param.size();

	float lim = 1./_param[paramSize-1];

	computeEltwiseDFuncParamWeightGrad(v, *_inputs[inpIdx],
								 _arrayPtr, _tempMatrixArray,
								 _tempC, _tempB, _param, lim,
								 _channels, _sizeIn, _sizeOut);

	_tempMatrixArray[0].ResizeAggStorage(_aggStorage._aggMatrix, _aggStorage._srcCPU);

	//_tempB.ResizeAggStorage(_aggStorageC._aggMatrix, _aggStorageC._srcCPU);B, C off

	if(passType == PASS_TRAIN)
	{
		for(int kp = 0; kp < paramSize-2; kp++)//paramSize-2, B, C off
		{

			double grad = 0;
			if(kp < paramSize-2)
				 grad = _tempMatrixArray[kp].sum_fast(_aggStorage._aggMatrix, _aggStorage._srcCPU);
			else if(kp == paramSize-2)
				grad =0;// _tempC.sum_fast(_aggStorageC._aggMatrix, _aggStorageC._srcCPU);
			else if(kp == paramSize-1)
				_tempB.sum_fast(_aggStorageC._aggMatrix, _aggStorageC._srcCPU);
			
	//should make orthognal projection to equal vector(sizeIn)

			double sum_grad = 0;
			for(int k = 0; k < _nstore; k++)
			{
				sum_grad += _grad_store[k*_param.size() + kp]*_grad_store[k*_param.size() + kp];
			}

			_grad_store[_nstore_count*_param.size() + kp] = grad;

			if(sum_grad > 0)
				grad = grad*sqrt(NSTORE)/sqrt(sum_grad);
		
			double eps = _epsP;
			double wc = _wc;

			_param_inc[kp] = _mom*_param_inc[kp] + eps*grad - wc*_param[kp];
				
	//debug
			if(kp != paramSize-2)
				_param[kp] += _param_inc[kp];
		}

		_nstore_count = (_nstore_count+1)%_nstore;

	#ifdef EL_SWITCH
		int paramSwSectionLen = (paramSize-2)/EL_SWITCH;
	#else
		int paramSwSectionLen = paramSize;
	#endif

		double sumScale = .5*_sizeIn*_sizeOut;
		int vect_len = _sizeIn*ELWISE_DFUNC_SEC;
		int vnorm_len = _sizeIn*3;


		//for(int k_sw = 0; k_sw < EL_SWITCH; k_sw++)
		 int k_sw = 0; 
		{
			double l1sum = 0;
			for(int k_out = 0; k_out < _sizeOut; k_out++)
			{
				for(int kinp = 0; kinp < vnorm_len; kinp++)
				{
					double pv = _param[k_out*EL_SWITCH*ELWISE_DFUNC_SEC*_sizeIn + k_sw*ELWISE_DFUNC_SEC*_sizeIn + kinp];
					l1sum += fabs(pv);
				}
			}

		
			assert(l1sum>0);

			for(int k_out = 0; k_out < _sizeOut; k_out++)
			{
				for(int kinp = 0; kinp < vnorm_len; kinp++)
					_param[k_out*EL_SWITCH*ELWISE_DFUNC_SEC*_sizeIn + k_sw*ELWISE_DFUNC_SEC*_sizeIn + kinp] *= sumScale/l1sum;
			}
		}


		if(minibatch == 0)
		{
			int nump = _sizeIn*ELWISE_DFUNC_SEC;
			int numl = (_param.size()+nump-1)/nump;
			printf("** params *** \n");
			for (int nk = 0; nk < numl; nk++)
			{
				for (int k = 0; k < nump; k++)
					if(k + nk*nump < _param.size())
					printf("%f ", _param[k + nk*nump]);
				printf("\n");
			}
		}

	}



	computeEltwiseDFuncGrad(v, *_inputs[inpIdx], _prev[inpIdx]->getActsGrad(), _param, _channels, _sizeIn, _sizeOut);

//		printf("EltwiseFuncLayer bpropActs end\n");

	if(_epsP >0 && minibatch == 0)
	{

		int numPixelPerGroup =  v.getNumElements()/_sizeOut;

		testGroupsEltwiseFunc(v, *_inputs[inpIdx],
								 _arrayPtr, _tempMatrixArray, _param,
								 _sizeIn, _sizeOut, _channels, 0);
		double gr0 = _tempMatrixArray[0].sum_fast(_aggStorage._aggMatrix, _aggStorage._srcCPU);
		double diff1 = _tempMatrixArray[1].sum_fast(_aggStorage._aggMatrix, _aggStorage._srcCPU);
		double diff2 = _tempMatrixArray[2].sum_fast(_aggStorage._aggMatrix, _aggStorage._srcCPU);
		printf("***EltwiseDFunc %s group test gr0 %f diff1_rel %f diff2_rel %f \n",_name.c_str(), gr0/numPixelPerGroup, diff1/gr0, diff2/gr0);


	}

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
	_startX = _start;
	_startY = _start;
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
    } else if(_pool == "maxabs") {
        return *new MaxAbsPoolLayer(convNet, paramsDict);
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
    convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _start, _stride, _outputsX, AvgPooler());
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

	//int rndstep = 2;
	//int csize = 6;


	//int rndX = rand()%(2*rndstep+csize);

	//if (rndX <  2*rndstep) 
	//	_startX = _start + rndX-rndstep;
	//else
	//	_startX = _start;

	//int rndY = rand()%(2*rndstep+csize);

	//if (rndY <  2*rndstep) 
	//	_startY = _start + rndY-rndstep;
	//else
	//	_startY = _start;

	//if(_name == "pool3" || gepoch >= 70)
	{
		_startX = _start;
		_startY = _start;
	}

    convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _startX, _startY, _stride, _outputsX, MaxPooler());
}

void MaxPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {

    convLocalMaxUndo(_prev[0]->getActs(), v, getActs(), _prev[inpIdx]->getActsGrad(), _sizeX, _startX, _startY, _stride, _outputsX, scaleTargets, 1);
}

/* 
 * =====================
 * MaxAbsPoolLayer
 * =====================
 */
MaxAbsPoolLayer::MaxAbsPoolLayer(ConvNet* convNet, PyObject* paramsDict) : PoolLayer(convNet, paramsDict, false) {
}

void MaxAbsPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _start, _stride, _outputsX, MaxAbsPooler());
}

void MaxAbsPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalMaxUndo(_prev[0]->getActs(), v, getActs(), _prev[inpIdx]->getActsGrad(), _sizeX, _start, _start, _stride, _outputsX, scaleTargets, 1);
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
    convLocalPool(images, _meanDiffs, _channels, _size, -_size/2, -_size/2, 1, _imgSize, AvgPooler());
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

double CostLayer::getErrorNum() {
    return _costv[1];
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
		acts.ResizeAggStorage(_aggStorage._aggMatrix, _aggStorage._srcCPU);
        _costv.clear();
        _costv.push_back(acts.sum_fast(_aggStorage._aggMatrix, _aggStorage._srcCPU));//should be max(1-t*act_prev, 0) instead
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
