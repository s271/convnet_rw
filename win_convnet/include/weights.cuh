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

#ifndef WEIGHTS_CUH
#define	WEIGHTS_CUH

#include <string>
#include <vector>
#include <iostream>
#include <cutil_inline.h>
#include <assert.h>
#include <nvmatrix.cuh>
#include <matrix.h>
#include "util.cuh"

using namespace std;


class Weights {
private:
    Matrix* _hWeights, *_hWeightsInc;

    NVMatrix* _weights, *_weightsInc, *_weightsGrad;

//bregman
	Matrix *_hAux_weights;
	vector<NVMatrix> _aux_weights;

	int _aux_filled;
	int _aux_update;
	int _aux_store_size;
	int _full_store_size;
	bool _active_aux;

//rmsprop
	int _norms_size;
	int _norms_filled;
	int _norms_update;
	vector<float> _norms2;
	float _rmsW;

    float _epsW, _epsWinit, _epsWprev, _wc, _wc_init, _mom, _mom_init;

	float _muL1;

	float _renorm;

    bool _onGPU, _useGrad;
    int _numUpdates;
    static bool _autoCopyToGPU;
    
    // Non-NULL if these weights are really shared from some other layer
    Weights* _srcWeights;

	void initAux();
 
public:

    NVMatrix& operator*() {
        return getW();
    }
    
    Weights(Weights& srcWeights, float epsW, float rmsW) : _srcWeights(&srcWeights), _epsW(epsW), _epsWinit(epsW), _epsWprev(epsW),
											    _wc(0), _wc_init(0), _muL1(0), _renorm(0), _onGPU(false), _numUpdates(0),
                                               _weights(NULL), _weightsInc(NULL), _weightsGrad(NULL), _rmsW(rmsW),
											   _active_aux(false), _aux_store_size(0), _aux_filled(0),
											   _aux_update(0), _full_store_size(0) {
        _hWeights = &srcWeights.getCPUW();
        _hWeightsInc = &srcWeights.getCPUWInc();
//bregman
//unfinished 
        _mom = srcWeights.getMom();
		_mom_init = _mom;
        _useGrad = srcWeights.isUseGrad();  
//rmsprop
	 _norms_size = 0;
	 _norms_filled = 0;
	 _norms_update = 0;

        if (_autoCopyToGPU) {
            copyToGPU();
        }
    }

    Weights(Matrix& hWeights, Matrix& hWeightsInc,
		float epsW, float rmsW, float wc, float mom, float muL1, float renorm, bool useGrad)
        : _srcWeights(NULL), _hWeights(&hWeights), _hWeightsInc(&hWeightsInc), _rmsW(rmsW),
		_hAux_weights(NULL),
			_numUpdates(0),
          _epsW(epsW), _epsWinit(epsW), _epsWprev(epsW), _wc(wc), _wc_init(wc), _mom(mom), _mom_init(mom), _muL1(muL1),
		  _renorm(renorm), _useGrad(useGrad), _onGPU(false), _weights(NULL), _active_aux(false),
		  _aux_store_size(0), _aux_filled(0), _aux_update(0), _full_store_size(0),
          _weightsInc(NULL), _weightsGrad(NULL) {
//rmsprop
	 _norms_size = 0;
	 _norms_filled = 0;
	 _norms_update = 0;

        if (_autoCopyToGPU) {
            copyToGPU();
        }
    }
      
    
    Weights(Matrix& hWeights, Matrix& hWeightsInc,
//bregman
		Matrix& hAux_weights, bool active_aux, int aux_store_size,
		float epsW, float rmsW, float wc, float mom, float muL1, float renorm, bool useGrad)
        : _srcWeights(NULL), _hWeights(&hWeights), _hWeightsInc(&hWeightsInc), _rmsW(rmsW),
//bregman
	_hAux_weights(&hAux_weights),
	_active_aux(active_aux),
	_aux_store_size(aux_store_size),
	_full_store_size(aux_store_size+1),
	_aux_filled(0),
	_aux_update(0),

		_numUpdates(0), _epsW(epsW), _epsWinit(epsW), _epsWprev(epsW),
		_wc(wc), _wc_init(wc), _mom(mom), _mom_init(mom), _muL1(muL1), _renorm(renorm),
		_useGrad(useGrad), _onGPU(false), _weights(NULL), 
        _weightsInc(NULL), _weightsGrad(NULL) {

//rmsprop
	 _norms_size = 0;
	 _norms_filled = 0;
	 _norms_update = 0;

        if (_autoCopyToGPU) {
            copyToGPU();
        }
    }
        
    ~Weights() {
        delete _hWeights;
        delete _hWeightsInc;
        if (_srcWeights == NULL) {
            delete _weights;
            delete _weightsInc;
            delete _weightsGrad;
        }
    }

    static void setAutoCopyToGPU(bool autoCopyToGPU) {
        _autoCopyToGPU = autoCopyToGPU;
    }
    
    NVMatrix& getW() {
        assert(_onGPU);
        return *_weights;
    }

    NVMatrix& getAuxUpdate() {
        assert(_onGPU);
        return  _aux_weights[_aux_update];
    }

    NVMatrix& getAuxSum() {
        assert(_onGPU);
        return _aux_weights[_aux_store_size];
    }

    NVMatrix& getAux(int ind) {
        assert(_onGPU);
        return  _aux_weights[ind];
    }

	float& getNorm2(int ind)
	{
		return _norms2[ind];
	}

	float& getNorm2Update()
	{
		return _norms2[_norms_update];
	}

	float getNormL2Avg();

	void CopyGradToAux();
    
    NVMatrix& getInc() {
        assert(_onGPU);
        return *_weightsInc;
    }
        
    NVMatrix& getGrad() {
        assert(_onGPU);
		//debug aux
		assert(_useGrad);
        return _useGrad ? *_weightsGrad : *_weightsInc;
    }

	void setAuxUpdateInd(int updInd);

	int getAuxUpdateInd();

	void stepAuxInd();
    
    Matrix& getCPUW() {
        return *_hWeights;
    }
    
    Matrix& getCPUWInc() {
        return *_hWeightsInc;
    }
    
    int getNumRows() const {
        return _hWeights->getNumRows();
    }
    
    int getNumCols() const {
        return _hWeights->getNumCols();
    }
    
    void copyToCPU(); 
    
    // This function is assumed to be called in the order in which the layers
    // were defined
    void copyToGPU();

    // Scale your gradient by epsW / numCases!
    void update(bool useAux);
	void rollback(float reduceScale);

	void procAux();

	void zeroAux();

	void zeroAux(int ind);

	void shrink(float lambda);
    
    int incNumUpdates() {
        if (_srcWeights != NULL) {
            return _srcWeights->incNumUpdates();
        }
        return _numUpdates++;
    }
    
    // Returns the number of times a gradient has been computed for this
    // weight matrix during the current pass (interval between two calls of update())
    // through the net. This number will only be greater than 1 if this weight matrix
    // is *shared* by multiple layers in the net.
    int getNumUpdates() const {
        if (_srcWeights != NULL) {
            return _srcWeights->getNumUpdates();
        }
        return _numUpdates;
    }
    
    float getEps() const {
        return _epsW;
    }

    float getEpsInit() const {
        return _epsWinit;
    }

    float getWcInit() const {
        return _wc_init;
    }

    void setEps(float epsW)  {
        _epsW = epsW;
    }

    void setWc(float wc)  {
        _wc = wc;
    }
    
    float getMom() const {
        return _mom;
    }

    float getMomInit() const {
        return _mom_init;
    }

    void setMom(float mom) {
        _mom = mom;
    }
    
    float getWC() const {
        return _wc;
    }
    
    bool isUseGrad() const { // is good grammar
        return _useGrad;
    }
};

class WeightList {
private:
    std::vector<Weights*> _weightList;

public:
    Weights& operator[](const int idx) const {
        return *_weightList[idx];
    }
    
    ~WeightList() {
        for (int i = 0; i < _weightList.size(); i++) {
            delete _weightList[i];
        }
    }
    
//    WeightList(MatrixV& hWeights, MatrixV& hWeightsInc, floatv& epsW, floatv& wc, floatv& mom, bool useGrads) : _initialized(false) {
//        initialize(hWeights, hWeightsInc, epsW, wc, mom, useGrads);
//    }
    
    WeightList() {
    }
    
//    void initialize(MatrixV& hWeights, MatrixV& hWeightsInc, floatv& epsW, floatv& wc, floatv& mom, bool useGrads) {
//        for (int i = 0; i < hWeights.size(); i++) {
//            _weightList.push_back(new Weights(*hWeights[i], *hWeightsInc[i], epsW[i], wc[i], mom[i], useGrads));
//        }
//        _initialized = true;
//        delete &hWeights;
//        delete &hWeightsInc;
//        delete &epsW;
//        delete &wc;
//        delete &mom;
//    }
    
    void addWeights(Weights& w) {
        _weightList.push_back(&w);
    }
    
//    void addWeights(WeightList& wl) {
//        for (int i = 0; i < wl.getSize(); i++) {
//            addWeights(wl[i]);
//        }
//    }
    
    void update(bool useAux) {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->update(useAux);
        }
    }

    void rollback(float reduceScale) {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->rollback(reduceScale);
        }
    }

    void procAux() {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->procAux();
        }
    }

    void zeroAux() {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->zeroAux();
        }
    }

    void shrink(float lambda) {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->shrink(lambda);
        }
    }


    void copyToCPU() {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->copyToCPU();
        }
    }

    void copyToGPU() {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->copyToGPU();
        }
    }
    
    int getSize() {
        return _weightList.size();
    }
};

//debug aux
extern int AUX_STORAGE;

#endif	/* WEIGHTS_CUH */