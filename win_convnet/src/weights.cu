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

#include <weights.cuh>

bool Weights::_autoCopyToGPU = false;

int AUX_STORAGE = 2;
//#define USE_PREC

void Weights::shrink(float lambda)
{
	if (_wc > 0) {
		_weights->shrink(lambda);			
	}
};

extern int rnd_aux;

void Weights::procAux() {

	if(!_active_aux)
		return;

    assert(_onGPU);

	//if(!_weightsGrad->isSameDims(getAuxSum()));
	//	getAuxSum().resize(*_weightsGrad);

	//if(_aux_filled == 0)
	//{
	//	zeroAux();
	//}

	//if(_aux_filled >= 0)
	//	getAuxSum().add(*_weightsGrad, 1.);

	//if(_aux_filled >= _aux_store_size)
	//{
	//	assert(getAuxSum().isSameDims(getAuxUpdate()));
	//	getAuxSum().add(getAuxUpdate(), -1.);//remove
	//}

	//if(!_weightsGrad->isSameDims(getAuxUpdate()));
	//	getAuxUpdate().resize(*_weightsGrad);

	//_weightsGrad->copy(getAuxUpdate());

	//_aux_filled = min(_aux_filled+1, _aux_store_size);
	//_aux_update = (_aux_update+1)%_aux_store_size;

}

void Weights::stepAuxInd()
{
	//_aux_update = (_aux_update+1)%_aux_store_size;
	//_aux_filled = min(_aux_filled+1, _aux_store_size);
}


void Weights::zeroAux() {
	if(_active_aux)
		getAuxSum().apply(NVMatrixOps::Zero());
}

void Weights::zeroAux(int ind) {
	if(_active_aux)
		getAux(ind).apply(NVMatrixOps::Zero());
}

#define BASE 0
#define PREV_GRAD 1
void Weights::rollback(float reduceScale) 
{
#ifdef USE_PREC
	getAuxSum().add(*_weightsInc, reduceScale-1);
#else	
	_weights->add(*_weightsInc, reduceScale-1);
#endif
}

// Scale your gradient by epsW / numCases!
void Weights::update(bool useAux) {
    // Only true owner of weights updates
    if (_srcWeights == NULL && _epsW > 0) {

        assert(_onGPU);
        if (_useGrad) {

			float scaleGrad = 1;

			//if(_weightsGrad->isSameDims(getAux(PREV_GRAD)))
			//{
			//	//float g2 = _weightsGrad->norm2();
			//	//float g2prev = getAux(PREV_GRAD).norm2();
			//	float dot =   getAux(PREV_GRAD).dotProduct(*_weightsGrad);
			//	//printf("1e9x g2 %f dot/g2prev %f g2/g2prev %f\n", 1e9*g2, dot/g2prev, g2/g2prev);
			//	if(dot < 0)
			//		scaleGrad *= .2;
			//}	

			_weightsInc->add(*_weightsGrad, _mom, scaleGrad);

			//if(!_weightsGrad->isSameDims(getAux(PREV_GRAD)));
			//	getAux(PREV_GRAD).resize(*_weightsGrad);
			//_weightsGrad->copy(getAux(PREV_GRAD));

        }

        if (_wc > 0) {
            //_weightsInc->addSignReg(*_weights, -_wc * _epsW);	
			_weightsInc->add(*_weights, -_wc * _epsW);				
        }
  
#ifdef USE_PREC
		if(!_weightsInc->isSameDims(getAuxSum()))
		{
			getAuxSum().resize(*_weightsInc);
			getAux(BASE).resize(*_weights);
			zeroAux();
			_weights->copy(getAux(BASE));
			_aux_filled = 0;
		}

		getAuxSum().add(*_weightsInc);
		getAuxSum().add(getAux(BASE), *_weights);
		_aux_filled++;
#else
		_weights->add(*_weightsInc);
#endif
		_numUpdates = 0;

		if(_renorm > 0)
		{

			float norm2 =  _weights->norm2();
			int size = _weights->getNumElements();	
			float layerNorm = sqrtf(norm2/size);

			if(layerNorm > _renorm)
			{	
				float renormScale = _renorm/layerNorm;
				_weights->scale(renormScale);
				getAuxSum().scale(renormScale);
				getAux(BASE).scale(renormScale);
			}
		}
#ifdef USE_PREC
		if(_aux_filled >= 64)
		{
			_weights->copy(getAux(BASE));
			zeroAux();
			_aux_filled = 0;
		}
#endif
    }
}

float Weights::getNormL2Avg()
{

	float l2 = 0;
	for(int i = 0; i < _norms_filled; i++)
		l2 += _norms2[i];

	float ninv = 0;
	if(_norms_filled > 0)ninv = 1./_norms_filled;
	return sqrt(l2*ninv);
}


void Weights::copyToCPU() {

    if (_srcWeights == NULL) {
        assert(_onGPU);
        _weights->copyToHost(*_hWeights);
        _weightsInc->copyToHost(*_hWeightsInc);
//bregman
		//if(_active_aux && _hAux_weights)
		//{
		//	_aux_weights[_aux_update].copyToHost(*_hAux_weights);
		//}
    }
}

void Weights::initAux()
{
	_aux_filled = 0;
	_aux_update = 0;

	for(int i = 0; i < _full_store_size; i++)
		_aux_weights.push_back(NVMatrix());

	//if(!_weightsInc->isSameDims(getAux(0)))
	//	getAux(0).resize(*_weightsInc);

	//_weightsInc->copy(getAux(0));


	//if(!_weights->isSameDims(getAux(0)))
	//	getAux(0).resize(*_weights);

	//_weights->copy(getAux(0));

	//_aux_weights[0].copyFromHost(*_hAux_weights, true);

	//for(int i = 1; i < _full_store_size; i++)
	//{

	//	_aux_weights[i].resize(_aux_weights[0]);
	//	_aux_weights[i].apply(NVMatrixOps::Zero());
	//}

}

void Weights::CopyGradToAux()
{
	assert(_useGrad);
	getAuxUpdate().resize(*_weightsGrad);
	_weightsGrad->copy(getAuxUpdate());
};

void Weights::setAuxUpdateInd(int updInd)
{
	_aux_update = updInd;
}

int Weights::getAuxUpdateInd()
{
	return _aux_update;
}

void Weights::copyToGPU() {

    if (_srcWeights == NULL) {

        _weights = new NVMatrix();
        _weightsInc = new NVMatrix();
        _weights->copyFromHost(*_hWeights, true);
        _weightsInc->copyFromHost(*_hWeightsInc, true);
        _weightsGrad = _useGrad ? new NVMatrix() : NULL;

	    _onGPU = true;

		//bregman
		if(_active_aux)
			initAux();

    } else {
        _weights = _srcWeights->_weights;
        _weightsInc = _srcWeights->_weightsInc;
        _weightsGrad = _srcWeights->_weightsGrad;

	    _onGPU = true;

		//bregman
		if(_active_aux)
			initAux();
    }
}
    