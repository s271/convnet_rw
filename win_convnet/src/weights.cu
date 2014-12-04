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

void Weights::shrink(float lambda)
{
	if (_wc > 0) {
		_weights->shrink(lambda);			
	}
};

// Scale your gradient by epsW / numCases!
void Weights::update(bool use_inc_drop) {
    // Only true owner of weights updates
    if (_srcWeights == NULL && _epsW > 0) {
        assert(_onGPU);
        if (_useGrad) {
            _weightsInc->add(*_weightsGrad, _mom, 1);
        }

        if (_wc > 0) {
            _weightsInc->addSignReg(*_weights, -_wc * _epsW);			
        }

		if(use_inc_drop)
		{
		    _inc_drop->resize(_weightsInc->getNumRows(), _weightsInc->getNumCols());
			_inc_drop->randomizeUniform();
			_inc_drop->biggerThanScalar(.1);
		   _weightsInc->eltwiseMult(*_inc_drop);
		}

        _weights->add(*_weightsInc);
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
			}
		}

    }
}