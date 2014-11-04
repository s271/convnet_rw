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

#ifndef LAYER_KERNELS_CUH
#define	LAYER_KERNELS_CUH

#include <cutil_inline.h>
#include <nvmatrix.cuh>
#include <vector>
using namespace std;

#define LOGREG_GRAD_THREADS_X      32
#define LOGREG_GRAD_THREADS_Y      4

#define LOGREG_ERR_THREADS_X        128
#define LOGREG_ERR_THREADS_Y        1

void computeLogregCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out);
void computeLogregGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff);

void computeL2SVMCost(NVMatrix& labels, NVMatrix& input, NVMatrix& cost_out, NVMatrix& correctPreds_out);
void computeL2SVMGrad(NVMatrix& labels, NVMatrix& acts, NVMatrix& target, bool add, float coeff);

void computeRLogCost(NVMatrix& labels, NVMatrix& probs,
					 NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out, NVMatrix& probWeights_out,
					 float p_pow);

void computeRLogGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target,  NVMatrix& probWeights,
					bool add, float coeff);

void computeSoftmaxGrad(NVMatrix& acts, NVMatrix& actsGrad, NVMatrix& target, bool add);

// Numerical stability optimization: this routine combines computeLogregGrad with computeSoftmaxGrad
// to avoi dividing and then multiplying by quantities that may be near zero.
void computeLogregSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff);
void computeRLogSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, NVMatrix& probWeights, bool add, float coeff);

#define ELWISE_FUNC_SEC 2

void computeEltwiseMaxGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& output, NVMatrix& target, bool add);

void computeEltwiseFuncParamGradSingle(NVMatrix& actGrad, NVMatrix& input,
								 NVMatrix& target, NVMatrix& target_m,
								 int pin, int pout, int size_in, int size_out);

void computeEltwiseFuncParamWeightGrad(NVMatrix& actGrad, NVMatrix& input,
								 void* arrayPtr, vector<NVMatrix>& tempMatrix, vector<double>& param,
								 int channels, int size_in, int size_out);

void computeEltwiseFuncGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& target,
								 vector<double>& param, int channels, int size_in, int size_out);

void computeEltwiseFuncAct(NVMatrix& input, NVMatrix& target, vector<double>& param, int channels, int size_in, int size_out);

void testGroupsEltwiseFunc(NVMatrix& actGrad, NVMatrix& input,
								 void* arrayPtr, vector<NVMatrix>& tempMatrix,
								 vector<double>& param,
								 int size_in, int size_out);

void computeMicroConvAct(NVMatrix& input, NVMatrix& target, vector<double>& param, int sizeX, int channels,
						 int imgSize, int imgPixels, int numFilters);
void computeMicroConvActGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& target,
							 vector<double>& param, int sizeModuleSide, int channels,
							int imgSize, int imgPixels, int numFilters);
void computeMicroConvWeightGrad(NVMatrix& actGrad, NVMatrix& input,
								vector<NVMatrix>& tempMatrix, void* arrayPtr,
								vector<double>& param, int sizeModuleSide, int channels,
								int imgSize, int imgPixels, int numFilters);

void computeVectFuncAct(NVMatrix& input, NVMatrix& target, vector<double>& param, int sizeV, int sizeH, int channels);

void computeVectFuncGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& target,
								 vector<double>& param,  int sizeV, int sizeH, int channels);
void computeVectFuncWeightGrad(NVMatrix& actGrad, NVMatrix& input,
								vector<NVMatrix>& tempMatrix,
								void* arrayPtr,
								vector<double>& param,  int sizeV, int sizeH, int channels);

void computeMAvgAct(NVMatrix& input, NVMatrix& target, int sizeModuleSide, int channels,
						 int imgSize, int imgPixels);

void computeMAvgGrad(NVMatrix& actGrad,  NVMatrix& target, int sizeModuleSide, int channels,
						 int imgSize, int imgPixels);

#endif	/* LAYER_KERNELS_CUH */

