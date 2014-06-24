/* 
 * Copyright (c) 2014, Sergey Ten
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

#ifndef TENSOR_TEMPLATE_H
#define	TENSOR_TEMPLATE_H
#include <assert.h>

#ifdef  __CUDACC__
#define DEVICE __device__ inline
#else
#define DEVICE
#endif

typedef int CudaPos;
typedef int LoopPos;

DEVICE void Split(CudaPos t_ind, CudaPos t_split, CudaPos& x, CudaPos& y)
{
	y = t_ind/t_split;
	x = t_ind%t_split;
}

#define SPLIT(src, split) CudaPos src##_x; CudaPos src##_y; Split(src, split, src##_x, src##_y);

struct Index
{
	int _step;
	CudaPos _ind;
	DEVICE Index(const int step){_step = step; _ind = 0;};
	DEVICE Index(const int step, const int ind){_step = step; _ind = ind;};
	DEVICE Index& operator<<(int shift)
	{
		_step <<= shift;
		return *this;
	}
};

struct LoopIndex
{
	int _step;
	LoopPos _pos;
	DEVICE LoopIndex(){};
	DEVICE LoopIndex(const int step){_step = step;};
	DEVICE LoopIndex& operator<<(int shift)
	{
		_step <<= shift;
		return *this;
	}
};

template <int loop_dims>
struct BaseIndex
{
	int _offset;
	int _n_loop_dims;
	LoopPos _loop_step[loop_dims];
	DEVICE BaseIndex(){_n_loop_dims = 0; _offset = 0;}

	DEVICE int GetFreeStep(int pos)
	{
		return _loop_step[pos];
	}

	DEVICE BaseIndex<loop_dims>& operator<(const Index indx)
	{
		_offset += indx._step*indx._ind;
		return *this;
	}

	DEVICE BaseIndex<loop_dims>& operator<(LoopIndex& indx)
	{
		_loop_step[_n_loop_dims] = indx._step;
		indx._pos = _n_loop_dims;
		_n_loop_dims++;
		return *this;
	}


};
//---------
/*
template <int dims>
struct BaseIndex
{
	int _ndims;
	int _dimSize;
	int _step[dims];
	int _ind[dims];

	DEVICE BaseIndex(){_ndims = 0; _dimSize = dims; memset(_ind, 0, sizeof(_ind));}

	DEVICE int GetLowBase(int pos)
	{
		int offset = 0;
#ifdef  __CUDACC__
#pragma unroll
#endif
		for(int k = pos+1; k < _ndims; k++)
			offset += _ind[k]*_step[k];
		return offset;
	}

	DEVICE int GetStep(int pos)
	{
		return _step[pos];
	}

	DEVICE BaseIndex<dims>& Insert(int step)
	{
		_step[_ndims] = step;
		_ndims++;
		return *this;
	}

	DEVICE BaseIndex<dims>& Insert(const Index& indx)
	{
		_step[_ndims] = indx._step;
		_ind[_ndims] = indx._ind;
		_ndims++;
		return *this;
	}

	DEVICE BaseIndex<dims>& Insert(SIndex& indx)
	{
		_step[_ndims] = indx._step;
		_ind[_ndims] = 0;
		indx._pos = _ndims;
		_ndims++;
		return *this;
	}

	template <int dims_ins>
	DEVICE BaseIndex<dims>& Insert(const BaseIndex<dims_ins> insBase)
	{
#ifdef  __CUDACC__
#pragma unroll
#endif
		for(int k_ins = 0; k_ins < insBase._ndims; k_ins++)
		{
			_step[_ndims + k_ins] = insBase._step[k_ins];
			_ind[_ndims + k_ins] = insBase._ind[k_ins];
		}
		_ndims+=insBase._ndims;
		return *this;
	}

	template <int dims_ins>
	DEVICE BaseIndex<dims>& operator<(const BaseIndex<dims_ins> insBase)
	{
		return Insert<dims_ins>(insBase);
	}

	DEVICE BaseIndex<dims>& operator<(Index insBase)
	{
		return Insert(insBase);
	}

	DEVICE BaseIndex<dims>& operator<(SIndex insBase)
	{
		return Insert(insBase);
	}

	DEVICE BaseIndex<dims>& operator<(int step)
	{
		return Insert(step);
	}

	DEVICE BaseIndex<dims>& operator<<(int shift)
	{
#ifdef  __CUDACC__
#pragma unroll
#endif
		for(int k = 0; k < _ndims; k++);
			_step[k] <<= shift;
		return *this;
	}

	DEVICE BaseIndex<dims>& operator<<(const Shift& shift)
	{
#ifdef  __CUDACC__
#pragma unroll
#endif
		for(int k = 0; k < _ndims; k++);
		{
			_step[k] <<= shift._sshift;
			_ind[k] <<= shift._ishift;
		}
		return *this;
	}

#ifndef  __CUDACC__
	void Assert()
	{
		assert(_ndims == dims);
	}

#endif

};

#define BASE_LOOP(ii, indx, base) for (uint ii = base.GetLowBase(indx._pos); ii < indx._size; ii += base.GetStep(indx._pos)) 

struct DimIndex
{
	int _dim;
	int _ind;
	DEVICE DimIndex(){_ind = 0;};
	DEVICE DimIndex(const int dim){_dim = dim; _ind = 0;};
	DEVICE DimIndex(const int dim, const int ind){_dim = dim; _ind = ind;};
};

template <int dims>
struct BaseDimIndex : public BaseIndex<dims>
{

	int _dim[dims];

	DEVICE BaseDimIndex(){_ndims = 0; _dimSize = dims; memset(_ind, 0, sizeof(_ind));}

	DEVICE BaseDimIndex<dims>& Insert(int dim)
	{
		_dim[_ndims] = dim;
		_ndims++;
		if(_ndims == dims)
			Finalize();
		return *this;
	}

	DEVICE BaseDimIndex<dims>& Insert(DimIndex& indx)
	{
		_dim[_ndims] = indx._dim;
		_ind[_ndims] = indx._ind;
		_ndims++;
		if(_ndims == dims)
			Finalize();
		return *this;
	}

	DEVICE BaseDimIndex<dims>& operator<(DimIndex insBase)
	{
		return Insert(insBase);
	}

	DEVICE BaseDimIndex<dims>& operator<(int dim)
	{
		return Insert(dim);
	}

	DEVICE void Finalize(int stepLow = 1)
	{
#ifndef  __CUDACC__
		Assert();
#endif
		_step[dims-1] = stepLow;
#ifdef  __CUDACC__
#pragma unroll
#endif
		for(int k = dims-2; k >= 0; k--)
			_step[k] = _dim[k]*_step[k+1];
	}

};


*/



#endif	/* TENSOR_TEMPLATE_H */

