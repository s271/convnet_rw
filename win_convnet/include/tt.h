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

struct Shift
{
	int _ishift;
	int _sshift;
	Shift(int ishift, int sshift){_ishift = ishift, _sshift = sshift;};
};

struct Index
{
	int _step;
	int _ind;
	Index(){_ind = 0;};
	Index(const int step){_step = step; _ind = 0;};
	Index(const int step, const int ind){_step = step; _ind = ind;};
	Index& operator<<(int shift)
	{
		_step <<= shift;
		return *this;
	}

	Index& operator<<(const Shift& shift)
	{
		_step <<= shift._sshift;
		_ind <<= shift._ishift;
		return *this;
	}
};

template <int dims>
struct BaseIndex
{
	int _ndims;
	int _dimSize;
	int _step[dims];
	int _ind[dims];

	BaseIndex(){_ndims = 0; _dimSize = dims; memset(_ind, 0, sizeof(_ind));}

	BaseIndex<dims+1> divisorSplit(int ngroups, int pos)
	{
		BaseIndex<dims+1> b_div;

//#pragma unroll
		for (int i = 0; i < pos; i++)
			b_div.Insert(_step[i]);

			b_div.Insert(ngroups);
			b_div.Insert(_step[pos]/ngroups);

//#pragma unroll
		for (int i = pos+1; i < _ndims; i++)
			b_div.Insert(_step[i]);
		
		return b_div;
	}

	BaseIndex<dims+1> quotientSplit(int quotient, int pos)
	{
		BaseIndex<dims+1> b_div;

//#pragma unroll
		for (int i = 0; i < pos; i++)
			b_div.Insert(_step[i]);

			b_div.Insert(_step[pos]/quotient);
			b_div.Insert(quotient);

//#pragma unroll
		for (int i = pos+1; i < _ndims; i++)
			b_div.Insert(_step[i]);
		
		return b_div;
	}

	BaseIndex<dims>& Insert(int step)
	{
		_step[_ndims] = step;
		_ndims++;
		return *this;
	}

	BaseIndex<dims>& Insert(Index& indx)
	{
		_step[_ndims] = indx._step;
		_ind[_ndims] = indx._ind;
		_ndims++;
		return *this;
	}

	template <class TBase>
	BaseIndex<dims>& Insert(TBase insBase)
	{
//#pragma unroll
		for(int k_ins = 0; k_ins < insBase._ndims; k_ins++)
		{
			_step[_ndims + k_ins] = insBase._step[k_ins];
			_ind[_ndims + k_ins] = insBase._ind[k_ins];
		}
		_ndims+=insBase._ndims;
		return *this;
	}

	template <class TBase>
	BaseIndex<dims>& operator<(TBase insBase)
	{
		return Insert<TBase>(insBase);
	}

	BaseIndex<dims>& operator<(Index insBase)
	{
		return Insert(insBase);
	}

	BaseIndex<dims>& operator<(int step)
	{
		return Insert(step);
	}

	BaseIndex<dims>& operator<<(int shift)
	{
		for(int k = 0; k < _ndims; k++);
			_step[k] <<= shift;
		return *this;
	}

	BaseIndex<dims>& operator<<(const Shift& shift)
	{
		for(int k = 0; k < _ndims; k++);
		{
			_step[k] <<= shift._sshift;
			_ind[k] <<= shift._ishift;
		}
		return *this;
	}

#ifndef CUDA_KERNEL
	void Assert()
	{
		assert(_ndims == dims);
	}

#endif

};

struct DimIndex
{
	int _dim;
	int _ind;
	DimIndex(){_ind = 0;};
	DimIndex(const int dim){_dim = dim; _ind = 0;};
	DimIndex(const int dim, const int ind){_dim = dim; _ind = ind;};
};

template <int dims>
struct BaseDimIndex
{
	int _ndims;
	int _dimSize;
	int _step[dims];
	int _dim[dims];
	int _ind[dims];

	BaseDimIndex(){_ndims = 0; _dimSize = dims; memset(_ind, 0, sizeof(_ind));}

	BaseDimIndex<dims>& Insert(int dim)
	{
		_dim[_ndims] = dim;
		_ndims++;
		if(_ndims == dims)
			Finalize();
		return *this;
	}

	BaseDimIndex<dims>& Insert(DimIndex& indx)
	{
		_dim[_ndims] = indx._dim;
		_ind[_ndims] = indx._ind;
		_ndims++;
		if(_ndims == dims)
			Finalize();
		return *this;
	}

	BaseDimIndex<dims>& operator<(DimIndex insBase)
	{
		return Insert(insBase);
	}

	BaseDimIndex<dims>& operator<(int dim)
	{
		return Insert(dim);
	}

	void Finalize(int stepLow = 1)
	{
#ifndef CUDA_KERNEL
		Assert();
#endif
		_step[dims-1] = stepLow;

		for(int k = dims-2; k >= 0; k--)
			_step[k] = _dim[k]*_step[k+1];
	}

#ifndef CUDA_KERNEL
	void Assert()
	{
		assert(_ndims == _dimSize);
	}

#endif

};






#endif	/* TENSOR_TEMPLATE_H */

