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
#define DEVICE_ __device__
#else
#define DEVICE
#define DEVICE_
#endif

typedef int CudaPos;
typedef int LoopPos;

DEVICE void Split(const CudaPos t_ind, const CudaPos t_split, CudaPos& x, CudaPos& y)
{
	y = t_ind/t_split;
	x = t_ind%t_split;
}

//#define SPLIT(src, split) CudaPos src##_x; CudaPos src##_y; Split(src, split, src##_x, src##_y);
#define SPLIT(src, split) CudaPos src##_##split##_x; CudaPos src##_##split##_y; Split(src, split, src##_##split##_x,  src##_##split##_y);

struct Index
{
	int _step;
	CudaPos _ind;
	DEVICE Index(const int step){_step = step; _ind = 0;};
	DEVICE Index(const int step, const int ind){_step = step; _ind = ind;};

};

struct LoopIndex
{
	int _step;
	int _size;
	int _start;
	LoopPos _pos;
	DEVICE LoopIndex(){};
	DEVICE LoopIndex(const int step){_step = step;};
	DEVICE LoopIndex(const int step, const int size){_step = step; _size = size;};

};

template <int loops>
struct LoopBlock
{
	int _nloops;
	LoopIndex _idx[loops];
	DEVICE LoopBlock(){_nloops = 0;};
	DEVICE LoopBlock<loops>& operator<(LoopIndex& indx)
	{
		_idx[_nloops] = indx;
		_nloops++;
		return *this;
	}
	DEVICE LoopBlock<loops>& operator>(int& name)
	{
		name = _nloops;
		return *this;
	}
};

struct Ref
{
	int _pos;
	DEVICE Ref(int pos){_pos =pos;};
};

struct RefSplitY
{
	int _pos;
	int _split;
	int _add;
	DEVICE RefSplitY(int split, int add, int pos){ _split = split; _add = add; _pos =pos;};
};

struct RefSplitX
{
	int _pos;
	DEVICE RefSplitX(int pos){_pos =pos;};
};

struct SplitPos
{
	int _pos;
};

template <int loop_dims>
struct BaseIndex
{
	int _offset;
	int _n_loop_dims;
	LoopPos _loop_step[loop_dims];
	LoopPos _loop_ref[loop_dims];


	DEVICE BaseIndex(){_n_loop_dims = 0; _offset = 0;}

	DEVICE int GetLoopStep(int pos)
	{
		return _loop_step[pos];
	}

	DEVICE void Insert(const Index indx)
	{
		_offset += indx._step*indx._ind;
	}

	DEVICE BaseIndex& operator<<(const Index indx)
	{
		Insert(indx);
		return *this;
	}

	DEVICE void Insert(const Ref ref_indx)
	{
		_loop_step[_n_loop_dims] = 1;
		_loop_ref[_n_loop_dims] = ref_indx._pos;
		_n_loop_dims++;	
	}

	DEVICE BaseIndex& operator<<(Ref ref_indx)
	{
		Insert(ref_indx);
		return *this;
	}

	DEVICE void Insert (int shift)
	{
		_offset <<= shift;

#ifdef  __CUDACC__
#pragma unroll
#endif
		for (int k = 0; k < _n_loop_dims; k++)
		{
			_loop_step[k] <<= shift;
		}	
	}

	DEVICE BaseIndex& operator << (int shift)
	{
		Insert(shif);
		return *this;
	}

	DEVICE void Out(int& pos)
	{
		pos =_n_loop_dims;
	}

	DEVICE BaseIndex& operator >>(int& pos)
	{
		Out(pos);
		return *this;
	}

};

template <int loop_dims, int split_dims>
struct SBaseIndex : BaseIndex<loop_dims>
{

	int _n_split_dims;
	int _split_stepX[split_dims];
	int _split_stepY[split_dims];
	int _split[split_dims];
	int _split_add[split_dims];
	int _split_ref[split_dims];

	DEVICE SBaseIndex() : BaseIndex<loop_dims>() {_n_split_dims = 0;}

	DEVICE SBaseIndex& operator<<(const RefSplitY& ref_indx)
	{
		_split_stepX[_n_split_dims] = 1;
		_split_stepY[_n_split_dims] = 0;
		_split_ref[_n_split_dims] = ref_indx._pos;
		_split_add[_n_split_dims] = ref_indx._add;
		_split[_n_split_dims] = ref_indx._split;
		_n_split_dims++;
		return *this;
	}

	DEVICE SBaseIndex& operator<<(RefSplitX ref_indx)
	{
		for(int k =0; k < _n_split_dims; k++)
		if(_split_ref[k] == ref_indx._pos)
			_split_stepY[k] = 1;
		return *this;
	}

	DEVICE void ShiftOther(int shift)
	{
		#ifdef  __CUDACC__
#pragma unroll
#endif
		for (int k = 0; k < _n_split_dims; k++)
		{
			_split_stepX[k] <<= shift;
			_split_stepY[k] <<= shift;
		}
	};

	DEVICE BaseIndex& operator >>(SplitPos& split_pos)
	{
		split_pos._pos =_n_split_dims;
		return *this;
	}

	DEVICE SBaseIndex& operator<<(const Index indx) 
	{
		Insert(indx);
		return *this;
	}

	DEVICE SBaseIndex& operator<<(const Ref ref_indx)
	{
		Insert(ref_indx);
		return *this;
	}

	DEVICE SBaseIndex& operator << (int shift)
	{
		Insert(shift);
		ShiftOther(shift);
		return *this;
	}

	DEVICE SBaseIndex& operator >>(int& pos)
	{
		Out(pos);
		return *this;
	}

};
//---------



#endif	/* TENSOR_TEMPLATE_H */

