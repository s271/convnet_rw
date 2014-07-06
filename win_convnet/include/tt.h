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

DEVICE void Split(const int t_ind, const int t_split, int& x, int& y)
{
	y = t_ind/t_split;
	x = t_ind%t_split;
}

#define SPLIT(src, split) int src##_##split##_x; int src##_##split##_y; Split(src, split, src##_##split##_x,  src##_##split##_y);

#define LOOP(i, loopBlock) for (int i = 0; i < loopBlock._idx[i##_]._size; i += loopBlock._idx[i##_]._step) 
#define OFFSET(i, indBlock) (i*indBlock._loop_step[i##_l])
#define OFFSETN(i, indBlock) (i*indBlock._loop_step[i##_##indBlock])
#define OFFSET_(i, l, indBlock) (i*indBlock._loop_step[l])

struct Index
{
	int _step;
	int _ind;
	DEVICE Index(const int step, const int ind){_step = step; _ind = ind;};
	DEVICE Index(const int ind){_step = 1; _ind = ind;};

};

struct LoopIndex
{
	int _step;
	int _size;
	int _start;
	int _pos;
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

struct Offset
{
	int _offset;

	DEVICE Offset(){_offset = 0;}

	DEVICE void Insert(const Index indx)
	{
		_offset += indx._step*indx._ind;
	}

	DEVICE Offset& operator<<(const Index indx)
	{
		Insert(indx);
		return *this;
	}

	DEVICE Offset& operator << (int shift)
	{
		_offset <<= shift;
		return *this;
	}
};

template <int loop_dims>
struct BaseIndex : Offset
{
	int _n_loop_dims;
	int _loop_step[loop_dims];

	DEVICE BaseIndex() : Offset() {_n_loop_dims = 0;}

	DEVICE int GetLoopStep(int pos)
	{
		return _loop_step[pos];
	}

	DEVICE BaseIndex& operator<<(const Index indx)
	{
		Offset::Insert(indx);
		return *this;
	}

	DEVICE void Insert(const Ref ref_indx)
	{
		_loop_step[_n_loop_dims] = 1;
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
		Insert(shift);
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

//---------



#endif	/* TENSOR_TEMPLATE_H */

