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

template<class IndexType>
struct Index
{
	IndexType _width;
	IndexType _i;
	Index(IndexType width, IndexType ind){_width = width; _i = ind;}
	Index(){_i = 0;};
};



template<class IndexTypeIn, class IndexTypeOut>
void SplitCounter(IndexTypeIn& inIndex, int split, IndexTypeOut& outIndex1, IndexTypeOut& outIndex2)
{
	outIndex1 = IndexTypeOut(inIndex._width/split, inIndex._i/split);
	outIndex2 = IndexTypeOut(split, inIndex._i%split);
}


template <int dims>
struct BaseIndex
{
	int _ndims;
	int _dimSize;
	int _dim[dims];
	int _ind[dims];

	BaseIndex(){_ndims = 0; _dimSize = dims; memset(_ind, 0, sizeof(_ind));};

	BaseIndex<dims+1> divisorSplit(int pos, int ngroups)
	{
		BaseIndex<dims+1> b_div;

#pragma unroll
		for (int i = 0; i < pos; i++)
			b_div.Insert(_dim[i]);

			b_div.Insert(ngroups);
			b_div.Insert(_dim[pos]/ngroups);

#pragma unroll
		for (int i = pos+1; i < _ndims; i++)
			b_div.Insert(_dim[i]);
		
		return b_div;
	}

	BaseIndex<dims+1> quotientSplit(int pos, int quotient)
	{
		BaseIndex<dims+1> b_div;

#pragma unroll
		for (int i = 0; i < pos; i++)
			b_div.Insert(_dim[i]);

			b_div.Insert(_dim[pos]/quotient);
			b_div.Insert(quotient);

#pragma unroll
		for (int i = pos+1; i < _ndims; i++)
			b_div.Insert(_dim[i]);
		
		return b_div;
	}

	int Insert(int dim_size){
		_dim[_ndims] = dim_size;
		_ndims++;
		return _ndims;};

	int operator<<(int dim_size)
	{
		return Insert(dim_size);
	}

	template <class TBase>
	int Insert(TBase insBase){
#pragma unroll
		for(int k_ins = 0; k_ins < insBase._ndims; k_ins++)
		{
			_dim[_ndims + k_ins] = insBase._dim[k_ins];
			_ind[_ndims + k_ins] = insBase._ind[k_ins];
		}
		_ndims+=insBase._ndims;
		return _ndims;};

	template <class TBase>
	int Insert(TBase insBase, int pos){
#pragma unroll
		for(int k_m = 0; k_m < _ndims-pos; k_m++)
		{
			_dim[pos + k_m + insBase._ndims] = _dim[pos + k_m];
			_ind[pos + k_m + insBase._ndims] = _ind[pos + k_m];
		}
#pragma unroll
		for(int k_ins = 0; k_ins < insBase._ndims; k_ins++)
		{
			_dim[pos + k_ins] = insBase._dim[k_ins];
			_ind[pos + k_ins] = insBase._ind[k_ins];
		}
		_ndims+=insBase._ndims;
		return _ndims;};

	template <class TBase>
	int operator<<(TBase insBase)
	{
		return Insert<TBase>(insBase);
	}

};






#endif	/* TENSOR_TEMPLATE_H */

