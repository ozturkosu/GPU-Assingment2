/*
 * sparse_representation.hpp
 * Copyright (C) 2018
 * 	P Sadayappan (saday) <psaday@gmail.com>
 * 	Aravind SUKUMARAN RAJAM (asr) <aravind_sr@outlook.com>
 *
 * Distributed under terms of the GNU LGPL3 license.
 */

#ifndef SPARSE_REPRESENTATION_HPP
#define SPARSE_REPRESENTATION_HPP

struct CSR
{
  int* row_indx[];
  int* col_id[];
	double* values[];

  int nrows;
  int ncols;
  int nnz;
};


struct CSC
{
  int* col_indx[];
  int* row_id[];
	double* values[];

	int nrows;
  int ncols;
  int nnz;
};

struct COO
{
  int* row_id[];
  int* col_id[];
	double* values[];

  int nrows;
  int ncols;
  int nnz;
};

#endif /* !SPARSE_REPRESENTATION_HPP */
