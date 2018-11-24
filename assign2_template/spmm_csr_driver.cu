/*
 * spmm_csr_driver.cu
 * Copyright (C) 2018
 *  P Sadayappan (saday) <psaday@gmail.com>
 *  Aravind SUKUMARAN RAJAM (asr) <aravind_sr@outlook.com>
 *
 * Distributed under terms of the GNU LGPL3 license.
 */

#include "mm_helper.hpp"
#include "sparse_representation.hpp"
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define TILE_WIDTH 32

void check_dmat(double* a, double *b, unsigned int n, unsigned int K, bool quit_on_err = true ) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int k = 0; k < K; ++k) {
            if(std::abs(a[i * K + k] - b[i * K + k]) > 1e-1) {
                std::cerr << "Possible error at " << i << std::endl;

                if(quit_on_err) {
                    exit(-1);
                }
            }
        }
    }

    if(quit_on_err)
        std::cout << "Verification succeeded\n";
    else
        std::cout << "Check error messages to see if verification succeeded. (No error msg == success)\n";
}

static unsigned int g_seed = 0X4B1D;
inline int fastrand() {
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x7FFF;
}

void init_dmat(double *a, unsigned int n, unsigned int K, double offset) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int k = 0; k < K; ++k) {
            a[i * K + k]  = i * K + k + offset;
            //a[i * K + j]  = fastrand() + offset;
        }
    }
}

void print_dmat(double *a, unsigned int n, unsigned int K) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < K; ++j) {
            std::cout << a[i * K + j]   << ' ';
        }
        std::cout << '\n';
    }
}

void print_CSR(CSR &mat) {
    for (unsigned int r = 0; r < mat.nrows; ++r) {
        unsigned int row_start = mat.row_indx[r];
        unsigned int row_end = mat.row_indx[r + 1];
        for (unsigned int j = row_start; j < row_end; ++j) {
            unsigned int col_id = mat.col_id[j];
            double val = mat.values[j];

	    std::cout << r << ' ' << col_id << ' ' <<  val << '\n';
        }
    }
}

void host_csr_spmm(CSR &mat, double * dmat_in, double * dmat_out, unsigned int K) {
    for (unsigned int r = 0; r < mat.nrows; ++r) {
        unsigned int row_start = mat.row_indx[r];
        unsigned int row_end = mat.row_indx[r + 1];

        for (unsigned int k = 0; k < K; ++k) {
            dmat_out[r * K + k] = 0;
        }

        for (unsigned int j = row_start; j < row_end; ++j) {
            unsigned int col_id = mat.col_id[j];
            double val = mat.values[j];

            for (unsigned int k = 0; k < K; ++k) {
                dmat_out[r * K + k] += val * dmat_in[col_id * K + k];
            }
        }

    }
}

//Emin Code start
__global__ void dev_csr_spmm(unsigned int * deviceCSRrow_indx , unsigned int * deviceCSRcol_id  , unsigned int * deviceCSRvalues,
   double * dmat_in_device, double* dmat_out_device , unsigned int K , unsigned int device_nrows ){


      int iy= blockIdx.y*blockDim.y + threadIdx.y ;
      int ix= blockIdx.x*blockDim.x+  threadIdx.x ;

      //int numberOfRowCSR = A.nrows;
      int numberOfRowCSR = device_nrows ;
      unsigned colId;
      //const int row = blockIdx.x * blockDim.x + threadIdx.x ;

      if ( iy < numberOfRowCSR && ix < K) {

        double sum=0.0;

        //unsigned int row_start = A.row_indx[iy] ;
        unsigned int row_start = deviceCSRrow_indx[iy];
        //unsigned int row_end = A.row_indx[iy + 1] ;
        unsigned int row_end = deviceCSRrow_indx[iy+1] ;


        for (unsigned i = row_start; i < row_end; i++) {
          /* code */
          //colId= A.col_id[i] ;
          colId = deviceCSRcol_id[i] ;
          //sum += A.values[i] * dmat_in_device[colId * K + ix] ;
          sum += deviceCSRvalues[i] ;
        }

        //dmat_out[ix][iy] = sum ;
        dmat_out_device[ix * K + iy] = sum ;
      }

}




int main(int argc, char *argv[]) {
    if(argc < 3) {
        std::cerr << "usage ./exec inputfile K  " << std::endl;
        exit(-1);
    }

    unsigned int K = std::atoi(argv[2]);
    CSR mat = read_matrix_market_to_CSR(argv[1]);
    //print_CSR(mat);
    std::cout << mat.nrows << ' ' << mat.ncols << ' ' << mat.nnz << ' ' << K << '\n';

    double *dmat_in = (double*)malloc(mat.ncols * K  * sizeof(double));
    double *dmat_out = (double*)malloc(mat.nrows * K * sizeof(double));
    double *dmat_out_GPU = (double*)malloc(mat.nrows * K * sizeof(double));

    init_dmat(dmat_in, mat.ncols, K,  1.0);
    //print_dmat(dmat_in, mat.ncols, K);

    host_csr_spmm(mat, dmat_in, dmat_out, K);


    //Prepeare for Kernel
    //CSR *temMat;
    //temMat->nrows = mat.nrows ;
    //temMat.->ncols = mat.ncols ;
    //temMat.->nnz = mat.nnz ;

    unsigned int* deviceCSRrow_indx;
    unsigned int* deviceCSRcol_id;
    double* deviceCSRvalues;

    unsigned int* device_nrows;
    unsigned int* device_ncols;
    unsigned int* nnz;

    cudaMalloc((void**) &deviceCSRrow_indx ,(mat.nrows +1) * sizeof(unsigned int)) ;
    cudaMalloc((void**) &deviceCSRcol_id , mat.ncols * sizeof(unsigned int)) ;
    cudaMalloc((void**) &deviceCSRvalues , mat.nnz * sizeof(double)) ;

    cudaMalloc((void**) &device_nrows, mat.nrows , sizeof(unsigned int));
    cudaMalloc((void**) &device_ncols, mat.ncols , sizeof(unsigned int));
    cudaMalloc((void**) &device_nnz, mat.nnz , sizeof(unsigned int));

    //cudaMalloc((void**) &(temMat->values) , mat.nnz * sizeof(double)) ;
    //cudaMalloc((void**) &(temMat->row_indx) , mat.nrows * sizeof(unsigned int)) ;
    //cudaMalloc((void**) &(temMat->col_id) , mat.ncols * sizeof(unsigned int)) ;

    //cudaMalloc((void**) &(temMat->nrows) , sizeof(unsigned int)) ;
    //cudaMalloc((void**) &(temMat->ncols) , sizeof(unsigned int)) ;
    //cudaMalloc((void**) &(temMat->nnz) , sizeof(unsigned int)) ;

    //Initialize device addresses since it can not be accessed directly
    //cudaMemcpy(temMat->values , mat.values , mat.nnz * sizeof(double) , cudaMemcpyHostToDevice) ;
    //cudaMemcpy(temMat->row_indx , mat.row_indx , mat.nrows * sizeof(unsigned int) , cudaMemcpyHostToDevice) ;
    //cudaMemcpy(temMat->col_id , mat.col_id , mat.ncols * sizeof(unsigned int) , cudaMemcpyHostToDevice) ;

    cudaMemcpy(deviceCSRrow_indx , mat.row_indx ,  mat.nrows * sizeof(unsigned int) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(deviceCSRcol_id, mat.col_id , mat.ncols * sizeof(unsigned int) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(deviceCSRvalues , mat.values , mat.nnz * sizeof(double) , cudaMemcpyHostToDevice) ;

    cudaMemcpy(device_nrows , mat.nrows , sizeof(unsigned int)) ;
    cudaMemcpy(device_ncols , mat.ncols , sizeof(unsigned int)) ;
    cudaMemcpy(device_nnz   , mat.nnz   , sizeof(unsigned int)) ;
    //cudaMemcpy(temMat->nrows , mat.nrows , 1*sizeof(unsigned int) , cudaMemcpyHostToDevice) ;
    //cudaMemcpy(temMat->ncols , mat.ncols , 1*sizeof(unsigned int) , cudaMemcpyHostToDevice) ;
    //cudaMemcpy(temMat->nnz , mat.nnz , 1*sizeof(unsigned int) , cudaMemcpyHostToDevice) ;

    //CSR A;
    //cudaMemcpyToSymbol( A , temMat , sizeof(CSR)) ;

    double *dmat_in_device ;
    cudaMalloc((void**) &dmat_in_device , mat.ncols * K * sizeof(double)) ;

    double *dmat_out_device ;
    cudaMalloc((void**) &dmat_out_device, mat.nrows * K * sizeof(double)) ;

    //copt to device
    cudaMemcpy( dmat_in_device , dmat_in , mat.ncols * K * sizeof(double) , cudaMemcpyHostToDevice ) ;
    cudaMemcpy( dmat_out_device, dmat_out, mat.nrows * K * sizeof(double) , cudaMemcpyHostToDevice ) ;


    //Initialize the Grid and Block Dimension

    dim3 dimGrid((K-1) / TILE_WIDTH + 1 , (mat.nrows -1)/ TILE_WIDTH +1 , 1  ) ;
    dim3 dimBlock(TILE_WIDTH , TILE_WIDTH , 1) ;

    dev_csr_spmm<<<dimGrid , dimBlock>>>(temMat , dmat_in_device , dmat_out_device , K) ;

    cudaMemcpy(dmat_out_GPU , dmat_out_device ,mat.nrows * K * sizeof(double) , cudaMemcpyDeviceToHost ) ;


    //td::cout << "replace one argument to the below function with the values from gpu " << std::endl;
    check_dmat(dmat_out, dmat_out_GPU, mat.nrows, K);

    //print_dmat(dmat_out, mat.nrows, K);


    free(mat.row_indx);
    free(mat.col_id);
    free(mat.values);

    cudaFree(deviceCSRrow_indx) ;
    cudaFree(deviceCSRcol_id) ;
    cudaFree(deviceCSRvalues) ;

    cudaFree(device_nrows) ;
    cudaFree(device_ncols) ;
    cudaFree(device_nnz ) ;
    return 0;
}
