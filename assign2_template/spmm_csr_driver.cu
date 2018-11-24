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

void check_dmat(double* a, double *b,  int n,  int K, bool quit_on_err = true ) {
    for ( int i = 0; i < n; ++i) {
        for ( int k = 0; k < K; ++k) {
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

static  int g_seed = 0X4B1D;
inline int fastrand() {
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x7FFF;
}

void init_dmat(double *a,  int n,  int K, double offset) {
    for ( int i = 0; i < n; ++i) {
        for ( int k = 0; k < K; ++k) {
            a[i * K + k]  = i * K + k + offset;
            //a[i * K + j]  = fastrand() + offset;
        }
    }
}

void print_dmat(double *a,  int n,  int K) {
    for ( int i = 0; i < n; ++i) {
        for ( int j = 0; j < K; ++j) {
            std::cout << a[i * K + j]   << ' ';
        }
        std::cout << '\n';
    }
}

void print_CSR(CSR &mat) {
    for ( int r = 0; r < mat.nrows; ++r) {
         int row_start = mat.row_indx[r];
         int row_end = mat.row_indx[r + 1];
        for ( int j = row_start; j < row_end; ++j) {
             int col_id = mat.col_id[j];
            double val = mat.values[j];

	    std::cout << r << ' ' << col_id << ' ' <<  val << '\n';
        }
    }
}

void host_csr_spmm(CSR &mat, double * dmat_in, double * dmat_out,  int K) {
    for ( int r = 0; r < mat.nrows; ++r) {
         int row_start = mat.row_indx[r];
         int row_end = mat.row_indx[r + 1];

        for ( int k = 0; k < K; ++k) {
            dmat_out[r * K + k] = 0;
        }

        for ( int j = row_start; j < row_end; ++j) {
             int col_id = mat.col_id[j];
            double val = mat.values[j];

            for ( int k = 0; k < K; ++k) {
                dmat_out[r * K + k] += val * dmat_in[col_id * K + k];
            }
        }

    }
}

//Emin Code start
__global__ void dev_csr_spmm( int * deviceCSRrow_indx ,  int * deviceCSRcol_id  ,  int * deviceCSRvalues,
   double * dmat_in_device, double* dmat_out_device ,  int K ,  int device_nrows ){


      int iy= blockIdx.y*blockDim.y + threadIdx.y ;
      int ix= blockIdx.x*blockDim.x+  threadIdx.x ;

      //int numberOfRowCSR = A.nrows;
      int numberOfRowCSR = device_nrows ;
      int colId;
      //const int row = blockIdx.x * blockDim.x + threadIdx.x ;

      if ( iy < numberOfRowCSR && ix < K) {

        double sum=0.0;

        // int row_start = A.row_indx[iy] ;
         int row_start = deviceCSRrow_indx[iy];
        // int row_end = A.row_indx[iy + 1] ;
         int row_end = deviceCSRrow_indx[iy+1] ;


        for (int i = row_start; i < row_end; i++) {
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

     int K = std::atoi(argv[2]);
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

     int* deviceCSRrow_indx;
     int* deviceCSRcol_id;
    double* deviceCSRvalues;

     int* device_nrows;
     int* device_ncols;
     int* device_nnz;

     int a=1;

    cudaMalloc((void**) &deviceCSRrow_indx ,(mat.nrows +1) * sizeof(int)) ;
    cudaMalloc((void**) &deviceCSRcol_id , mat.ncols * sizeof(int)) ;
    cudaMalloc((void**) &deviceCSRvalues , mat.nnz * sizeof(double)) ;

    cudaMalloc((void**) &device_nrows, mat.nrows , a*sizeof(int));
    cudaMalloc((void**) &device_ncols, mat.ncols , a*sizeof(int));
    cudaMalloc((void**) &device_nnz, mat.nnz , a*sizeof(int));

    //cudaMalloc((void**) &(temMat->values) , mat.nnz * sizeof(double)) ;
    //cudaMalloc((void**) &(temMat->row_indx) , mat.nrows * sizeof( int)) ;
    //cudaMalloc((void**) &(temMat->col_id) , mat.ncols * sizeof( int)) ;

    //cudaMalloc((void**) &(temMat->nrows) , sizeof( int)) ;
    //cudaMalloc((void**) &(temMat->ncols) , sizeof( int)) ;
    //cudaMalloc((void**) &(temMat->nnz) , sizeof( int)) ;

    //Initialize device addresses since it can not be accessed directly
    //cudaMemcpy(temMat->values , mat.values , mat.nnz * sizeof(double) , cudaMemcpyHostToDevice) ;
    //cudaMemcpy(temMat->row_indx , mat.row_indx , mat.nrows * sizeof( int) , cudaMemcpyHostToDevice) ;
    //cudaMemcpy(temMat->col_id , mat.col_id , mat.ncols * sizeof( int) , cudaMemcpyHostToDevice) ;

    cudaMemcpy(deviceCSRrow_indx , mat.row_indx ,  mat.nrows * sizeof( int) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(deviceCSRcol_id, mat.col_id , mat.ncols * sizeof( int) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(deviceCSRvalues , mat.values , mat.nnz * sizeof(double) , cudaMemcpyHostToDevice) ;

    cudaMemcpy(device_nrows , mat.nrows , sizeof( int) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(device_ncols , mat.ncols , sizeof( int) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(device_nnz   , mat.nnz   , sizeof( int) , cudaMemcpyHostToDevice) ;
    //cudaMemcpy(temMat->nrows , mat.nrows , 1*sizeof( int) , cudaMemcpyHostToDevice) ;
    //cudaMemcpy(temMat->ncols , mat.ncols , 1*sizeof( int) , cudaMemcpyHostToDevice) ;
    //cudaMemcpy(temMat->nnz , mat.nnz , 1*sizeof( int) , cudaMemcpyHostToDevice) ;

    //CSR A;
    //cudaMemcpyToSymbol( A , temMat , sizeof(CSR)) ;

    double *dmat_in_device ;
    cudaMalloc((void**) &dmat_in_device , mat.ncols * K * sizeof(double)) ;

    double *dmat_out_device ;
    cudaMalloc((void**) &dmat_out_device, mat.nrows * K * sizeof(double)) ;

    //copy to device
    cudaMemcpy( dmat_in_device , dmat_in , mat.ncols * K * sizeof(double) , cudaMemcpyHostToDevice ) ;
    cudaMemcpy( dmat_out_device, dmat_out, mat.nrows * K * sizeof(double) , cudaMemcpyHostToDevice ) ;


    //Initialize the Grid and Block Dimension

    dim3 dimGrid((K-1) / TILE_WIDTH + 1 , (mat.nrows -1)/ TILE_WIDTH +1 , 1  ) ;
    dim3 dimBlock(TILE_WIDTH , TILE_WIDTH , 1) ;

    dev_csr_spmm<<<dimGrid , dimBlock>>>(deviceCSRrow_indx, deviceCSRcol_id, deviceCSRvalues , dmat_in_device , dmat_out_device , K , device_nrows) ;

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
