/*
 * spmm_csc_driver.cu
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

void host_csc_spmm(CSC mat, double * dmat_in, double * dmat_out, unsigned int K) {
    for (unsigned int r = 0; r < mat.nrows; ++r) {
        for (unsigned int k = 0; k < K; ++k) {
            dmat_out[r * K + k] = 0;
        }
    }
    for (unsigned int c = 0; c < mat.ncols; ++c) {
        unsigned int col_start = mat.col_indx[c];
        unsigned int col_end = mat.col_indx[c + 1];

        for (unsigned int r = col_start; r < col_end; ++r) {
            unsigned int row_id = mat.row_id[r];
            double val = mat.values[r];

            for (unsigned int k = 0; k < K; ++k) {
                dmat_out[row_id * K + k] += val * dmat_in[c * K + k];
            }
        }

    }
}


//Emin Code start
__global__ void dev_csc_spmm(unsigned int * deviceCSCcol_indx , unsigned int * deviceCSCrow_id  ,  double * deviceCSCvalues,
   double * dmat_in_device, double* dmat_out_device ,  int K , unsigned int device_ncols , unsigned int device_nrows){



      const int row=blockIdx.y * blockDim.y + threadIdx.y;
      const int col= blockIdx.x * blockDim.x + threadIdx.x ;



      unsigned int numberOfColCSC = device_ncols ;


      if ( (col < numberOfColCSC) && (row < K) ) {

            //printf(" thread %d , block %d \n",  col , row);
            //__syncthreads();

            double sum=0;
            int rowId;

            // int row_start = A.row_indx[iy] ;
             unsigned int col_start = deviceCSCcol_indx[col];
             //printf(" row_start = %d thread %d , block %d \n", row_start,  col , row);
            // int row_end = A.row_indx[iy + 1] ;
             unsigned int col_end = deviceCSCcol_indx[col+1] ;
             //printf(" row_end = %d thread %d , block %d \n", row_end,  col , row);


            for ( int element = col_start; element < col_end; element++) {
                  /* code */

                  //colId= A.col_id[i] ;
                  rowId = deviceCSCrow_id[element] ;
                  //printf(" rolId = %d thread %d , block %d \n", rowId,  col , row);

                  double value = deviceCSCvalues[element] ;
                  double value2 = dmat_in_device[col * K + row] ;


                  //Lets try atomic operation
                  sum = value * value2;
                  atomicAdd(&dmat_out_device[rowId * K + row] ,sum );
                  //printf(" sum =  %d ,thread %d , block %d", sum, col , row);
                  //__syncthreads();
            }

      }

}


int main(int argc, char *argv[]) {
    if(argc < 3) {
        std::cerr << "usage ./exec inputfile K  " << std::endl;
        exit(-1);
    }


    unsigned int K = std::atoi(argv[2]);
    CSC mat = read_matrix_market_to_CSC(argv[1]);
    std::cout << mat.nrows << ' ' << mat.ncols << ' ' << mat.nnz << ' ' << K << '\n';

    //Cuda Events
    // events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent) ;
    cudaEventCreate(&stopEvent)  ;

    //int TILE_WIDTH = K+1;

    cudaEvent_t startEventMemKer , stopEventMemKer ;
    cudaEventCreate(&startEventMemKer);
    cudaEventCreate(&stopEventMemKer) ;



    double *dmat_in = (double*)malloc(mat.ncols * K  * sizeof(double));
    double *dmat_out = (double*)malloc(mat.nrows * K * sizeof(double));
    double *dmat_out_D = (double*)malloc(mat.nrows * K * sizeof(double));

    double *dmat_out_GPU = (double*)malloc(mat.nrows * K * sizeof(double));

    init_dmat(dmat_in, mat.ncols, K, 1.0);
    //print_dmat(dmat_in, mat.ncols, K);

    host_csc_spmm(mat, dmat_in, dmat_out, K);

    unsigned int* deviceCSCcol_indx;
    unsigned int* deviceCSCrow_id;
    double* deviceCSCvalues;


    cudaMalloc((void**) &deviceCSCcol_indx ,(mat.ncols +1) * sizeof(unsigned int)) ;
    cudaMalloc((void**) &deviceCSCrow_id , mat.nnz * sizeof(unsigned int)) ;
    cudaMalloc((void**) &deviceCSCvalues , mat.nnz * sizeof(double)) ;

    double *dmat_in_device ;
    cudaMalloc((void**) &dmat_in_device , mat.ncols * K * sizeof(double)) ;

    double *dmat_out_device ;
    cudaMalloc((void**) &dmat_out_device, mat.nrows * K * sizeof(double)) ;

    cudaEventRecord(startEventMemKer, 0);





    cudaMemcpy(deviceCSCcol_indx , mat.col_indx ,  (mat.ncols+1) * sizeof(unsigned int) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(deviceCSCrow_id, mat.row_id , mat.nnz * sizeof(unsigned int) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(deviceCSCvalues , mat.values , mat.nnz * sizeof(double) , cudaMemcpyHostToDevice) ;



    //copy to device
    cudaMemcpy( dmat_in_device , dmat_in , mat.ncols * K * sizeof(double) , cudaMemcpyHostToDevice ) ;



    //Initialize the Grid and Block Dimension

    //dim3 dimGrid((K-1) / TILE_WIDTH + 1 , (mat.ncols -1)/TILE_WIDTH +1 , 1  ) ;
    //
    //dim3 dimGrid( (K-1) / TILE_WIDTH +1  , (mat.ncols -1)/TILE_WIDTH+1 , 1  ) ;
    //dim3 dimGrid( (K-1) / TILE_WIDTH +1  , (mat.ncols -1)/TILE_WIDTH+1 , 1  ) ;
    dim3 dimGrid(  (mat.ncols -1)/TILE_WIDTH+1 ,  (K-1) / TILE_WIDTH +1 , 1  ) ;
    dim3 dimBlock(TILE_WIDTH , TILE_WIDTH , 1) ;

    cudaEventRecord(startEvent, 0);

    dev_csc_spmm<<<dimGrid , dimBlock>>>(deviceCSCcol_indx, deviceCSCrow_id, deviceCSCvalues , dmat_in_device , dmat_out_device , K , mat.ncols, mat.nrows) ;

    cudaEventRecord(stopEvent, 0) ;

    cudaEventSynchronize(startEvent);
    cudaEventSynchronize(stopEvent);

    float timeforKernel;
    cudaEventElapsedTime(&timeforKernel, startEvent, stopEvent) ;

    printf("  Time for Kernel : %f\n",  timeforKernel);


    cudaMemcpy(dmat_out_GPU , dmat_out_device ,mat.nrows * K * sizeof(double) , cudaMemcpyDeviceToHost ) ;

    cudaEventRecord(stopEventMemKer, 0) ;

    cudaEventSynchronize(startEventMemKer);
    cudaEventSynchronize(stopEventMemKer);

    float timeforMemKernel;
    cudaEventElapsedTime(&timeforMemKernel, startEventMemKer, stopEventMemKer) ;
    printf("  Time for Mem Cpy and Kernel : %f\n",  timeforMemKernel);

    //std::cout << "replace one argument to the below function with the values from gpu " << std::endl;

    //std::cout << "replace one argument to the below function with the values from gpu " << std::endl;
    //std::cout << "CPU\n";
    //print_dmat(dmat_out, mat.nrows , K);
    //std::cout << "GPU\n";
    //print_dmat(dmat_out_GPU,  mat.nrows , K);

    check_dmat(dmat_out, dmat_out_GPU, mat.nrows, K);

    //Lets compute GFLOP
    unsigned int twoKnnz= 2 * K * mat.nnz ;
    printf("  2 * K * nnz : %d\n",  twoKnnz);


    float GFLOP = (twoKnnz / timeforMemKernel )/1000000; ;
    printf("  GFLOP : %f\n",  GFLOP);


    //print_dmat(dmat_out, mat.nrows, K);


    free(mat.col_indx);
    free(mat.row_id);
    free(mat.values);

    cudaFree(deviceCSCcol_indx) ;
    cudaFree(deviceCSCrow_id) ;
    cudaFree(deviceCSCvalues) ;

    cudaEventDestroy(startEvent) ;
    cudaEventDestroy(stopEvent ) ;

    cudaEventDestroy(startEventMemKer);
    cudaEventDestroy(stopEventMemKer) ;

    return 0;
}
