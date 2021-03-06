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
#include <omp.h>
#include <cstdlib>
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>


#define TILE_WIDTH 32
#define CHUNK_SIZE 100


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
__global__ void dev_csr_spmm(unsigned int * deviceCSRrow_indx , unsigned int * deviceCSRcol_id  ,  double * deviceCSRvalues,
   double * dmat_in_device, double* dmat_out_device ,  int K , unsigned int device_nrows ){


      //int row= blockIdx.y*blockDim.y + threadIdx.y ;
      //const int row=blockIdx.y * blockDim.y + threadIdx.y ;
      //const int col= blockIdx.x * blockDim.x + threadIdx.x ;

      const int col=blockIdx.y * blockDim.y + threadIdx.y ;
      const int row= blockIdx.x * blockDim.x + threadIdx.x ;


      unsigned int numberOfRowCSR = device_nrows ;

      //const int row = blockIdx.x * blockDim.x + threadIdx.x ;
      //printf(" Rows = %d col %d , row %d \n", numberOfRowCSR,  col , row);

      if ( (row < numberOfRowCSR) && (col < K) ) {

            //printf(" thread %d , block %d \n",  col , row);

            double sum=0;
            int colId;

            // int row_start = A.row_indx[iy] ;
             unsigned int row_start = deviceCSRrow_indx[row];
             //printf(" row_start = %d thread %d , block %d \n", row_start,  col , row);
            // int row_end = A.row_indx[iy + 1] ;
             unsigned int row_end = deviceCSRrow_indx[row+1] ;
             //printf(" row_end = %d thread %d , block %d \n", row_end,  col , row);

             dmat_out_device[row * K + col] =0;

            for ( int element = row_start; element < row_end; element++) {
                  /* code */

                  //colId= A.col_id[i] ;
                  colId = deviceCSRcol_id[element] ;
                  //printf(" colId = %d thread %d , block %d \n", colId,  col , row);

                  double value = deviceCSRvalues[element] ;
                  double value2 = dmat_in_device[colId * K + col] ;

                  //printf(" value %d  thread %d , block %d \n", value,  col , row);

                  sum = sum +  value * value2 ;

                  //printf(" sum =  %d ,thread %d , block %d", sum, col , row);
            }
            //__synctreads();
            //dmat_out[ix][iy] = sum ;
            //printf(" sum = %d thread %d , block %d \n", sum,  col , row);
            dmat_out_device[row * K + col] = sum ;
            //printf("dvice matrix %d\n", dmat_out_device[row * K + col] );
      }

}




int main(int argc, char *argv[]) {
    if(argc < 3) {
        std::cerr << "usage ./exec inputfile K  " << std::endl;
        exit(-1);
    }

    int K = std::atoi(argv[2]);
    CSR mat = read_matrix_market_to_CSR(argv[1]);

    //Cuda Events
    // events for timing
    double timeKernelCPUstart;
    double timeKernelCPUfinish;

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent) ;
    cudaEventCreate(&stopEvent)  ;


    cudaEvent_t startEventMemKer , stopEventMemKer ;
    cudaEventCreate(&startEventMemKer);
    cudaEventCreate(&stopEventMemKer) ;


    //Lets implement pinned memory
    CSR pinnedMat;
    cudaHostAlloc(&pinnedMat.row_indx , (mat.nrows +1)* sizeof(unsigned int), cudaHostAllocMapped ) ;
    cudaHostAlloc(&pinnedMat.col_id , mat.nnz * sizeof(unsigned int) , cudaHostAllocMapped) ;
    cudaHostAlloc(&pinnedMat.values , mat.nnz * sizeof(double), cudaHostAllocMapped) ;

    memcpy(pinnedMat.row_indx , mat.row_indx ,(mat.nrows +1)* sizeof(unsigned int)) ;
    memcpy(pinnedMat.col_id , mat.col_id ,mat.nnz * sizeof(unsigned int) ) ;
    memcpy(pinnedMat.values , mat.values ,mat.nnz * sizeof(double)) ;

    pinnedMat.nrows=mat.nrows ;
    pinnedMat.ncols=mat.ncols ;
    pinnedMat.nnz = mat.nnz ;

    std::cout << mat.nrows << ' ' << mat.ncols << ' ' << mat.nnz << ' ' << K << '\n';

    double *dmat_in = (double*)malloc(mat.ncols * K  * sizeof(double));

    //double *dmat_inPin;

    double *dmat_out = (double*)malloc(mat.nrows * K * sizeof(double));
    double *dmat_out_GPU = (double*)malloc(mat.nrows * K * sizeof(double));

    init_dmat(dmat_in, mat.ncols, K,  1.0);


    host_csr_spmm(mat, dmat_in, dmat_out, K);

    unsigned int* deviceCSRrow_indx;
    unsigned int* deviceCSRcol_id;
    double* deviceCSRvalues;

    //printf("malloc for CPU and GPU\n");

    cudaMalloc((void**) &deviceCSRrow_indx ,(mat.nrows +1) * sizeof(unsigned int)) ;
    //cudaMemset(deviceCSRrow_indx , 0 ,(mat.nrows +1) * sizeof(unsigned int) ) ;

    cudaMalloc((void**) &deviceCSRcol_id , mat.nnz * sizeof(unsigned int)) ;
    cudaMalloc((void**) &deviceCSRvalues , mat.nnz * sizeof(double)) ;

    double *dmat_in_device ;
    cudaMalloc((void**) &dmat_in_device , mat.ncols * K * sizeof(double)) ;



    double *dmat_out_device ;
    cudaMalloc((void**) &dmat_out_device, mat.nrows * K * sizeof(double)) ;

    //cudaMemset(dmat_out_device , 0 , mat.nrows * K * sizeof(double)) ;

    int count = (mat.nrows- 1) / CHUNK_SIZE + 1;

    cudaStream_t * stream = new cudaStream_t[count] ;

    //printf("transfer from CPU to GPU\n");


    cudaMemcpyAsync(deviceCSRcol_id , pinnedMat.col_id , mat.nnz * sizeof(unsigned int) , cudaMemcpyHostToDevice , 0);
    cudaMemcpyAsync(deviceCSRvalues , pinnedMat.values , mat.nnz * sizeof(double) , cudaMemcpyHostToDevice ,0)  ;
    cudaMemcpyAsync( dmat_in_device  , dmat_in  , mat.ncols * K * sizeof(double) , cudaMemcpyHostToDevice ,0 ) ;

    //cudaStreamCreate(stream0) ;


    timeKernelCPUstart=omp_get_wtime( );
    std::clock_t c_start = std::clock();

    float min=10000000000000;
    float max=0;
    float average=0;
    float time;

    for (int i = 0; i < count; i++) {
      /* code */

        cudaStreamCreate(&stream[i]) ;

        cudaEvent_t startTime, stopTime ;
        cudaEventCreate(&startTime) ;
        cudaEventCreate(&stopTime ) ;

        const int start = i * CHUNK_SIZE ;
        //const int end  = min(mat.nrows , (i +1) * CHUNK_SIZE) ;

        int end;

        if(mat.nrows < (i +1) * CHUNK_SIZE)
            end = mat.nrows;
        else
            end = (i +1) * CHUNK_SIZE ;
        //printf("end -start = %i\n ", dif);


        cudaEventRecord(startTime, 0);
        cudaEventSynchronize(startTime);
        //printf("stream number  = %d\n", i);
        cudaMemcpyAsync(deviceCSRrow_indx + start , pinnedMat.row_indx + start, (end - start +1 )* sizeof(unsigned int) , cudaMemcpyHostToDevice, stream[i]) ;

        dim3 dimGrid( ( end -start -1 -1)/TILE_WIDTH +1 , (K-1) / TILE_WIDTH + 1 ,  1  ) ;
        dim3 dimBlock(TILE_WIDTH , TILE_WIDTH , 1) ;

        dev_csr_spmm<<<dimGrid , dimBlock ,0 , stream[i] >>> ((deviceCSRrow_indx + start), deviceCSRcol_id , deviceCSRvalues  , dmat_in_device   , (dmat_out_device + start * K) , K, end -start) ;

        //cudaMemcpy(dmat_out_GPU , dmat_out_device ,mat.nrows * K * sizeof(double) , cudaMemcpyDeviceToHost ) ;
        cudaMemcpyAsync( (dmat_out_GPU + start*K ), (dmat_out_device +start*K ), (end -start  ) * K * sizeof(double) , cudaMemcpyDeviceToHost, stream[i] ) ;

        cudaEventRecord(stopTime , 0) ;
        cudaEventSynchronize(stopTime);
        cudaEventElapsedTime(&time, startTime, stopTime) ;

        if(time > max)
            max=time;

        if(time < min)
            min=time;

        average = average + time;

    }

    for (int i = 0; i < count; i++)
    {

      cudaStreamSynchronize(stream[i]) ;
      cudaStreamDestroy(stream[i]);
    }
    timeKernelCPUfinish=omp_get_wtime( );

    //std::clock_t c_end = std::clock();
    //std::cout << "replace one argument to the below function with the values from gpu " << std::endl;
    //std::cout << "CPU\n";
    //print_dmat(dmat_out, mat.nrows , K);
    //std::cout << "GPU\n";
    //print_dmat(dmat_out_GPU,  mat.nrows , K);
    check_dmat(dmat_out, dmat_out_GPU, mat.nrows, K);

    //Lets compute GFLOP
    unsigned int twoKnnz= 2 * K * mat.nnz ;
    printf("  2 * K * nnz : %d\n",  twoKnnz);


    float GFLOP = (twoKnnz / (timeKernelCPUfinish-timeKernelCPUstart) )/1000000000 ;
    printf("  GFLOP : %f\n",  GFLOP);

    average=average/count;

    printf("Min Cuda Stream Event : %f\n",  min);
    printf("Max Cuda Stream Event : %f\n",  max);
    printf("Average Cuda Stream Event : %f\n",  average);

    //double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    //std::cout << "CPU time used: " << time_elapsed_ms<< " ms\n";




    free(mat.row_indx);
    free(mat.col_id);
    free(mat.values);

    cudaFree(deviceCSRrow_indx) ;
    cudaFree(deviceCSRcol_id) ;
    cudaFree(deviceCSRvalues) ;

    cudaFreeHost(pinnedMat.row_indx);
    cudaFreeHost(pinnedMat.col_id) ;
    cudaFreeHost(pinnedMat.values) ;

    delete [] stream;
    //delete stream0 ;

    //cudaFree(device_nrows) ;
    //cudaFree(device_ncols) ;
    //cudaFree(device_nnz ) ;
    return 0;
}
