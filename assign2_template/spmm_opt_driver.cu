/*
 * spmm_opt_driver.cu
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
#include <cuda.h>

#define TILE_WIDTH 32
#define MAX_BLOCK 50000
#define CHUNK_SIZE 1000

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
__global__ void dev_opt_spmm_2(unsigned int * deviceCSRrow_indx , unsigned int * deviceCSRcol_id  ,  double * deviceCSRvalues,
   double * dmat_in_device, double* dmat_out_device ,  int K , unsigned int device_nrows ){

     __shared__ double vals[TILE_WIDTH] ;

      //int row= blockIdx.y*blockDim.y + threadIdx.y ;
      //const int thread_id_x=blockIdx.x * blockDim.x + threadIdx.x;
      //const int thread_id_y=blockIdx.y * blockDim.y + threadIdx.y;

      const int thread_id_x=blockIdx.x  * blockDim.x + threadIdx.x;

      //const int col= blockIdx.x * blockDim.x + threadIdx.x ;
      const int warp_id = thread_id_x /32 ;

      //const int irow= warp_id / K ;
      //onst int icol= warp_id & (K-1) ;

      int irow=warp_id ;
      int lane = thread_id_x & (31) ;


      unsigned int numberOfRowCSR = device_nrows ;

      //const int row = blockIdx.x * blockDim.x + threadIdx.x ;
      //printf(" Rows = %d thread %d , block %d \n", numberOfRowCSR,  col , row);

      if ( irow < numberOfRowCSR ) {

          for(int icol =0 ; icol < K ; icol++)
          {
                //printf(" icol %d , irow %d \n",  icol , irow);

                int colId;
                double sum=0;

                // int row_start = A.row_indx[iy] ;
                 unsigned int row_start = deviceCSRrow_indx[irow];

                 unsigned int row_end = deviceCSRrow_indx[irow+1] ;
                 //printf(" row_end = %d thread %d , block %d \n", row_end,  col , row);

                 //dmat_out_device[row * K + col] =0;
                 __syncthreads();
                 vals[threadIdx.x] = 0 ;
                    __syncthreads();

                 for ( int element = row_start + lane ; element < row_end; element+=32) {
                      /* code */

                      //colId= A.col_id[i] ;
                      colId = deviceCSRcol_id[element] ;
                      //printf(" colId = %d thread %d , block %d \n", colId,  col , row);

                      double value = deviceCSRvalues[element] ;
                      double value2 = dmat_in_device[colId * K + icol] ;

                      //printf(" colId = %d thread %d , block %d \n", colId,  threadIdx.x , irow);

                      //vals[threadIdx.x] += value + value2 ;
                        sum=value * value2;
                        atomicAdd(&vals[threadIdx.x] ,value * value2 );

                      //printf(" sum =  %d ,thread %d , block %d", sum, col , row);
                 }
                //Parallel Reduction
                __syncthreads();
                if(lane < 16)
                     atomicAdd(&vals[threadIdx.x] , vals[threadIdx.x + 16]) ;
                 if(lane < 8 )
                     atomicAdd(&vals[threadIdx.x] , vals[threadIdx.x + 8]) ;
                 if(lane < 4 )
                     atomicAdd(&vals[threadIdx.x] , vals[threadIdx.x + 4]) ;
                 if(lane < 2 )
                     atomicAdd(&vals[threadIdx.x] , vals[threadIdx.x + 2]) ;
                 if(lane < 1 )
                     atomicAdd(&vals[threadIdx.x] , vals[threadIdx.x + 1]) ;


                //__syncthreads();
                //dmat_out[ix][iy] = sum ;
                //printf(" sum = %d thread %d , block %d \n", sum,  col , row);
                __syncthreads();
                if(lane == 0)
                  atomicAdd(&dmat_out_device[irow * K + icol] , vals[threadIdx.x]) ;
                //printf("dvice matrix %d\n", dmat_out_device[row * K + col] );
          }
      }

}

//Emin Code start
__global__ void dev_opt_spmm(unsigned int * deviceCSRrow_indx , unsigned int * deviceCSRcol_id  ,  double * deviceCSRvalues,
   double * dmat_in_device, double* dmat_out_device ,  int K , unsigned int device_nrows ){

     __shared__ double vals[TILE_WIDTH] ;

      //int row= blockIdx.y*blockDim.y + threadIdx.y ;
      const int thread_id_x=blockIdx.x * blockDim.x + threadIdx.x;

      const int warp_id = thread_id_x /32 ;

      const int irow= warp_id / K ;
      const int icol= warp_id & (K-1) ;



      int lane = thread_id_x & (31) ;


      unsigned int numberOfRowCSR = device_nrows ;

      if ( irow < numberOfRowCSR && icol < K) {


            int colId;

            // int row_start = A.row_indx[iy] ;
             unsigned int row_start = deviceCSRrow_indx[irow];
             //printf(" row_start = %d thread %d , block %d \n", row_start,  col , row);
            // int row_end = A.row_indx[iy + 1] ;
             unsigned int row_end = deviceCSRrow_indx[irow+1] ;

             vals[threadIdx.x] = 0 ;

             for ( unsigned int element = row_start + lane ; element < row_end; element+=32) {
                  /* code */

                  //colId= A.col_id[i] ;
                  colId = deviceCSRcol_id[element] ;
                  //printf(" colId = %d thread %d , block %d \n", colId,  col , row);

                  double value = deviceCSRvalues[element] ;
                  double value2 = dmat_in_device[colId * K + icol] ;


                  atomicAdd(&vals[threadIdx.x] ,value * value2 );
                  //printf(" sum =  %d ,thread %d , block %d", sum, col , row);
             }
            //Parallel Reduction

            if(lane < 16)
                  atomicAdd(&vals[threadIdx.x] , vals[threadIdx.x + 16]) ;
            if(lane < 8 )
                  atomicAdd(&vals[threadIdx.x] , vals[threadIdx.x + 8]) ;
            if(lane < 4 )
                  atomicAdd(&vals[threadIdx.x] , vals[threadIdx.x + 4]) ;
            if(lane < 2 )
                  atomicAdd(&vals[threadIdx.x] , vals[threadIdx.x + 2]) ;
            if(lane < 1 )
                  atomicAdd(&vals[threadIdx.x] , vals[threadIdx.x + 1]) ;


            if(lane == 0)
                  atomicAdd(&dmat_out_device[irow * K + icol] , vals[threadIdx.x]) ;
            //printf("dvice matrix %d\n", dmat_out_device[row * K + col] );
      }

}


int main(int argc, char *argv[]) {
    if(argc < 3) {
        std::cerr << "usage ./exec inputfile K  " << std::endl;
        exit(-1);
    }

    unsigned int K = std::atoi(argv[2]);
    CSR mat = read_matrix_market_to_CSR(argv[1]);
    std::cout << mat.nrows << ' ' << mat.ncols << ' ' << mat.nnz << ' ' << K << '\n';

    double timeKernelCPUstart;
    double timeKernelCPUfinish;


    double *dmat_in = (double*)malloc(mat.ncols * K  * sizeof(double));
    double *dmat_out = (double*)malloc(mat.nrows * K * sizeof(double));

    init_dmat(dmat_in, mat.ncols, K,  1.0);

    /// No need to optimize host;
    host_csr_spmm(mat, dmat_in, dmat_out, K);

    //Cuda Events
    // events for timing
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

    //double *dmat_in = (double*)malloc(mat.ncols * K  * sizeof(double));
    //double *dmat_out = (double*)malloc(mat.nrows * K * sizeof(double));
    double *dmat_out_GPU = (double*)malloc(mat.nrows * K * sizeof(double));


    unsigned int* deviceCSRrow_indx;
    unsigned int* deviceCSRcol_id;
    double* deviceCSRvalues;


    cudaMalloc((void**) &deviceCSRrow_indx ,(mat.nrows +1) * sizeof(unsigned int)) ;
    cudaMalloc((void**) &deviceCSRcol_id , mat.nnz * sizeof(unsigned int)) ;
    cudaMalloc((void**) &deviceCSRvalues , mat.nnz * sizeof(double)) ;

    double *dmat_in_device ;
    cudaMalloc((void**) &dmat_in_device , mat.ncols * K * sizeof(double)) ;

    double *dmat_out_device ;
    cudaMalloc((void**) &dmat_out_device, mat.nrows * K * sizeof(double)) ;

    cudaEventRecord(startEventMemKer, 0);

    //cudaMemcpy(deviceCSRrow_indx , mat.row_indx ,(mat.nrows+1) * sizeof(unsigned int) , cudaMemcpyHostToDevice) ;
    //cudaMemcpy(deviceCSRcol_id , mat.col_id , mat.nnz * sizeof(unsigned int) , cudaMemcpyHostToDevice ) ;
    //cudaMemcpy(deviceCSRvalues , mat.values , mat.nnz * sizeof(double) , cudaMemcpyHostToDevice)  ;

    cudaMemcpy(deviceCSRrow_indx , pinnedMat.row_indx ,(mat.nrows+1) * sizeof(unsigned int) , cudaMemcpyHostToDevice );


    cudaMemcpyAsync(deviceCSRcol_id , pinnedMat.col_id , mat.nnz * sizeof(unsigned int) , cudaMemcpyHostToDevice ,0);
    cudaMemcpyAsync(deviceCSRvalues , pinnedMat.values , mat.nnz * sizeof(double) , cudaMemcpyHostToDevice ,0)  ;

    //copy to device
    cudaMemcpyAsync( dmat_in_device , dmat_in , mat.ncols * K * sizeof(double) , cudaMemcpyHostToDevice ,0) ;
    //cudaMemcpy( dmat_out_device, dmat_out, mat.nrows * K * sizeof(double) , cudaMemcpyHostToDevice ) ;

    int count = (mat.nrows- 1) / CHUNK_SIZE + 1;
    cudaStream_t * stream = new cudaStream_t[count] ;

    float min=10000000000000;
    float max=0;
    float average=0;
    float time;

    timeKernelCPUstart=omp_get_wtime( );

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

          cudaEventRecord(startTime, 0);
          cudaEventSynchronize(startTime);

          cudaMemcpyAsync(deviceCSRrow_indx + start , pinnedMat.row_indx + start, (end - start +1 )* sizeof(unsigned int) , cudaMemcpyHostToDevice, stream[i]) ;

          dim3 dimGrid( ( end -start  ) *K +1 , 1 ,  1  ) ; //for dev_opt_spmm

          //dim3 dimGrid(  end -start   +1 , 1 ,  1  ) ;
          dim3 dimBlock(TILE_WIDTH, 1 , 1) ; //

          dev_opt_spmm<<<dimGrid ,  dimBlock , 0, stream[i] >>>(deviceCSRrow_indx + start, deviceCSRcol_id, deviceCSRvalues , dmat_in_device , (dmat_out_device + start * K ), K , end-start); //

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

    for (int i = 0; i < count; i++) {
      cudaStreamSynchronize(stream[i]) ;
      cudaStreamDestroy(stream[i]);
    }

    timeKernelCPUfinish=omp_get_wtime( );
    check_dmat(dmat_out, dmat_out_GPU, mat.nrows, K);

    printf("  Time for Mem Cpy and Kernel : %f\n",  timeKernelCPUfinish);

    float timeforKernel;
    cudaEventElapsedTime(&timeforKernel, startEvent, stopEvent) ;
    printf("  Time for Kernel : %f\n",  timeforKernel);



    //Lets compute GFLOP
    unsigned int twoKnnz= 2 * K * mat.nnz ;
    printf("  2 * K * nnz : %d\n",  twoKnnz);


    float GFLOP = (twoKnnz / (timeKernelCPUfinish-timeKernelCPUstart) )/1000000000 ;
    printf("  GFLOP : %f\n",  GFLOP);


    average=average/count;

    printf("Min Cuda Stream Event : %f\n",  min);
    printf("Max Cuda Stream Event : %f\n",  max);
    printf("Average Cuda Stream Event : %f\n",  average);



    free(mat.row_indx);
    free(mat.col_id);
    free(mat.values);

    cudaFree(deviceCSRrow_indx) ;
    cudaFree(deviceCSRcol_id) ;
    cudaFree(deviceCSRvalues) ;

    cudaFreeHost(pinnedMat.row_indx);
    cudaFreeHost(pinnedMat.col_id) ;
    cudaFreeHost(pinnedMat.values) ;




    return 0;
}
