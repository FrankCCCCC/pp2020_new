#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <omp.h>

#define SIZEOFINT sizeof(int)
#define BLOCK_DIM 24
#define TH_DIM 24

const int INF = ((1 << 30) - 1);
int n, m, padding_n;
int *Dist, *Dist_s;
// int *Dist_cuda;
int up_part_b_size = 0, bottom_part_b_size = 0;
int *Dist_cuda0, *Dist_cuda1;

void show_mat(int *start_p, int vertex_num){
    for(int i = 0; i < vertex_num; i++){
        for(int j = 0; j < vertex_num; j++){
            if(start_p[i * vertex_num + j] == INF){
                printf("INF\t  ");
            }else{
                printf("%d\t  ", start_p[i * vertex_num + j]);
            }   
        }
        printf("\n");
    }
}

void show_mat_cuda(int *start_p, int vertex_num, int device_id){
    int *temp = (int*)malloc(SIZEOFINT * vertex_num * vertex_num);
    cudaSetDevice(device_id);
    cudaMemcpy(temp, start_p, (SIZEOFINT * vertex_num * vertex_num), cudaMemcpyDeviceToHost);

    for(int i = 0; i < vertex_num; i++){
        for(int j = 0; j < vertex_num; j++){
            if(temp[i * vertex_num + j] == INF){
                printf("INF\t  ");
            }else{
                printf("%d\t  ", temp[i * vertex_num + j]);
            }   
        }
        printf("\n");
    }
}

void malloc_Dist(){
    cudaHostAlloc(&Dist, SIZEOFINT * padding_n * padding_n, cudaHostAllocPortable);
    // Dist = (int*)malloc(SIZEOFINT * padding_n * padding_n);
    Dist_s = (int*)malloc(SIZEOFINT * n * n);
}
int getDist(int i, int j){return Dist[i * padding_n + j];}
int *getDistAddr(int i, int j){return &(Dist[i * padding_n + j]);}
void setDist(int i, int j, int val){Dist[i * padding_n + j] = val;}

void setup_DistCuda(){
    // cudaMalloc((void **)&Dist_cuda, SIZEOFINT * padding_n * padding_n);
    // cudaMemcpy(Dist_cuda, Dist, (padding_n * padding_n * SIZEOFINT), cudaMemcpyHostToDevice);

    // int *Dist_cudas[2];
    // Dist_cudas[0] = Dist_cuda0;
    // Dist_cudas[1] = Dist_cuda1;
    // #pragma omp parallel
    // for(int i=0; i<2; i++){
    //     cudaSetDevice(i);
    //     cudaDeviceEnablePeerAccess(i, 0);
    //     cudaMalloc((void **)&(Dist_cudas[i]), SIZEOFINT * padding_n * padding_n);
    //     cudaMemcpy((Dist_cudas[i]), Dist, (padding_n * padding_n * SIZEOFINT), cudaMemcpyHostToDevice);
    // }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(0, 0);
    cudaMalloc((void **)&Dist_cuda0, SIZEOFINT * padding_n * padding_n);
    cudaMemcpyAsync(Dist_cuda0, Dist, (padding_n * padding_n * SIZEOFINT), cudaMemcpyHostToDevice, stream);

    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaMalloc((void **)&Dist_cuda1, SIZEOFINT * padding_n * padding_n);
    cudaMemcpy(Dist_cuda1, Dist, (padding_n * padding_n * SIZEOFINT), cudaMemcpyHostToDevice);

    cudaStreamDestroy(stream);
}
void back_DistCuda(){
    // cudaMemcpy(Dist, Dist_cuda, (padding_n * padding_n * SIZEOFINT), cudaMemcpyDeviceToHost);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaSetDevice(0);
    cudaMemcpyAsync(Dist, Dist_cuda0, (BLOCK_DIM * up_part_b_size * padding_n * SIZEOFINT), cudaMemcpyDeviceToHost, stream);

    cudaSetDevice(1);
    cudaMemcpy(&(Dist[BLOCK_DIM * up_part_b_size * padding_n]), &(Dist_cuda1[BLOCK_DIM * up_part_b_size * padding_n]), (BLOCK_DIM * bottom_part_b_size * padding_n * SIZEOFINT), cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream);
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    padding_n = ((n + BLOCK_DIM - 1) / BLOCK_DIM) * BLOCK_DIM;
    malloc_Dist();

    for (int i = 0; i < padding_n; i++) {
        for (int j = 0; j < padding_n; j++) {
            if (i == j) {
                setDist(i, j, 0);
                // Dist[i][j] = 0;
            } else {
                setDist(i, j, INF);
                // Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; i++) {
        fread(pair, sizeof(int), 3, file);
        setDist(pair[0], pair[1], pair[2]);
        // Dist[pair[0]][pair[1]] = pair[2];
    }
    // cudaMemcpy(Dist_cuda, Dist, (n * n * SIZEOFINT), cudaMemcpyHostToDevice);
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // if (Dist[i][j] >= INF) Dist[i][j] = INF;
            if (getDist(i, j) >= INF) setDist(i, j, INF);
            Dist_s[i * n + j] = getDist(i, j);
        }
        // fwrite(Dist[i], sizeof(int), n, outfile);
        // fwrite(getDistAddr(i, 0), SIZEOFINT, n, outfile);
    }
    fwrite(Dist_s, sizeof(int), n * n, outfile);
    fclose(outfile);
}

__forceinline__
__device__ void block_calc(int* C, int* A, int* B, int bj, int bi) {
    for (int k = 0; k < BLOCK_DIM; k++) {
        int sum0 = A[bi*BLOCK_DIM + k] + B[k*BLOCK_DIM + bj];
        // int sum1 = A[(bi + TH_DIM)*BLOCK_DIM + k] + B[k*BLOCK_DIM + bj];
        // int sum2 = A[bi*BLOCK_DIM + k] + B[k*BLOCK_DIM + (bj + TH_DIM)];
        // int sum3 = A[(bi + TH_DIM)*BLOCK_DIM + k] + B[k*BLOCK_DIM + (bj + TH_DIM)];

        C[bi*BLOCK_DIM + bj] = min(C[bi*BLOCK_DIM + bj], sum0);
        // C[(bi + TH_DIM)*BLOCK_DIM + bj] = min(C[(bi + TH_DIM)*BLOCK_DIM + bj], sum1);
        // C[bi*BLOCK_DIM + (bj + TH_DIM)] = min(C[bi*BLOCK_DIM + (bj + TH_DIM)], sum2);
        // C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)] = min(C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)], sum3);
        __syncthreads();
    }
}

__forceinline__
__device__ void block_calc_rev_async(int* C, int* A, int* B, int bj, int bi) {
    #pragma unroll 10
    for (int k = 0; k < BLOCK_DIM; k++) {
        int sum0 = A[k*BLOCK_DIM + bi] + B[k*BLOCK_DIM + bj];
        // int sum1 = A[k*BLOCK_DIM + (bi + TH_DIM)] + B[k*BLOCK_DIM + bj];
        // int sum2 = A[k*BLOCK_DIM + bi] + B[k*BLOCK_DIM + (bj + TH_DIM)];
        // int sum3 = A[k*BLOCK_DIM + (bi + TH_DIM)] + B[k*BLOCK_DIM + (bj + TH_DIM)];

        C[bi*BLOCK_DIM + bj] = min(C[bi*BLOCK_DIM + bj], sum0);
        // C[(bi + TH_DIM)*BLOCK_DIM + bj] = min(C[(bi + TH_DIM)*BLOCK_DIM + bj], sum1);
        // C[bi*BLOCK_DIM + (bj + TH_DIM)] = min(C[bi*BLOCK_DIM + (bj + TH_DIM)], sum2);
        // C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)] = min(C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)], sum3);
    }
}

__global__ void floyd_warshall_block_kernel_phase1_mw(int n, int k, int* graph, int *dst0, int *dst1) {
    const unsigned int bi = threadIdx.y;
    const unsigned int bj = threadIdx.x;

    __shared__ int C[BLOCK_DIM * BLOCK_DIM];

    // Transfer to temp shared arrays
    C[bi*BLOCK_DIM + bj] = graph[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + bj)];
    // C[(bi + TH_DIM)*BLOCK_DIM + bj] = graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + bj)];
    // C[bi*BLOCK_DIM + (bj + TH_DIM)] = graph[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + (bj + TH_DIM))];
    // C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)] = graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + (bj + TH_DIM))];

    __syncthreads();
    
    block_calc(C, C, C, bi, bj);

    // Transfer back to graph
    dst0[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + bj)] = C[bi*BLOCK_DIM + bj];
    // dst0[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + bj)] = C[(bi + TH_DIM)*BLOCK_DIM + bj];
    // dst0[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + (bj + TH_DIM))] = C[bi*BLOCK_DIM + (bj + TH_DIM)];
    // dst0[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + (bj + TH_DIM))] = C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)];

    dst1[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + bj)] = C[bi*BLOCK_DIM + bj];
    // dst1[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + bj)] = C[(bi + TH_DIM)*BLOCK_DIM + bj];
    // dst1[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + (bj + TH_DIM))] = C[bi*BLOCK_DIM + (bj + TH_DIM)];
    // dst1[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + (bj + TH_DIM))] = C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)];
}

__global__ void floyd_warshall_block_kernel_phase1(int n, int k, int* graph) {
    const unsigned int bi = threadIdx.y;
    const unsigned int bj = threadIdx.x;

    __shared__ int C[BLOCK_DIM * BLOCK_DIM];

    // Transfer to temp shared arrays
    C[bi*BLOCK_DIM + bj] = graph[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + bj)];
    // C[(bi + TH_DIM)*BLOCK_DIM + bj] = graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + bj)];
    // C[bi*BLOCK_DIM + (bj + TH_DIM)] = graph[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + (bj + TH_DIM))];
    // C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)] = graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + (bj + TH_DIM))];

    __syncthreads();
    
    block_calc(C, C, C, bi, bj);

    // Transfer back to graph
    graph[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + bj)] = C[bi*BLOCK_DIM + bj];
    // graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + bj)] = C[(bi + TH_DIM)*BLOCK_DIM + bj];
    // graph[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + (bj + TH_DIM))] = C[bi*BLOCK_DIM + (bj + TH_DIM)];
    // graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + (bj + TH_DIM))] = C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)];
}


__global__ void floyd_warshall_block_kernel_phase2(int n, int k, int* graph) {
    // BlockDim is one dimensional (Straight along diagonal)
    // Blocks themselves are two dimensional
    // Phase 2 1/2
    const unsigned int i = blockIdx.x;
    const unsigned int bi = threadIdx.y;
    const unsigned int bj = threadIdx.x;

    __shared__ int A[BLOCK_DIM * BLOCK_DIM];
    __shared__ int B[BLOCK_DIM * BLOCK_DIM];
    __shared__ int C[BLOCK_DIM * BLOCK_DIM];

    C[bi*BLOCK_DIM + bj] = graph[(i*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + bj)];
    // C[(bi + TH_DIM)*BLOCK_DIM + bj] = graph[(i*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + bj)];
    // C[bi*BLOCK_DIM + (bj + TH_DIM)] = graph[(i*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + (bj + TH_DIM))];
    // C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)] = graph[(i*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + (bj + TH_DIM))];

    B[bi*BLOCK_DIM + bj] = graph[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + bj)];
    // B[(bi + TH_DIM)*BLOCK_DIM + bj] = graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + bj)];
    // B[bi*BLOCK_DIM + (bj + TH_DIM)] = graph[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + (bj + TH_DIM))];
    // B[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)] = graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + (bj + TH_DIM))];

    __syncthreads();

    block_calc(C, C, B, bi, bj);

    graph[(i*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + bj)] = C[bi*BLOCK_DIM + bj];
    // graph[(i*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + bj)] = C[(bi + TH_DIM)*BLOCK_DIM + bj];
    // graph[(i*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + (bj + TH_DIM))] = C[bi*BLOCK_DIM + (bj + TH_DIM)];
    // graph[(i*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + (bj + TH_DIM))] = C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)];

    // Phase 2 2/2

    C[bi*BLOCK_DIM + bj] = graph[(k*BLOCK_DIM + bi)*n + (i*BLOCK_DIM + bj)];
    // C[(bi + TH_DIM)*BLOCK_DIM + bj] = graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (i*BLOCK_DIM + bj)];
    // C[bi*BLOCK_DIM + (bj + TH_DIM)] = graph[(k*BLOCK_DIM + bi)*n + (i*BLOCK_DIM + (bj + TH_DIM))];
    // C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)] = graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (i*BLOCK_DIM + (bj + TH_DIM))];

    __syncthreads();

    block_calc(C, B, C, bi, bj);

    // Block C is the only one that could be changed
    graph[(k*BLOCK_DIM + bi)*n + (i*BLOCK_DIM + bj)] = C[bi*BLOCK_DIM + bj];
    // graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (i*BLOCK_DIM + bj)] = C[(bi + TH_DIM)*BLOCK_DIM + bj];
    // graph[(k*BLOCK_DIM + bi)*n + (i*BLOCK_DIM + (bj + TH_DIM))] = C[bi*BLOCK_DIM + (bj + TH_DIM)];
    // graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (i*BLOCK_DIM + (bj + TH_DIM))] = C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)];
}


__global__ void floyd_warshall_block_kernel_phase3(int n, int k, int* graph, int start_x, int start_y) {
    // BlockDim is one dimensional (Straight along diagonal)
    // Blocks themselves are two dimensional
    const unsigned int i = start_y + blockIdx.y;
    const unsigned int j = start_x + blockIdx.x;
    const unsigned int bi = threadIdx.y;
    const unsigned int bj = threadIdx.x;

    __shared__ int A[BLOCK_DIM * BLOCK_DIM];
    __shared__ int B[BLOCK_DIM * BLOCK_DIM];
    __shared__ int C[BLOCK_DIM * BLOCK_DIM];

    C[bi*BLOCK_DIM + bj] = graph[(i*BLOCK_DIM + bi)*n + (j*BLOCK_DIM + bj)];
    // C[(bi + TH_DIM)*BLOCK_DIM + bj] = graph[(i*BLOCK_DIM + (bi + TH_DIM))*n + (j*BLOCK_DIM + bj)];
    // C[bi*BLOCK_DIM + (bj + TH_DIM)] = graph[(i*BLOCK_DIM + bi)*n + (j*BLOCK_DIM + (bj + TH_DIM))];
    // C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)] = graph[(i*BLOCK_DIM + (bi + TH_DIM))*n + (j*BLOCK_DIM + (bj + TH_DIM))];

    A[bj*BLOCK_DIM + bi] = graph[(i*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + bj)];
    // A[bj*BLOCK_DIM + (bi + TH_DIM)] = graph[(i*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + bj)];
    // A[(bj + TH_DIM)*BLOCK_DIM + bi] = graph[(i*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + (bj + TH_DIM))];
    // A[(bj + TH_DIM)*BLOCK_DIM + (bi + TH_DIM)] = graph[(i*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + (bj + TH_DIM))];

    B[bi*BLOCK_DIM + bj] = graph[(k*BLOCK_DIM + bi)*n + (j*BLOCK_DIM + bj)];
    // B[(bi + TH_DIM)*BLOCK_DIM + bj] = graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (j*BLOCK_DIM + bj)];
    // B[bi*BLOCK_DIM + (bj + TH_DIM)] = graph[(k*BLOCK_DIM + bi)*n + (j*BLOCK_DIM + (bj + TH_DIM))];
    // B[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)] = graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (j*BLOCK_DIM + (bj + TH_DIM))];

    __syncthreads();

    block_calc_rev_async(C, A, B, bi, bj);

    __syncthreads();

    graph[(i*BLOCK_DIM + bi)*n + (j*BLOCK_DIM + bj)] = C[bi*BLOCK_DIM + bj];
    // graph[(i*BLOCK_DIM + (bi + TH_DIM))*n + (j*BLOCK_DIM + bj)] = C[(bi + TH_DIM)*BLOCK_DIM + bj];
    // graph[(i*BLOCK_DIM + bi)*n + (j*BLOCK_DIM + (bj + TH_DIM))] = C[bi*BLOCK_DIM + (bj + TH_DIM)];
    // graph[(i*BLOCK_DIM + (bi + TH_DIM))*n + (j*BLOCK_DIM + (bj + TH_DIM))] = C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)];
}

void block_FW_cuda() {
    const int blocks = padding_n / BLOCK_DIM;
    const int row_b_size = BLOCK_DIM * padding_n;
    up_part_b_size = (blocks+1)/2;
    bottom_part_b_size = blocks/2;
    // printf("Up Blocks: %d, Bottom Blocks: %d\n", up_part_b_size, bottom_part_b_size);

    dim3 block_dim(TH_DIM, TH_DIM, 1);
    dim3 phase3_grid(blocks, blocks, 1);
    dim3 phase31_grid(blocks, up_part_b_size, 1);
    dim3 phase32_grid(blocks, bottom_part_b_size, 1);

    for (int k = 0; k < blocks; k++) {
        // Phase 1
        cudaSetDevice(0);
        floyd_warshall_block_kernel_phase1<<<1, block_dim>>>(padding_n, k, Dist_cuda0);

        cudaSetDevice(1);
        floyd_warshall_block_kernel_phase1<<<1, block_dim>>>(padding_n, k, Dist_cuda1);

        // Phase 2
        cudaSetDevice(0);
        floyd_warshall_block_kernel_phase2<<<blocks, block_dim>>>(padding_n, k, Dist_cuda0);

        cudaSetDevice(1);
        floyd_warshall_block_kernel_phase2<<<blocks, block_dim>>>(padding_n, k, Dist_cuda1);

        // Phase 3
        cudaSetDevice(0);
        floyd_warshall_block_kernel_phase3<<<phase3_grid, block_dim>>>(padding_n, k, Dist_cuda0, 0, 0);
        // floyd_warshall_block_kernel_phase3<<<phase31_grid, block_dim>>>(padding_n, k, Dist_cuda0, 0, 0);

        // cudaSetDevice(1);
        // floyd_warshall_block_kernel_phase3<<<phase32_grid, block_dim>>>(padding_n, k, Dist_cuda1, 0, up_part_b_size);
    }
}

void block_FW_cuda0() {
    const int blocks = padding_n / BLOCK_DIM;
    const int row_b_size = BLOCK_DIM * padding_n;
    up_part_b_size = (blocks+1)/2;
    bottom_part_b_size = blocks/2;
    // printf("Up Blocks: %d, Bottom Blocks: %d\n", up_part_b_size, bottom_part_b_size);

    dim3 block_dim(TH_DIM, TH_DIM, 1);
    dim3 phase31_grid(blocks, up_part_b_size, 1);
    dim3 phase32_grid(blocks, bottom_part_b_size, 1);

    for (int k = 0; k < blocks; k++) {
        if(k < up_part_b_size){
            // Stage 1
            // printf("Round %d Before Copy\n", k);
            // printf("Matrix 0\n");
            // show_mat_cuda(Dist_cuda0, padding_n, 0);
            // printf("Matrix 1\n");
            // show_mat_cuda(Dist_cuda1, padding_n, 1);
            cudaSetDevice(0);
            floyd_warshall_block_kernel_phase1<<<1, block_dim>>>(padding_n, k, Dist_cuda0);
            floyd_warshall_block_kernel_phase2<<<blocks, block_dim>>>(padding_n, k, Dist_cuda0);
            cudaMemcpyPeer(&(Dist_cuda1[k * row_b_size]), 1, &(Dist_cuda0[k * row_b_size]), 0, SIZEOFINT * row_b_size); 
            // printf("%d. %d ~ %d Copy Done\n", k, k * row_b_size,  (k * row_b_size) + (row_b_size), row_b_size);

            cudaDeviceSynchronize();

            cudaSetDevice(1);
            // printf("After Copy\n");
            // printf("Matrix 0\n");
            // show_mat_cuda(Dist_cuda0, padding_n, 0);
            // printf("Matrix 1\n");
            // show_mat_cuda(Dist_cuda1, padding_n, 1);
            // printf("Down Part\n------------\n");
            floyd_warshall_block_kernel_phase2<<<blocks, block_dim>>>(padding_n, k, Dist_cuda1);
        }else{
            // Stage 2
            cudaSetDevice(1);
            floyd_warshall_block_kernel_phase1<<<1, block_dim>>>(padding_n, k, Dist_cuda1);
            floyd_warshall_block_kernel_phase2<<<blocks, block_dim>>>(padding_n, k, Dist_cuda1);
            cudaMemcpyPeer(&(Dist_cuda0[k * row_b_size]), 0, &(Dist_cuda1[k * row_b_size]), 1, SIZEOFINT * row_b_size); 
            // printf("%d. %d ~ %d Copy Done\n", k, k * row_b_size,  (k * row_b_size) + (row_b_size), row_b_size);

            cudaDeviceSynchronize();

            cudaSetDevice(0);
            // printf("Up Part\n------------\n");
            floyd_warshall_block_kernel_phase2<<<blocks, block_dim>>>(padding_n, k, Dist_cuda0);
        }

        // Phase 3
        cudaSetDevice(0);
        floyd_warshall_block_kernel_phase3<<<phase31_grid, block_dim>>>(padding_n, k, Dist_cuda0, 0, 0);

        cudaSetDevice(1);
        floyd_warshall_block_kernel_phase3<<<phase32_grid, block_dim>>>(padding_n, k, Dist_cuda1, 0, up_part_b_size);
    }
}

void block_FW_cuda1() {
    const int blocks = padding_n / BLOCK_DIM;
    const int row_b_size = BLOCK_DIM * padding_n;
    up_part_b_size = (blocks+1)/2;
    bottom_part_b_size = blocks/2;
    // printf("Up Blocks: %d, Bottom Blocks: %d\n", up_part_b_size, bottom_part_b_size);

    dim3 block_dim(TH_DIM, TH_DIM, 1);
    // dim3 phase3_grid(blocks, blocks, 1);
    dim3 phase31_grid(blocks, up_part_b_size, 1);
    dim3 phase32_grid(blocks, bottom_part_b_size, 1);

    const int num_stream = 2;
    cudaStream_t streams[num_stream];
    for(int i=0; i<num_stream; i++) {cudaStreamCreate(&streams[i]);}

    for (int k = 0; k < blocks; k++) {
        // Phase 1
        cudaSetDevice(0);
        floyd_warshall_block_kernel_phase1<<<1, block_dim, 0>>>(padding_n, k, Dist_cuda0);

        cudaSetDevice(1);
        floyd_warshall_block_kernel_phase1<<<1, block_dim, 0>>>(padding_n, k, Dist_cuda1);

        // Phase 2
        cudaSetDevice(0);
        floyd_warshall_block_kernel_phase2<<<blocks, block_dim, 0>>>(padding_n, k, Dist_cuda0);

        cudaSetDevice(1);
        floyd_warshall_block_kernel_phase2<<<blocks, block_dim, 0>>>(padding_n, k, Dist_cuda1);

        // Phase 3
        cudaSetDevice(0);
        floyd_warshall_block_kernel_phase3<<<phase31_grid, block_dim, 0>>>(padding_n, k, Dist_cuda0, 0, 0);

        cudaSetDevice(1);
        floyd_warshall_block_kernel_phase3<<<phase32_grid, block_dim, 0>>>(padding_n, k, Dist_cuda1, 0, up_part_b_size);

        // Transfer data to another GPU
        int next_k = k + 1;
        if(next_k < up_part_b_size){
            cudaMemcpyPeerAsync(&(Dist_cuda1[next_k * row_b_size]), 1, &(Dist_cuda0[next_k * row_b_size]), 0, SIZEOFINT * row_b_size); 
        }else if(up_part_b_size < next_k < blocks){
            cudaMemcpyPeerAsync(&(Dist_cuda0[next_k * row_b_size]), 0, &(Dist_cuda1[next_k * row_b_size]), 1, SIZEOFINT * row_b_size); 
        }
    }
    for(int i=0; i<num_stream; i++) {cudaStreamDestroy(streams[i]);}
}

void block_FW_cuda2() {
    const int blocks = padding_n / BLOCK_DIM;
    const int row_b_size = BLOCK_DIM * padding_n;
    up_part_b_size = (blocks+1)/2;
    bottom_part_b_size = blocks/2;
    // printf("Up Blocks: %d, Bottom Blocks: %d\n", up_part_b_size, bottom_part_b_size);

    dim3 block_dim(TH_DIM, TH_DIM, 1);
    dim3 phase31_grid(blocks, up_part_b_size, 1);
    dim3 phase32_grid(blocks, bottom_part_b_size, 1);

    for (int k = 0; k < blocks; k++) {
        if(k < up_part_b_size){
            // Stage 1
            // printf("Round %d Before Copy\n", k);
            // printf("Matrix 0\n");
            // show_mat_cuda(Dist_cuda0, padding_n, 0);
            // printf("Matrix 1\n");
            // show_mat_cuda(Dist_cuda1, padding_n, 1);
            cudaSetDevice(0);
            floyd_warshall_block_kernel_phase1<<<1, block_dim>>>(padding_n, k, Dist_cuda0);
            floyd_warshall_block_kernel_phase2<<<blocks, block_dim>>>(padding_n, k, Dist_cuda0);
            cudaMemcpyPeer(&(Dist_cuda1[k * row_b_size]), 1, &(Dist_cuda0[k * row_b_size]), 0, SIZEOFINT * row_b_size); 
            // printf("%d. %d ~ %d Copy Done\n", k, k * row_b_size,  (k * row_b_size) + (row_b_size), row_b_size);

            cudaDeviceSynchronize();

            cudaSetDevice(1);
            // printf("After Copy\n");
            // printf("Matrix 0\n");
            // show_mat_cuda(Dist_cuda0, padding_n, 0);
            // printf("Matrix 1\n");
            // show_mat_cuda(Dist_cuda1, padding_n, 1);
            // printf("Down Part\n------------\n");
            floyd_warshall_block_kernel_phase2<<<blocks, block_dim>>>(padding_n, k, Dist_cuda1);
        }else{
            // Stage 2
            cudaSetDevice(1);
            floyd_warshall_block_kernel_phase1<<<1, block_dim>>>(padding_n, k, Dist_cuda1);
            floyd_warshall_block_kernel_phase2<<<blocks, block_dim>>>(padding_n, k, Dist_cuda1);
            cudaMemcpyPeer(&(Dist_cuda0[k * row_b_size]), 0, &(Dist_cuda1[k * row_b_size]), 1, SIZEOFINT * row_b_size); 
            // printf("%d. %d ~ %d Copy Done\n", k, k * row_b_size,  (k * row_b_size) + (row_b_size), row_b_size);

            cudaDeviceSynchronize();

            cudaSetDevice(0);
            // printf("Up Part\n------------\n");
            floyd_warshall_block_kernel_phase2<<<blocks, block_dim>>>(padding_n, k, Dist_cuda0);
        }

        // Phase 3
        cudaSetDevice(0);
        floyd_warshall_block_kernel_phase3<<<phase31_grid, block_dim>>>(padding_n, k, Dist_cuda0, 0, 0);

        cudaSetDevice(1);
        floyd_warshall_block_kernel_phase3<<<phase32_grid, block_dim>>>(padding_n, k, Dist_cuda1, 0, up_part_b_size);
    }
}

int main(int argc, char* argv[]) {
    input(argv[1]);
    // show_mat(getDistAddr(0, 0), padding_n);
    // printf("Original Matix\n");
    setup_DistCuda();
    // printf("Vertice: %d, Edge: %d, BLOCK_DIM: %d\n", n, m, BLOCK_DIM);
    block_FW_cuda1();
    back_DistCuda();
    // show_mat(getDistAddr(0, 0), n);
    
    output(argv[2]);
    // show_mat(Dist, padding_n);
    // printf("------------\n");
    // show_mat(Dist_s, n);
    return 0;
}