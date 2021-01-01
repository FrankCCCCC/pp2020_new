#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define SIZEOFINT sizeof(int)
#define BLOCK_DIM 2
#define TH_DIM 2

const int INF = ((1 << 30) - 1);
int n, m, padding_n;
int *Dist, *Dist_s;
int *Dist_cuda;
int *Dist_cuda0, *Dist_cuda1;

// void show_mat(int *start_p, int vertex_num){
//     for(int i = 0; i < vertex_num; i++){
//         for(int j = 0; j < vertex_num; j++){
//             if(start_p[i * vertex_num + j] == INF){
//                 printf("INF\t  ");
//             }else{
//                 printf("%d\t  ", start_p[i * vertex_num + j]);
//             }   
//         }
//         printf("\n");
//     }
// }

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

    cudaSetDevice(0);
    cudaMalloc((void **)&Dist_cuda0, SIZEOFINT * padding_n * padding_n);
    cudaMemcpy(Dist_cuda0, Dist, (padding_n * padding_n * SIZEOFINT), cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    cudaMalloc((void **)&Dist_cuda1, SIZEOFINT * padding_n * padding_n);
    cudaMemcpy(Dist_cuda1, Dist, (padding_n * padding_n * SIZEOFINT), cudaMemcpyHostToDevice);
}
void back_DistCuda(){
    // cudaMemcpy(Dist, Dist_cuda, (padding_n * padding_n * SIZEOFINT), cudaMemcpyDeviceToHost);
    cudaSetDevice(0);
    cudaMemcpy(Dist, Dist_cuda0, (padding_n * padding_n * SIZEOFINT), cudaMemcpyDeviceToHost);
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
    const unsigned int j = start_x + blockIdx.x;
    const unsigned int i = start_y + blockIdx.y;
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
    // int round = padding_n / B;
    const int blocks = padding_n / BLOCK_DIM;
    dim3 block_dim(TH_DIM, TH_DIM, 1);
    dim3 phase31_grid((blocks+1)/2, blocks, 1);
    dim3 phase32_grid(blocks/2, blocks, 1);

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
        floyd_warshall_block_kernel_phase3<<<phase31_grid, block_dim>>>(padding_n, k, Dist_cuda0, 0, 0);
        cudaMemcpyPeer(Dist_cuda1, 1, Dist_cuda0, 0, SIZEOFINT * ((blocks+1)/2 * BLOCK_DIM) * padding_n); 

        cudaSetDevice(1);
        floyd_warshall_block_kernel_phase3<<<phase32_grid, block_dim>>>(padding_n, k, Dist_cuda1, (blocks+1)/2, 0);
        cudaMemcpyPeer(&(Dist_cuda0[((blocks+1)/2 * BLOCK_DIM) * padding_n]), 0, &(Dist_cuda1[((blocks+1)/2 * BLOCK_DIM) * padding_n]), 1, SIZEOFINT * (blocks/2 * BLOCK_DIM) * padding_n); 

        cudaDeviceSynchronize();
    }
}

// void block_FW_cuda(int B) {
//     int round = padding_n / B;
    
//     for (int r = 0; r < round; r++) {
//         // printf("Round: %d in total: %d\n", r, round);
//         // fflush(stdout);
//         /* Phase 1*/
//         cudaSetDevice(0);
//         phase1_cal_cuda<<<1, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT>>>(Dist_cuda0, padding_n, B, r, r, r);

//         cudaSetDevice(1);
//         phase1_cal_cuda<<<1, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT>>>(Dist_cuda1, padding_n, B, r, r, r);

//         /* Phase 2*/
//         const int num_stream = 2;
//         const dim3 grid_dim_p21(1, round);
//         const dim3 grid_dim_p22(round, 1);
//         cudaStream_t streams[num_stream];

//         cudaSetDevice(0);
//         for(int i=0; i<num_stream; i++) {cudaStreamCreate(&streams[i]);}
//         //  (block_width, block_height): (round, 1)
//         phase21_cal_cuda<<<grid_dim_p21, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT, streams[0]>>>(Dist_cuda0, padding_n, B, r, r, 0);
//         //  (block_width, block_height): (1, round)
//         phase22_cal_cuda<<<grid_dim_p22, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT, streams[1]>>>(Dist_cuda0, padding_n, B, r, 0, r);
//         for(int i=0; i<num_stream; i++) {cudaStreamDestroy(streams[i]);}

//         cudaSetDevice(1);
//         for(int i=0; i<num_stream; i++) {cudaStreamCreate(&streams[i]);}
//         //  (block_width, block_height): (round, 1)
//         phase21_cal_cuda<<<grid_dim_p21, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT, streams[0]>>>(Dist_cuda1, padding_n, B, r, r, 0);
//         //  (block_width, block_height): (1, round)
//         phase22_cal_cuda<<<grid_dim_p22, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT, streams[1]>>>(Dist_cuda1, padding_n, B, r, 0, r);
//         for(int i=0; i<num_stream; i++) {cudaStreamDestroy(streams[i]);}

//         // /* Phase 3*/
//         // const dim3 grid_dim_p3(round, round);
//         // cudaSetDevice(0);
//         // phase3_cal_cuda<<<grid_dim_p3, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT>>>(Dist_cuda0, padding_n, B, r, 0, 0);
//         // // cudaMemcpyPeer(Dist_cuda1, 1, Dist_cuda0, 0, SIZEOFINT * padding_n * padding_n); 

//         // cudaSetDevice(1);
//         // phase3_cal_cuda<<<grid_dim_p3, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT>>>(Dist_cuda1, padding_n, B, r, 0, 0);
//         // // cudaMemcpyPeer(Dist_cuda0, 0, Dist_cuda1, 1, SIZEOFINT * padding_n * padding_n); 

//         // cudaDeviceSynchronize();
        
//         const dim3 grid_dim_p31((round+1)/2, round);
//         const dim3 grid_dim_p32(round/2, round);
//         cudaEvent_t d0_done, d1_done;
//         cudaEventCreate(&d0_done);
//         cudaEventCreate(&d1_done);

//         cudaSetDevice(0);
//         phase3_cal_cuda<<<grid_dim_p31, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT>>>(Dist_cuda0, padding_n, B, r, 0, 0);
//         cudaEventRecord(d0_done);
//         cudaEventSynchronize(d1_done);
//         cudaMemcpyPeer(Dist_cuda1, 1, Dist_cuda0, 0, SIZEOFINT * ((round+1)/2 * B) * padding_n); 

//         cudaSetDevice(1);
//         phase3_cal_cuda<<<grid_dim_p32, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT>>>(Dist_cuda1, padding_n, B, r, (round+1)/2, 0);
//         cudaEventRecord(d1_done);
//         cudaEventSynchronize(d0_done);
//         cudaMemcpyPeer(&(Dist_cuda0[(round/2 * B) * padding_n]), 0, &(Dist_cuda1[(round/2 * B) * padding_n]), 1, SIZEOFINT * (round/2 * B) * padding_n); 
        
//         cudaDeviceSynchronize();

//         // cudaSetDevice(0);
//         // for(int i=0; i<num_stream; i++) {cudaStreamCreate(&streams[i]);}
//         // phase3_cal_cuda<<<grid_dim_p31, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT, streams[0]>>>(Dist_cuda, padding_n, B, r, 0, 0);
//         // phase3_cal_cuda<<<grid_dim_p32, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT, streams[1]>>>(Dist_cuda, padding_n, B, r, (round+1)/2, 0);
//         // for(int i=0; i<num_stream; i++) {cudaStreamDestroy(streams[i]);}
//     }
// }

// void block_FW_cuda_p(int B) {
//     cpu_set_t cpu_set;
//     sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
//     int cpu_num = CPU_COUNT(&cpu_set);
//     printf("CPU: %d\n", cpu_num);
// }

int main(int argc, char* argv[]) {
    input(argv[1]);
    // show_mat(getDistAddr(0, 0), n);
    setup_DistCuda();
    // printf("Vertice: %d, Edge: %d, B: %d\n", n, m, B);
    block_FW_cuda();
    back_DistCuda();
    // show_mat(getDistAddr(0, 0), n);
    
    output(argv[2]);
    // show_mat(getDistAddr(0, 0), n);
    return 0;
}