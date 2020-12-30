// #include <stdio.h>
// #include <stdlib.h>
// #include <pthread.h>
// #include <cuda_runtime.h>
// #include <cuda.h>

// #define SIZEOFINT sizeof(int)
// const int INF = ((1 << 30) - 1);
// const int blockdim_x = 32, blockdim_y = 32;
// const dim3 block_dim(blockdim_x, blockdim_y);
// const int B = 32;
// const int Share_Mem_Size = 64;
// const int Share_Mem_Size_sq = Share_Mem_Size * Share_Mem_Size;
// const int Share_Mem_Row_Size = B;
// int n, m, padding_n;
// int *Dist, *Dist_s;
// int *Dist_cuda0, *Dist_cuda1;

// // void show_mat(int *start_p, int vertex_num){
// //     for(int i = 0; i < vertex_num; i++){
// //         for(int j = 0; j < vertex_num; j++){
// //             if(start_p[i * vertex_num + j] == INF){
// //                 printf("INF\t  ");
// //             }else{
// //                 printf("%d\t  ", start_p[i * vertex_num + j]);
// //             }   
// //         }
// //         printf("\n");
// //     }
// // }

// void malloc_Dist(){
//     Dist = (int*)malloc(SIZEOFINT * padding_n * padding_n);
//     Dist_s = (int*)malloc(SIZEOFINT * n * n);
// }
// int getDist(int i, int j){return Dist[i * padding_n + j];}
// int *getDistAddr(int i, int j){return &(Dist[i * padding_n + j]);}
// void setDist(int i, int j, int val){Dist[i * padding_n + j] = val;}

// void setup_DistCuda(){
//     cudaSetDevice(0);
//     cudaMalloc((void **)&Dist_cuda0, SIZEOFINT * padding_n * padding_n);
//     cudaMemcpy(Dist_cuda0, Dist, (padding_n * padding_n * SIZEOFINT), cudaMemcpyHostToDevice);

//     cudaSetDevice(1);
//     cudaMalloc((void **)&Dist_cuda1, SIZEOFINT * padding_n * padding_n);
//     cudaMemcpy(Dist_cuda1, Dist, (padding_n * padding_n * SIZEOFINT), cudaMemcpyHostToDevice);
// }
// void back_DistCuda(){
//     cudaMemcpy(Dist, Dist_cuda0, (padding_n * padding_n * SIZEOFINT), cudaMemcpyDeviceToHost);
// }

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define SIZEOFINT sizeof(int)
const int INF = ((1 << 30) - 1);
const int blockdim_x = 32, blockdim_y = 32;
const dim3 block_dim(blockdim_x, blockdim_y);
const int B = 32;
const int Share_Mem_Size = 64;
const int Share_Mem_Size_sq = Share_Mem_Size * Share_Mem_Size;
const int Share_Mem_Row_Size = B;
int n, m, padding_n;
int *Dist, *Dist_s;
int *Dist_cuda;

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
    // cudaHostAlloc(&Dist, SIZEOFINT * padding_n * padding_n, cudaHostAllocMapped);
    // Dist = (int*)malloc(SIZEOFINT * padding_n * padding_n);
    // cudaMallocHost(&Dist, SIZEOFINT * padding_n * padding_n);
    
    Dist_s = (int*)malloc(SIZEOFINT * n * n);
}
int getDist(int i, int j){return Dist[i * padding_n + j];}
int *getDistAddr(int i, int j){return &(Dist[i * padding_n + j]);}
void setDist(int i, int j, int val){Dist[i * padding_n + j] = val;}

void setup_DistCuda(){
    cudaMalloc((void **)&Dist_cuda, SIZEOFINT * padding_n * padding_n);
    cudaMemcpy(Dist_cuda, Dist, (padding_n * padding_n * SIZEOFINT), cudaMemcpyHostToDevice);
    // cudaHostGetDevicePointer(&Dist_cuda, Dist, 0);
}
void back_DistCuda(){
    cudaMemcpy(Dist, Dist_cuda, (padding_n * padding_n * SIZEOFINT), cudaMemcpyDeviceToHost);
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    padding_n = ((n + B - 1) / B) * B;
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

extern __shared__ int sm[];
__global__ void phase1_cal_cuda(int *dist, int vertex_num, int B, int Round, int block_start_x, int block_start_y) {
    // i-j block
    int *a = &(sm[0]);

    // To calculate B*B elements in the block (b_i, b_j)
    // For each block, it need to compute B times
    int b_i = block_start_x + blockIdx.x;
    int b_j = block_start_y + blockIdx.y;

    // To calculate original index of elements in the block (b_i, b_j)
    // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
    int block_internal_start_x = b_i * B;
    int block_internal_start_y = b_j * B;
    
    a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = dist[(block_internal_start_x + threadIdx.x) * vertex_num + (block_internal_start_y + threadIdx.y)];
    __syncthreads();

    // Relax Path
    for (int k = 0; k < B; k++) {
        int d = a[(threadIdx.x) * Share_Mem_Row_Size + (k)] + a[(k) * Share_Mem_Row_Size + (threadIdx.y)];
        a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = min(d, a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)]);
        __syncthreads();
    }
    // Move modified block to global memory
    dist[(block_internal_start_x + threadIdx.x) * vertex_num + (block_internal_start_y + threadIdx.y)] = a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)];
}

extern __shared__ int sm[];
__global__ void phase3_cal_cuda(int *dist, int vertex_num, int B, int Round, int block_start_x, int block_start_y) {
    // const int Share_Mem_Row_Size3 = 32;
    // const int Share_Mem_Size_sq = 64 * 64;
    // i-j block
    int *a = &(sm[0]);
    // i-k block
    int *b = &(sm[Share_Mem_Size_sq]);
    // k-j block
    int *c = &(sm[2 * Share_Mem_Size_sq]);

    // To calculate B*B elements in the block (b_i, b_j)
    // For each block, it need to compute B times
    int b_i = block_start_x + blockIdx.x;
    int b_j = block_start_y + blockIdx.y;

    // To calculate original index of elements in the block (b_i, b_j)
    // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
    int block_internal_start_x = b_i * B;
    int block_internal_start_y = b_j * B;
    int block_internal_start_k = Round * B;
    
    a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = dist[(block_internal_start_x + threadIdx.x) * vertex_num + (block_internal_start_y + threadIdx.y)];
    c[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = dist[(block_internal_start_k + threadIdx.x) * vertex_num + (block_internal_start_y + threadIdx.y)];
    // Reverse the row and column to ensure column-major iteration
    b[(threadIdx.y) * Share_Mem_Row_Size + (threadIdx.x)] = dist[(block_internal_start_x + threadIdx.x) * vertex_num + (block_internal_start_k + threadIdx.y)];
    __syncthreads();

    // Relax Path
    #pragma unroll 5
    for (int k = 0; k < B; k++) {
        int d = b[k * Share_Mem_Row_Size + (threadIdx.x)] + c[k * Share_Mem_Row_Size + (threadIdx.y)];
        a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = min(d, a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)]);
    }

    // Move modified block to global memory
    dist[(block_internal_start_x + threadIdx.x) * vertex_num + (block_internal_start_y + threadIdx.y)] = a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)];
}
extern __shared__ int sm[];
__global__ void phase21_cal_cuda(int *dist, int vertex_num, int B, int Round, int block_start_x, int block_start_y) {
    // i-j block
    int *a = &(sm[0]);
    // i-k block
    int *b = &(sm[Share_Mem_Size_sq]);

    // To calculate B*B elements in the block (b_i, b_j)
    // For each block, it need to compute B times
    int b_i = block_start_x + blockIdx.x;
    int b_j = block_start_y + blockIdx.y;

    // To calculate original index of elements in the block (b_i, b_j)
    // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
    int block_internal_start_x = b_i * B;
    int block_internal_start_y = b_j * B;
    int block_internal_start_k = Round * B;
    
    a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = dist[(block_internal_start_x + threadIdx.x) * vertex_num + (block_internal_start_y + threadIdx.y)];
    // Reverse the row and column to ensure column-major iteration
    b[(threadIdx.y) * Share_Mem_Row_Size + (threadIdx.x)] = dist[(block_internal_start_x + threadIdx.x) * vertex_num + (block_internal_start_k + threadIdx.y)];
    __syncthreads();

    // Relax Path
    for (int k = 0; k < B; k++) {
        int d = b[(k) * Share_Mem_Row_Size + (threadIdx.x)] + a[(k) * Share_Mem_Row_Size + (threadIdx.y)];
        a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = min(d, a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)]);
        __syncthreads();
    }
    // Move modified block to global memory
    dist[(block_internal_start_x + threadIdx.x) * vertex_num + (block_internal_start_y + threadIdx.y)] = a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)];
}
extern __shared__ int sm[];
__global__ void phase22_cal_cuda(int *dist, int vertex_num, int B, int Round, int block_start_x, int block_start_y) {
    // i-j block
    int *a = &(sm[0]);
    // k-j block
    int *c = &(sm[2 * Share_Mem_Size_sq]);

    // To calculate B*B elements in the block (b_i, b_j)
    // For each block, it need to compute B times
    int b_i = block_start_x + blockIdx.x;
    int b_j = block_start_y + blockIdx.y;

    // To calculate original index of elements in the block (b_i, b_j)
    // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
    int block_internal_start_x = b_i * B;
    int block_internal_start_y = b_j * B;
    int block_internal_start_k = Round * B;
    
    a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = dist[(block_internal_start_x + threadIdx.x) * vertex_num + (block_internal_start_y + threadIdx.y)];
    c[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = dist[(block_internal_start_k + threadIdx.x) * vertex_num + (block_internal_start_y + threadIdx.y)];
    __syncthreads();

    // Relax Path
    for (int k = 0; k < B; k++) {
        int d = a[(threadIdx.x) * Share_Mem_Row_Size + (k)] + c[(k) * Share_Mem_Row_Size + (threadIdx.y)];
        a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = min(d, a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)]);
        __syncthreads();
    }
    // Move modified block to global memory
    dist[(block_internal_start_x + threadIdx.x) * vertex_num + (block_internal_start_y + threadIdx.y)] = a[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)];
}

void block_FW_cuda(int B) {
    int round = padding_n / B;
    for (int r = 0; r < round; r++) {
        // printf("Round: %d in total: %d\n", r, round);
        // fflush(stdout);
        /* Phase 1*/
        phase1_cal_cuda<<<1, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT>>>(Dist_cuda, padding_n, B, r, r, r);

        /* Phase 2*/
        const int num_stream = 2;
        const dim3 grid_dim_p21(1, round);
        const dim3 grid_dim_p22(round, 1);
        cudaStream_t streams[num_stream];
        for(int i=0; i<num_stream; i++) {cudaStreamCreate(&streams[i]);}
        //  (block_width, block_height): (round, 1)
        phase21_cal_cuda<<<grid_dim_p21, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT, streams[0]>>>(Dist_cuda, padding_n, B, r, r, 0);
        //  (block_width, block_height): (1, round)
        phase22_cal_cuda<<<grid_dim_p22, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT, streams[1]>>>(Dist_cuda, padding_n, B, r, 0, r);
        for(int i=0; i<num_stream; i++) {cudaStreamDestroy(streams[i]);}

        // printf("After\n");
        /* Phase 3*/
        const dim3 grid_dim_p3(round, round);
        phase3_cal_cuda<<<grid_dim_p3, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT>>>(Dist_cuda, padding_n, B, r, 0, 0);
        // for(int i=0; i<num_stream; i++) {cudaStreamCreate(&streams[i]);}
        // const dim3 grid_dim_p31((round+1)/2, round);
        // phase3_cal_cuda<<<grid_dim_p31, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT, streams[0]>>>(Dist_cuda, padding_n, B, r, 0, 0);

        // const dim3 grid_dim_p32(round/2, round);
        // phase3_cal_cuda<<<grid_dim_p32, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT, streams[1]>>>(Dist_cuda, padding_n, B, r, (round+1)/2, 0);
        // for(int i=0; i<num_stream; i++) {cudaStreamDestroy(streams[i]);}
    }
}

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
    block_FW_cuda(B);
    back_DistCuda();
    // show_mat(getDistAddr(0, 0), n);
    
    output(argv[2]);
    // show_mat(getDistAddr(0, 0), n);
    return 0;
}