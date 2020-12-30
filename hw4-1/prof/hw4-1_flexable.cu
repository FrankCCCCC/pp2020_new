#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define SIZEOFINT sizeof(int)
const int INF = ((1 << 30) - 1);
const int blockdim_x = 8, blockdim_y = 64;
const dim3 block_dim(blockdim_x, blockdim_y);
const int B = 64;
const int Share_Mem_Size = 64;
const int Share_Mem_Size_sq = Share_Mem_Size * Share_Mem_Size;
const int Share_Mem_Row_Size = B;
int n, m, padding_n;
int *Dist;
int *Dist_cuda;

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

void malloc_Dist(){Dist = (int*)malloc(SIZEOFINT * padding_n * padding_n);}
int getDist(int i, int j){return Dist[i * padding_n + j];}
int *getDistAddr(int i, int j){return &(Dist[i * padding_n + j]);}
void setDist(int i, int j, int val){Dist[i * padding_n + j] = val;}

void setup_DistCuda(){
    cudaMalloc((void **)&Dist_cuda, SIZEOFINT * padding_n * padding_n);
    cudaMemcpy(Dist_cuda, Dist, (padding_n * padding_n * SIZEOFINT), cudaMemcpyHostToDevice);
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
        }
        // fwrite(Dist[i], sizeof(int), n, outfile);
        fwrite(getDistAddr(i, 0), SIZEOFINT, n, outfile);
    }
    // fwrite(getDistAddr(0, 0), sizeof(int), n * n, outfile);
    fclose(outfile);
}
// __device__ int min_u(int a, int b) {
//     int diff = a - b;
//     int dsgn = diff >> 31;
//     return b + (diff & dsgn);
// }
__forceinline__ __device__ void assignAij(int *dist, int (*AM), int vertex_num, int Round, int block_internal_start_x, int block_internal_end_x, int block_internal_start_y, int block_internal_end_y){
    #pragma unroll 2
    for (int i = block_internal_start_x + threadIdx.x; i < block_internal_end_x; i+=blockDim.x) {
        #pragma unroll 2
        for (int j = block_internal_start_y + threadIdx.y; j < block_internal_end_y; j+=blockDim.y) {
            (AM)[(i - block_internal_start_x) * Share_Mem_Row_Size + (j - block_internal_start_y)] = dist[i * vertex_num + j];
        }
    }

    // (AM)[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = dist[(block_internal_start_x + threadIdx.x) * vertex_num + (block_internal_start_y + threadIdx.y)];
}

__forceinline__ __device__ void assignCkj(int *dist, int (*CM), int vertex_num, int Round, int block_internal_start_x, int block_internal_end_x, int block_internal_start_y, int block_internal_end_y){
    #pragma unroll 2
    for (int k = Round * B + threadIdx.x; k < (Round + 1) * B && k < vertex_num; k+=blockDim.x) {
        #pragma unroll 2
        for (int j = block_internal_start_y + threadIdx.y; j < block_internal_end_y; j+=blockDim.y) {
            (CM)[(k - Round * B) * Share_Mem_Row_Size + (j - block_internal_start_y)] = dist[k * vertex_num + j];
        }
    }

    // (CM)[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = dist[(Round * B + threadIdx.x) * vertex_num + (block_internal_start_y + threadIdx.y)];
}

__forceinline__ __device__ void assignBik(int *dist, int (*BM), int vertex_num, int Round, int block_internal_start_x, int block_internal_end_x, int block_internal_start_y, int block_internal_end_y){
    #pragma unroll 2
    for (int k = Round * B + threadIdx.y; k < (Round + 1) * B && k < vertex_num; k+=blockDim.y) {
        #pragma unroll 2
        for (int i = block_internal_start_x + threadIdx.x; i < block_internal_end_x; i+=blockDim.x) {
            (BM)[(i - block_internal_start_x) * Share_Mem_Row_Size + (k - Round * B)] = dist[i * vertex_num + k];
        }
    }
    // (BM)[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = dist[(block_internal_start_x + threadIdx.x) * vertex_num + (Round * B + threadIdx.y)];
}

__forceinline__ __device__ void assignBik_r(int *dist, int (*BM), int vertex_num, int Round, int block_internal_start_x, int block_internal_end_x, int block_internal_start_y, int block_internal_end_y){
    #pragma unroll 2
    for (int k = Round * B + threadIdx.y; k < (Round + 1) * B && k < vertex_num; k+=blockDim.y) {
        #pragma unroll 2
        for (int i = block_internal_start_x + threadIdx.x; i < block_internal_end_x; i+=blockDim.x) {
            (BM)[(k - Round * B) * Share_Mem_Row_Size + (i - block_internal_start_x)] = dist[i * vertex_num + k];
        }
    }
    // (BM)[(threadIdx.y) * Share_Mem_Row_Size + (threadIdx.x)] = dist[(block_internal_start_x + threadIdx.x) * vertex_num + (Round * B + threadIdx.y)];
}

__forceinline__ __device__ void relax(int (*AM), int (*BM), int (*CM), int vertex_num, int Round, int block_internal_start_x, int block_internal_end_x, int block_internal_start_y, int block_internal_end_y){
    // Relax Path
    for (int k = Round * B; k < (Round + 1) * B && k < vertex_num; k++) {
        #pragma unroll 2
        for (int i = block_internal_start_x + threadIdx.x; i < block_internal_end_x; i+=blockDim.x) {
            int bv = (BM)[(i - block_internal_start_x) * Share_Mem_Row_Size + (k - Round * B)];
            #pragma unroll 2
            for (int j = block_internal_start_y + threadIdx.y; j < block_internal_end_y; j+=blockDim.y) {
                int d = bv + (CM)[(k - Round * B) * Share_Mem_Row_Size + (j - block_internal_start_y)];
                (AM)[(i - block_internal_start_x) * Share_Mem_Row_Size + (j - block_internal_start_y)] = min(d, (AM)[(i - block_internal_start_x) * Share_Mem_Row_Size + (j - block_internal_start_y)]);
            }
        }
        // int bv = (BM)[(threadIdx.x) * Share_Mem_Row_Size + (k - Round * B)];
        // int d = bv + (CM)[(k - Round * B) * Share_Mem_Row_Size + (threadIdx.y)];
        // (AM)[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = min(d, (AM)[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)]);
        __syncthreads();
    }
}

__forceinline__ __device__ void relax_r(int (*AM), int (*BM), int (*CM), int vertex_num, int Round, int block_internal_start_x, int block_internal_end_x, int block_internal_start_y, int block_internal_end_y){
    // Relax Path
    for (int k = Round * B; k < (Round + 1) * B && k < vertex_num; k++) {
        #pragma unroll 2
        for (int i = block_internal_start_x + threadIdx.x; i < block_internal_end_x; i+=blockDim.x) {
            int bv = (BM)[(k - Round * B) * Share_Mem_Row_Size + (i - block_internal_start_x)];
            #pragma unroll 2
            for (int j = block_internal_start_y + threadIdx.y; j < block_internal_end_y; j+=blockDim.y) {
                int d = bv + (CM)[(k - Round * B) * Share_Mem_Row_Size + (j - block_internal_start_y)];
                (AM)[(i - block_internal_start_x) * Share_Mem_Row_Size + (j - block_internal_start_y)] = min(d, (AM)[(i - block_internal_start_x) * Share_Mem_Row_Size + (j - block_internal_start_y)]);
            }
        }
        // int bv = (BM)[(k - Round * B) * Share_Mem_Row_Size + (threadIdx.x)];
        // int d = bv + (CM)[(k - Round * B) * Share_Mem_Row_Size + (threadIdx.y)];
        // (AM)[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = min(d, (AM)[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)]);
        __syncthreads();
    }
}

__forceinline__ __device__ void relax_r_async(int (*AM), int (*BM), int (*CM), int vertex_num, int Round, int block_internal_start_x, int block_internal_end_x, int block_internal_start_y, int block_internal_end_y){
    // Relax Path
    #pragma unroll 5
    for (int k = Round * B; k < (Round + 1) * B && k < vertex_num; k++) {
        #pragma unroll 2
        for (int i = block_internal_start_x + threadIdx.x; i < block_internal_end_x; i+=blockDim.x) {
            int bv = (BM)[(k - Round * B) * Share_Mem_Row_Size + (i - block_internal_start_x)];
            #pragma unroll 2
            for (int j = block_internal_start_y + threadIdx.y; j < block_internal_end_y; j+=blockDim.y) {
                int d = bv + (CM)[(k - Round * B) * Share_Mem_Row_Size + (j - block_internal_start_y)];
                (AM)[(i - block_internal_start_x) * Share_Mem_Row_Size + (j - block_internal_start_y)] = min(d, (AM)[(i - block_internal_start_x) * Share_Mem_Row_Size + (j - block_internal_start_y)]);
            }
        }

        // int bv = (BM)[(k - Round * B) * Share_Mem_Row_Size + (threadIdx.x)];
        // int d = bv + (CM)[(k - Round * B) * Share_Mem_Row_Size + (threadIdx.y)];
        // (AM)[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)] = min(d, (AM)[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)]);
    }
}

__forceinline__ __device__ void flush(int *dist, int (*AM), int vertex_num, int block_internal_start_x, int block_internal_end_x, int block_internal_start_y, int block_internal_end_y){
    // Move modified block to global memory
    #pragma unroll 2
    for (int i = block_internal_start_x + threadIdx.x; i < block_internal_end_x; i+=blockDim.x) {
        #pragma unroll 2
        for (int j = block_internal_start_y + threadIdx.y; j < block_internal_end_y; j+=blockDim.y) {
            dist[i * vertex_num + j] = (AM)[(i - block_internal_start_x) * Share_Mem_Row_Size + (j - block_internal_start_y)];
        }
    }

    // dist[(block_internal_start_x + threadIdx.x) * vertex_num + (block_internal_start_y + threadIdx.y)] = (AM)[(threadIdx.x) * Share_Mem_Row_Size + (threadIdx.y)];
}
extern __shared__ int sm[];
__global__ void phase1_cal_cuda(int *dist, int vertex_num, int edge_num, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    // printf("%d\n", dist[1]);
    // i-j block
    // __shared__ int a[Share_Mem_Size_sq];
    int *a = &(sm[0]);

    int b_i = block_start_x + blockIdx.x;
    int b_j = block_start_y + blockIdx.y;
    // To calculate B*B elements in the block (b_i, b_j)
    // For each block, it need to compute B times

    // To calculate original index of elements in the block (b_i, b_j)
    // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
    int block_internal_start_x = b_i * B;
    int block_internal_end_x = (b_i + 1) * B;
    int block_internal_start_y = b_j * B;
    int block_internal_end_y = (b_j + 1) * B;
    
    assignAij(dist, a, vertex_num, Round, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
    __syncthreads();

    // Relax Path
    relax(a, a, a, vertex_num, Round, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
    // Move modified block to global memory
    flush(dist, a, vertex_num, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
}

extern __shared__ int sm[];
__global__ void phase3_cal_cuda(int *dist, int vertex_num, int edge_num, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    // printf("%d\n", dist[1]);
    // i-j block
    // __shared__ int a[Share_Mem_Size_sq];
    int *a = &(sm[0]);
    // i-k block
    // __shared__ int b[Share_Mem_Size_sq];
    int *b = &(sm[Share_Mem_Size_sq]);
    // k-j block
    // __shared__ int c[Share_Mem_Size_sq];
    int *c = &(sm[2 * Share_Mem_Size_sq]);

    int b_i = block_start_x + blockIdx.x;
    int b_j = block_start_y + blockIdx.y;
    // To calculate B*B elements in the block (b_i, b_j)
    // For each block, it need to compute B times

    // To calculate original index of elements in the block (b_i, b_j)
    // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
    int block_internal_start_x = b_i * B;
    int block_internal_end_x = (b_i + 1) * B;
    int block_internal_start_y = b_j * B;
    int block_internal_end_y = (b_j + 1) * B;
    
    assignAij(dist, a, vertex_num, Round, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
    assignCkj(dist, c, vertex_num, Round, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
    // Reverse the row and column to ensure column-major iteration
    assignBik_r(dist, b, vertex_num, Round, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
    // assignBik(dist, b, vertex_num, Round, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
    __syncthreads();

    // Relax Path
    relax_r_async(a, b, c, vertex_num, Round, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
    // Move modified block to global memory
    flush(dist, a, vertex_num, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
}
extern __shared__ int sm[];
__global__ void phase21_cal_cuda(int *dist, int vertex_num, int edge_num, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    // printf("%d\n", dist[1]);
    // i-j block
    // __shared__ int a[Share_Mem_Size_sq];
    int *a = &(sm[0]);
    // i-k block
    // __shared__ int b[Share_Mem_Size_sq];
    int *b = &(sm[Share_Mem_Size_sq]);

    int b_i = block_start_x + blockIdx.x;
    int b_j = block_start_y + blockIdx.y;
    // To calculate B*B elements in the block (b_i, b_j)
    // For each block, it need to compute B times

    // To calculate original index of elements in the block (b_i, b_j)
    // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
    int block_internal_start_x = b_i * B;
    int block_internal_end_x = (b_i + 1) * B;
    int block_internal_start_y = b_j * B;
    int block_internal_end_y = (b_j + 1) * B;
    
    assignAij(dist, a, vertex_num, Round, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
    // Reverse the row and column to ensure column-major iteration
    assignBik_r(dist, b, vertex_num, Round, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
    __syncthreads();

    // Relax Path
    relax_r(a, b, a, vertex_num, Round, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
    // Move modified block to global memory
    flush(dist, a, vertex_num, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
}
extern __shared__ int sm[];
__global__ void phase22_cal_cuda(int *dist, int vertex_num, int edge_num, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    // printf("%d\n", dist[1]);
    // i-j block
    // __shared__ int a[Share_Mem_Size_sq];
    int *a = &(sm[0]);
    // k-j block
    // __shared__ int c[Share_Mem_Size_sq];
    int *c = &(sm[2 * Share_Mem_Size_sq]);

    int b_i = block_start_x + blockIdx.x;
    int b_j = block_start_y + blockIdx.y;
    // To calculate B*B elements in the block (b_i, b_j)
    // For each block, it need to compute B times

    // To calculate original index of elements in the block (b_i, b_j)
    // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
    int block_internal_start_x = b_i * B;
    int block_internal_end_x = (b_i + 1) * B;
    int block_internal_start_y = b_j * B;
    int block_internal_end_y = (b_j + 1) * B;
    
    assignAij(dist, a, vertex_num, Round, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
    assignCkj(dist, c, vertex_num, Round, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
    __syncthreads();

    // Relax Path
    relax(a, a, c, vertex_num, Round, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
    // Move modified block to global memory
    flush(dist, a, vertex_num, block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y);
}

void block_FW_cuda(int B) {
    int round = padding_n / B;
    for (int r = 0; r < round; r++) {
        // printf("Round: %d in total: %d\n", r, round);
        // fflush(stdout);
        /* Phase 1*/
        phase1_cal_cuda<<<1, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT>>>(Dist_cuda, padding_n, m, B, r, r, r, 1, 1);

        /* Phase 2*/
        const int num_stream = 2;
        const dim3 grid_dim_p21(1, round);
        const dim3 grid_dim_p22(round, 1);
        cudaStream_t streams[num_stream];
        for(int i=0; i<num_stream; i++) {cudaStreamCreate(&streams[i]);}
        phase21_cal_cuda<<<grid_dim_p21, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT, streams[0]>>>(Dist_cuda, padding_n, m, B, r, r, 0, round, 1);
        phase22_cal_cuda<<<grid_dim_p22, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT, streams[1]>>>(Dist_cuda, padding_n, m, B, r, 0, r, 1, round);
        for(int i=0; i<num_stream; i++) {
            cudaStreamDestroy(streams[i]);
        }

        // printf("After\n");
        /* Phase 3*/
        const dim3 grid_dim_p3(round, round);
        phase3_cal_cuda<<<grid_dim_p3, block_dim, 3*Share_Mem_Size_sq*SIZEOFINT>>>(Dist_cuda, padding_n, m, B, r, 0, 0, round, round);
    }
}

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