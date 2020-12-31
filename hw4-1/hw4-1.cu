#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define SIZEOFINT sizeof(int)
#define BLOCK_DIM 32
#define TH_DIM 32

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
    // cudaHostAlloc(&Dist, SIZEOFINT * padding_n * padding_n, cudaHostAllocMapped);
    Dist = (int*)malloc(SIZEOFINT * padding_n * padding_n);
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
  #pragma unroll 5
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

  __syncthreads();

  // Transfer back to graph
  graph[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + bj)] = C[bi*BLOCK_DIM + bj];
  // graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + bj)] = C[(bi + TH_DIM)*BLOCK_DIM + bj];
  // graph[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + (bj + TH_DIM))] = C[bi*BLOCK_DIM + (bj + TH_DIM)];
  // graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + (bj + TH_DIM))] = C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)];
}


__global__ void floyd_warshall_block_kernel_phase2(int n, int k, int* graph) {
  // BlockDim is one dimensional (Straight along diagonal)
  // Blocks themselves are two dimensional
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

  __syncthreads();

  graph[(i*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + bj)] = C[bi*BLOCK_DIM + bj];
  // graph[(i*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + bj)] = C[(bi + TH_DIM)*BLOCK_DIM + bj];
  // graph[(i*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + (bj + TH_DIM))] = C[bi*BLOCK_DIM + (bj + TH_DIM)];
  // graph[(i*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + (bj + TH_DIM))] = C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)];

  // Phase 2 1/2

  C[bi*BLOCK_DIM + bj] = graph[(k*BLOCK_DIM + bi)*n + (i*BLOCK_DIM + bj)];
  // C[(bi + TH_DIM)*BLOCK_DIM + bj] = graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (i*BLOCK_DIM + bj)];
  // C[bi*BLOCK_DIM + (bj + TH_DIM)] = graph[(k*BLOCK_DIM + bi)*n + (i*BLOCK_DIM + (bj + TH_DIM))];
  // C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)] = graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (i*BLOCK_DIM + (bj + TH_DIM))];

  A[bi*BLOCK_DIM + bj] = graph[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + bj)];
  // A[(bi + TH_DIM)*BLOCK_DIM + bj] = graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + bj)];
  // A[bi*BLOCK_DIM + (bj + TH_DIM)] = graph[(k*BLOCK_DIM + bi)*n + (k*BLOCK_DIM + (bj + TH_DIM))];
  // A[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)] = graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (k*BLOCK_DIM + (bj + TH_DIM))];

  __syncthreads();

  block_calc(C, A, C, bi, bj);

  __syncthreads();

  // Block C is the only one that could be changed
  graph[(k*BLOCK_DIM + bi)*n + (i*BLOCK_DIM + bj)] = C[bi*BLOCK_DIM + bj];
  // graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (i*BLOCK_DIM + bj)] = C[(bi + TH_DIM)*BLOCK_DIM + bj];
  // graph[(k*BLOCK_DIM + bi)*n + (i*BLOCK_DIM + (bj + TH_DIM))] = C[bi*BLOCK_DIM + (bj + TH_DIM)];
  // graph[(k*BLOCK_DIM + (bi + TH_DIM))*n + (i*BLOCK_DIM + (bj + TH_DIM))] = C[(bi + TH_DIM)*BLOCK_DIM + (bj + TH_DIM)];
}


__global__ void floyd_warshall_block_kernel_phase3(int n, int k, int* graph) {
  // BlockDim is one dimensional (Straight along diagonal)
  // Blocks themselves are two dimensional
  const unsigned int j = blockIdx.x;
  const unsigned int i = blockIdx.y;
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


void block_FW_cuda(int B) {
    // int round = padding_n / B;
    const int blocks = (padding_n + BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 block_dim(TH_DIM, TH_DIM, 1);
    dim3 phase4_grid(blocks, blocks, 1);

  // std::cout << "Launching Kernels Blocks: " << blocks << " Size " << padding_n << "\n";
  for (int k = 0; k < blocks; k++) {
    floyd_warshall_block_kernel_phase1<<<1, block_dim>>>(padding_n, k, Dist_cuda);

    floyd_warshall_block_kernel_phase2<<<blocks, block_dim>>>(padding_n, k, Dist_cuda);

    floyd_warshall_block_kernel_phase3<<<phase4_grid, block_dim>>>(padding_n, k, Dist_cuda);
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