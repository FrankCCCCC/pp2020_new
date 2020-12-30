#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define SIZEOFINT sizeof(int)

const dim3 block_dim(32, 32);

const int INF = ((1 << 30) - 1);

const int B = 64;
int n, m;
int *Dist;
int *Dist_cuda;

// CUDA Timer
// 0: Computing, 1: H2D, 2: D2H, 3: I/O Read, 4: I/O Write
const int timer_len = 5;
cudaEvent_t start[timer_len], stop[timer_len];

void init_cuda_timer(){
    for(int i=0; i<timer_len; i++){
        cudaEventCreate(&(start[i]));
        cudaEventCreate(&(stop[i]));
    }
}

void show_time(){
    float sum = 0;
    printf("%d,\t", n);
    for(int i=0; i<timer_len-1; i++){
        float t = 0;
        cudaEventSynchronize(stop[i]);
        cudaEventElapsedTime(&t, start[i], stop[i]);
        printf("%f,\t", t);
        sum += t;
    }
    float t = 0;
    cudaEventSynchronize(stop[timer_len-1]);
    cudaEventElapsedTime(&t, start[timer_len-1], stop[timer_len-1]);
    sum += t;
    printf("%f, %f\n", t, sum);
}


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

void malloc_Dist(int vertex_num){Dist = (int*)malloc(SIZEOFINT * vertex_num * vertex_num);}
int getDist(int i, int j, int vertex_num){return Dist[i * vertex_num + j];}
int *getDistAddr(int i, int j, int vertex_num){return &(Dist[i * vertex_num + j]);}
void setDist(int i, int j, int val, int vertex_num){Dist[i * vertex_num + j] = val;}

void setup_DistCuda(int vertex_num){
    cudaEventRecord(start[1]);
    cudaMalloc((void **)&Dist_cuda, SIZEOFINT * vertex_num * vertex_num);
    cudaMemcpy(Dist_cuda, Dist, (n * n * SIZEOFINT), cudaMemcpyHostToDevice);
    cudaEventRecord(stop[1]);
}
void back_DistCuda(int vertex_num){
    cudaEventRecord(start[2]);
    cudaMemcpy(Dist, Dist_cuda, (n * n * SIZEOFINT), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop[2]);
}
// int getDistCuda(int i, int j, int vertex_num){return Dist_cuda[i * vertex_num + j];}
// int *getDistAddrCuda(int i, int j, int vertex_num){return &(Dist_cuda[i * vertex_num + j]);}
// void setDistCuda(int i, int j, int val, int vertex_num){Dist_cuda[i * vertex_num + j] = val;}

void input(char* infile) {
    cudaEventRecord(start[3], 0);
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    malloc_Dist(n);
    // malloc_DistCuda(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                setDist(i, j, 0, n);
                // Dist[i][j] = 0;
            } else {
                setDist(i, j, INF, n);
                // Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; i++) {
        fread(pair, sizeof(int), 3, file);
        setDist(pair[0], pair[1], pair[2], n);
        // Dist[pair[0]][pair[1]] = pair[2];
    }
    // cudaMemcpy(Dist_cuda, Dist, (n * n * SIZEOFINT), cudaMemcpyHostToDevice);
    fclose(file);
    cudaEventRecord(stop[3], 0);
}

void output(char* outFileName) {
    cudaEventRecord(start[4], 0);
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // if (Dist[i][j] >= INF) Dist[i][j] = INF;
            if (getDist(i, j, n) >= INF) setDist(i, j, INF, n);
        }
        // fwrite(Dist[i], sizeof(int), n, outfile);
        // fwrite(getDistAddr(i, 0, n), sizeof(int), n, outfile);
    }
    fwrite(getDistAddr(0, 0, n), sizeof(int), n * n, outfile);
    fclose(outfile);
    cudaEventRecord(stop[4], 0);
}

__global__ void cal_cuda(int *dist, int vertex_num, int edge_num, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    const int Share_Mem_Size = 64;
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;
    // printf("%d\n", dist[1]);
    // i-j block
    int (*AM)[Share_Mem_Size][Share_Mem_Size];
    __shared__ int a[Share_Mem_Size][Share_Mem_Size];
    // i-k block
    int (*BM)[Share_Mem_Size][Share_Mem_Size];
    __shared__ int b[Share_Mem_Size][Share_Mem_Size];
    // k-j block
    int (*CM)[Share_Mem_Size][Share_Mem_Size];
    __shared__ int c[Share_Mem_Size][Share_Mem_Size];

    for (int b_i = block_start_x + blockIdx.x; b_i < block_end_x; b_i+=gridDim.x) {
        for (int b_j = block_start_y + blockIdx.y; b_j < block_end_y; b_j+=gridDim.y) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times

            // To calculate original index of elements in the block (b_i, b_j)
            // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
            int block_internal_start_x = b_i * B;
            int block_internal_end_x = (b_i + 1) * B;
            int block_internal_start_y = b_j * B;
            int block_internal_end_y = (b_j + 1) * B;

            if (block_internal_end_x > vertex_num) block_internal_end_x = vertex_num;
            if (block_internal_end_y > vertex_num) block_internal_end_y = vertex_num;
            
            // if(threadIdx.x == 0 && threadIdx.y == 0){
            //     printf("(%d %d) A(%d:%d, %d:%d) B(%d:%d, %d:%d) C(%d:%d, %d:%d) CAL(%d:%d, %d:%d, %d:%d)\n", 
            //            blockDim.x, blockDim.y, 
            //            block_internal_start_x + threadIdx.x, block_internal_end_x, block_internal_start_y + threadIdx.y, block_internal_end_y,
            //            block_internal_start_x + threadIdx.x, block_internal_end_x, Round * B, (Round + 1) * B < vertex_num? (Round + 1) * B : vertex_num,
            //            Round * B, (Round + 1) * B < vertex_num? (Round + 1) * B : vertex_num, block_internal_start_y + threadIdx.y, block_internal_end_y,
            //            block_internal_start_x + threadIdx.x, block_internal_end_x, block_internal_start_y + threadIdx.y, block_internal_end_y, Round * B, (Round + 1) * B < vertex_num? (Round + 1) * B : vertex_num
            //         );
            // }
            
            // AM = &a;
            // for (int i = block_internal_start_x + threadIdx.x; i < block_internal_end_x; i+=blockDim.x) {
            //     for (int j = block_internal_start_y + threadIdx.y; j < block_internal_end_y; j+=blockDim.y) {
            //         a[i - block_internal_start_x][j - block_internal_start_y] = dist[i * vertex_num + j];
            //     }
            // }

            // if(Round != b_i){
            //     CM = &c;
            //     for (int k = Round * B + threadIdx.x; k < (Round + 1) * B && k < vertex_num; k+=blockDim.x) {
            //         for (int j = block_internal_start_y + threadIdx.y; j < block_internal_end_y; j+=blockDim.y) {
            //             c[k - Round * B][j - block_internal_start_y] = dist[k * vertex_num + j];
            //         }
            //     }
            // }else{CM = &a;}

            // if(Round != b_j){
            //     BM = &b;
            //     for (int k = Round * B + threadIdx.y; k < (Round + 1) * B && k < vertex_num; k+=blockDim.y) {
            //         for (int i = block_internal_start_x + threadIdx.x; i < block_internal_end_x; i+=blockDim.x) {
            //             b[i - block_internal_start_x][k - Round * B] = dist[i * vertex_num + k];
            //         }
            //     }
            // }else{BM = &a;}
            // __syncthreads();

            // Relax Path
            for (int k = Round * B; k < (Round + 1) * B && k < vertex_num; k++) {
                for (int i = block_internal_start_x + threadIdx.x; i < block_internal_end_x; i+=blockDim.x) {
                    for (int j = block_internal_start_y + threadIdx.y; j < block_internal_end_y; j+=blockDim.y) {
                        int d = dist[i * vertex_num + k] + dist[k * vertex_num + j];
                        // __syncthreads();
                        if (d < dist[i * vertex_num + j]) {
                            // a[i - block_internal_start_x][j - block_internal_start_y] = d;
                            dist[i * vertex_num + j] = d;
                        }
                    }
                }
                __syncthreads();
            }
            // Move modified block to global memory
            // for (int i = block_internal_start_x + threadIdx.x; i < block_internal_end_x; i+=blockDim.x) {
            //     for (int j = block_internal_start_y + threadIdx.y; j < block_internal_end_y; j+=blockDim.y) {
            //         dist[i * vertex_num + j] = (*AM)[i - block_internal_start_x][j - block_internal_start_y];
            //     }
            // }
        }
    }
}

void block_FW_cuda(int B) {
    int round = (n + B - 1) / B;
    cudaEventRecord(start[0], 0);
    for (int r = 0; r < round; r++) {
        // printf("Round: %d in total: %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        cal_cuda<<<1, block_dim>>>(Dist_cuda, n, m, B, r, r, r, 1, 1);

        /* Phase 2*/
        cal_cuda<<<r, block_dim>>>(Dist_cuda, n, m, B, r, r, 0, r, 1);
        cal_cuda<<<round - r - 1, block_dim>>>(Dist_cuda, n, m, B, r, r, r + 1, round - r - 1, 1);
        cal_cuda<<<r, block_dim>>>(Dist_cuda, n, m, B, r, 0, r, 1, r);
        cal_cuda<<<round - r - 1, block_dim>>>(Dist_cuda, n, m, B, r, r + 1, r, 1, round - r - 1);

        /* Phase 3*/
        const dim3 grid_dim0(r, r);
        const dim3 grid_dim1(round - r - 1, r);
        const dim3 grid_dim2(r, round - r - 1);
        const dim3 grid_dim3(round - r - 1, round - r - 1);
        cal_cuda<<<grid_dim0, block_dim>>>(Dist_cuda, n, m, B, r, 0, 0, r, r);
        cal_cuda<<<grid_dim1, block_dim>>>(Dist_cuda, n, m, B, r, 0, r + 1, round - r - 1, r);
        cal_cuda<<<grid_dim2, block_dim>>>(Dist_cuda, n, m, B, r, r + 1, 0, r, round - r - 1);
        cal_cuda<<<grid_dim3, block_dim>>>(Dist_cuda, n, m, B, r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
    cudaEventRecord(stop[0], 0);
}

int main(int argc, char* argv[]) {
    init_cuda_timer();
    input(argv[1]);
    // show_mat(getDistAddr(0, 0, n), n);
    setup_DistCuda(n);
    // printf("Vertice: %d, Edge: %d, B: %d\n", n, m, B);
    block_FW_cuda(B);
    back_DistCuda(n);
    // show_mat(getDistAddr(0, 0, n), n);
    
    output(argv[2]);
    // show_mat(getDistAddr(0, 0, n), n);
    show_time();
    return 0;
}