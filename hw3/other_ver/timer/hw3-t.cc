#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <emmintrin.h>

#include "../../libs/timer/timer.h"

// Timer
Timer t;
Timer t_ts[30];
int is_enable_show = 1;

#define DISINF 6000*1000+1000
#define FULL 2147483647
#define DISZSELF 0
#define SIZEOFINT sizeof(int)
#define VECGAP 4
#define VECSCALE 2
#define STOPVAL -1

// int vec_counter = 0, non_vec_counter = 0;

int cpu_num = 0;
int vertex_num = 0, edge_num = 0, graph_size = 0;
int is_residual = 0;
int *buf = NULL;
int *graph = NULL;
int *block_deps = NULL;
int num_blocks = 0, block_size = 0, block_num_squr = 0, block_num_cubic = 0;
int block_assign_step = 0;

const int zero_vec[VECGAP] = {0};
const int one_vec[VECGAP] = {1, 1, 1, 1};
const unsigned int full_vec[VECGAP] = {FULL, FULL, FULL, FULL};
const __m128i zero_v = _mm_loadu_si128((const __m128i*)zero_vec);
const __m128i one_v = _mm_loadu_si128((const __m128i*)one_vec);
const __m128i full_v = _mm_loadu_si128((const __m128i*)full_vec);

void show_mat(int *g, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%d\t", g[i * n + j]);
        }
        printf("\n");
    }
}

void show_m128i(__m128i *m){
    printf("(%d\t%d\t%d\t%d)\n", ((int*)m)[0], ((int*)m)[1], ((int*)m)[2], ((int*)m)[3]);
}
int get_graph_row(int idx){return idx / vertex_num;}
int get_graph_col(int idx){return idx % vertex_num;}
int get_graph_idx(int row, int col){return row * vertex_num + col;}
int* get_graph_addr(int row, int col){return &(graph[row*vertex_num + col]);}
int get_graph(int row, int col){return graph[row*vertex_num + col];}
void set_graph(int row, int col, int val){graph[row*vertex_num + col] = val;}

typedef struct{
    int i;
    int j;
    int k;
}BlockDim;
void init_block();
BlockDim get_BlockDim(int, int, int);
BlockDim get_block_pos(int, int, int);
BlockDim get_block_size(int, int, int);

void graph_malloc();
void omp_buf2graph(int *);

void relax_v(int*, int*, int*);
void relax(int, int, int*);
void relax_block(int, int, int);
void block_floyd_warshall();

int main(int argc, char** argv) {
    if(argc >= 5){
        if(argv[4][0] == 'N'){t.disable_show(); is_enable_show=0;}
    }

    t.start_rec("total");

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    cpu_num = CPU_COUNT(&cpu_set);
    // printf("%d cpus available\n", cpu_num);

    // assert(argc == 3);
    t.start_rec("io");
    FILE *f_r = NULL, *f_w = NULL;
    f_r = fopen(argv[1], "r");
    f_w = fopen(argv[2], "w");
    assert(f_r != NULL);
    assert(f_w != NULL);

    fread(&vertex_num, SIZEOFINT, 1, f_r);
    fread(&edge_num, SIZEOFINT, 1, f_r);
    t.pause_rec("io");
    graph_size = vertex_num * vertex_num;
    t.start_rec("mem");
    buf = (int*)malloc(edge_num * SIZEOFINT * 3);
    t.pause_rec("mem");
    t.start_rec("io");
    fread(buf, SIZEOFINT, edge_num * 3, f_r);
    t.pause_rec("io");
    t.start_rec("mem");
    graph_malloc();
    t.pause_rec("mem");
    init_block();
    
    // printf("Vertex: %d Edge: %d\n", vertex_num, edge_num);
    omp_buf2graph(buf);
    t.start_rec("cpu");
    block_floyd_warshall();
    t.pause_rec("cpu");

    t.start_rec("io");
    fwrite(graph, SIZEOFINT, graph_size, f_w);
    t.pause_rec("io");

    t.pause_rec("total");

    for(int i=0; i < cpu_num; i++){
        if(!is_enable_show){t_ts[i].disable_show();}
        t_ts[i].show_rec("thread");
    }
    t.show_rec("cpu");
    t.show_rec("mem");
    t.show_rec("io");
    t.show_rec("total");

    const char *t_ts_order[1] = {"thread"};
    const char *order[4] = {"cpu", "mem", "io", "total"};
    if(argc >= 5){
        for(int i=0; i < cpu_num; i++){
            t_ts[i].report("dump.csv", t_ts_order, 1);
        }
        t.report("dump.csv", order, 4);
    }

    return 0;
}

void graph_malloc(){
    graph = (int*)malloc(graph_size * SIZEOFINT);
    memset(graph, DISZSELF, graph_size * SIZEOFINT);
}

void omp_buf2graph(int *buf){
    const int EDGE0REMARK = -1; 
    #pragma omp for schedule(guided)
    for(int i = omp_get_thread_num()*3; i < edge_num*3; i+=(omp_get_num_threads()*3)){
        // printf("Func: Edge %d - SRC: %d DST: %d WEIGHT: %d\n", i, buf[i], buf[i + 1], buf[i + 2]);
        if(buf[i + 2] == 0){
            set_graph(buf[i], buf[i + 1], EDGE0REMARK);
        }else{
            set_graph(buf[i], buf[i + 1], buf[i + 2]);
        }
    }

    #pragma omp for schedule(guided)
    for(int idx = omp_get_thread_num(); idx < graph_size; idx+=omp_get_num_threads()){
        int i = idx / vertex_num, j = idx % vertex_num;
        if(get_graph(i, j) == 0 && i != j){
            set_graph(i, j, DISINF);
        }else if(get_graph(i, j) == EDGE0REMARK){
            set_graph(i, j, 0);
        }
    }
}

void init_block(){
    // Set Block Size
    block_size = 64;
    // block_size = (((int)ceil(vertex_num / sqrt(cpu_num))) >> VECSCALE) << VECSCALE;
    if(block_size > vertex_num){block_size = vertex_num;}
    else if(block_size < VECGAP){block_size = VECGAP;}
    // printf("Block Size: %d\n", block_size);
    is_residual = vertex_num % block_size > 0;
    
    // Set Number of blocks
    num_blocks = vertex_num / block_size + is_residual;

    block_num_squr = num_blocks * num_blocks;
    block_num_cubic = num_blocks * block_num_squr;
}
BlockDim get_BlockDim(int i, int j, int k){
    BlockDim b = {i, j, k};
    return b;
}
BlockDim get_block_pos(int b_i, int b_j, int b_k){
    BlockDim bd;

    bd.i = block_size * b_i;
    bd.j = block_size * b_j;
    bd.k = block_size * b_k;
    return bd;
}

BlockDim get_block_size(int b_i, int b_j, int b_k){
    BlockDim bd;
    const int quo = vertex_num / block_size;
    if(b_i < quo){bd.i = block_size;}
    else if(b_i == num_blocks - 1){bd.i = vertex_num % block_size;}
    else{bd.i = 0;}
    
    if(b_j < quo){bd.j = block_size;}
    else if(b_j == num_blocks - 1){bd.j = vertex_num % block_size;}
    else{bd.j = 0;}

    if(b_k < quo){bd.k = block_size;}
    else if(b_k == num_blocks - 1){bd.k = vertex_num % block_size;}
    else{bd.k = 0;}
    return bd;
}

// Relax with intermediate sequence k, from sequence i to j
int relax_v(int *aij, int aik, int *akj){
    __m128i aij_v = _mm_loadu_si128((const __m128i*)aij);
    // printf("aij_v:\n");
    const int aik_vec[VECGAP] = {aik, aik, aik, aik};
    __m128i aik_v = _mm_loadu_si128((const __m128i*)aik_vec);
    // printf("aik_v:\n");
    __m128i akj_v = _mm_loadu_si128((const __m128i*)akj);
    // printf("akj_v:\n");

    __m128i sum_v = _mm_add_epi32(aik_v, akj_v);
    // printf("sum_v:\n");
    __m128i compare_gt_v = _mm_cmpgt_epi32(aij_v, sum_v);
    // printf("compare_gt_v:\n");
    __m128i compare_let_v = _mm_xor_si128(compare_gt_v, full_v);
    // printf("compare_let_v:\n");

    __m128i compgt_sum = _mm_and_si128(compare_gt_v, sum_v);
    // printf("compgt_sum:\n");
    __m128i complet_aij = _mm_and_si128(compare_let_v, aij_v);
    // printf("complet_aij:\n");
    __m128i res_v = _mm_or_si128(_mm_and_si128(compare_gt_v, sum_v), _mm_and_si128(compare_let_v, aij_v));
    // printf("res_v:\n");

    _mm_storeu_si128((__m128i*)aij, res_v);
    // printf("AIJ: %d %d %d %d\n", aij[0], aij[1], aij[2], aij[3]);

    return ((int*)(&compare_gt_v))[0] || ((int*)(&compare_gt_v))[1] || ((int*)(&compare_gt_v))[2] || ((int*)(&compare_gt_v))[3];
}
// Relax with intermediate node k, from node i to j
int relax_s(int *aij, int aik, int akj){
    if((*aij) > aik + akj){
        (*aij) = aik + akj;
        return 1;
    }
    return 0;
}
// Relax the node from A(i,j) to A(i,j+size), includes node which j+size > vertex_num
void relax(int idx, int ak, int size){
    int ai = idx / vertex_num, aj = idx % vertex_num;
    int i = ai, j = aj, remain_size = size;
    for(i = ai; i < vertex_num; i++){
        if(remain_size <= 0){return;}
        int truncated_size = j + remain_size > vertex_num? vertex_num - j : remain_size;
        int vec_size = (truncated_size >> VECSCALE) << VECSCALE;
        int vec_end = j + vec_size, single_end = j + truncated_size;
        remain_size -= truncated_size;
        
        // Relax with Vectorization speed up
        for(; j < vec_end; j+=VECGAP){
            relax_v(get_graph_addr(i, j), get_graph(i, ak), get_graph_addr(ak, j));
        }
        // Single relax
        for(; j < single_end; j++){
            relax_s(get_graph_addr(i, j), get_graph(i, ak), get_graph(ak, j));
        }
        j = 0;
    }
}
// b_i, b_j, b_k are the index of the block on the dimension i, j, k
void relax_block(int b_i, int b_j, int b_k){
    BlockDim bidx = get_block_pos(b_i, b_j, b_k);
    BlockDim bdim = get_block_size(b_i, b_j, b_k);
    // printf("B(%d %d %d), IDX(%d %d %d) DIM(%d %d %d)\n", b_i, b_j, b_k, bidx.i, bidx.j, bidx.k, bdim.i, bdim.j, bdim.k);
    for(int k = bidx.k; k < bidx.k + bdim.k; k++){
        for(int i = bidx.i; i < bidx.i + bdim.i; i++){
            relax(get_graph_idx(i, bidx.j), k, bdim.j);
        }
    }
}
// Without Vectorization, b_i, b_j, b_k are the index of the block on the dimension i, j, k
void relax_block_s(int b_i, int b_j, int b_k){
    BlockDim bidx = get_block_pos(b_i, b_j, b_k);
    BlockDim bdim = get_block_size(b_i, b_j, b_k);
    // printf("B(%d %d %d), IDX(%d %d %d) DIM(%d %d %d)\n", b_i, b_j, b_k, bidx.i, bidx.j, bidx.k, bdim.i, bdim.j, bdim.k);
    // printf("Thread %d B(%d %d %d), IDX(%d %d %d) DIM(%d %d %d)\n", omp_get_num_threads(), b_i, b_j, b_k, bidx.i, bidx.j, bidx.k, bdim.i, bdim.j, bdim.k);
    for(int k = bidx.k; k < bidx.k + bdim.k; k++){
        for(int i = bidx.i; i < bidx.i + bdim.i; i++){
            for(int j = bidx.j; j < bidx.j + bdim.j; j++){
                relax_s(get_graph_addr(i, j), get_graph(i, k), get_graph(k, j));
            }
        }
    }
}

void block_floyd_warshall(){
    for(int k = 0; k < num_blocks; k++){
        relax_block(k, k, k);
        
        #pragma omp parallel num_threads(cpu_num)
        {
            t_ts[omp_get_thread_num()].start_rec("thread");
            #pragma omp for schedule(static)
            for(int j = 0; j < num_blocks; j++){
                if(j == k){continue;}
                relax_block(k, j, k);
            }
            
            #pragma omp for schedule(static) 
            for(int i = 0; i < num_blocks; i++){
                if(i == k){continue;}
                relax_block(i, k, k);
            }
            #pragma omp for schedule(static) collapse(2)
            for(int i = 0; i < num_blocks; i++){
                for(int j = 0; j < num_blocks; j++){
                    if(i == k || j == k){continue;}
                    relax_block(i, j, k);
                }
            }
            t_ts[omp_get_thread_num()].pause_rec("thread");
        }
    }
}