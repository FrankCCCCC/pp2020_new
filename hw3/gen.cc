#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define SIZEOFINT 4

const int vertex_num = 5861;
const int weight_density = 95;
int w_buf[vertex_num * vertex_num * 3 + 2] = {0};
int edge_num = 0;

int get_graph_row(int idx){return idx / vertex_num;}
int get_graph_col(int idx){return idx % vertex_num;}
int get_graph_idx(int row, int col){return row * vertex_num + col;}
// int* get_graph_addr(int row, int col){return &(graph[row*vertex_num + col]);}
// int get_graph(int row, int col, FILE *f_w){fwrite(w_buf, SIZEOFINT, 3, f_w);}

void set_graph_info(){
    w_buf[0] = vertex_num; w_buf[1] = edge_num;
}
void set_graph(int row, int col, int val){
    w_buf[edge_num * 3 + 2] = row; 
    w_buf[edge_num * 3 + 2 + 1] = col; 
    w_buf[edge_num * 3 + 2 + 2] = val; 
    edge_num++;
}
void write_file(FILE *f_w){
    fwrite(w_buf, SIZEOFINT, edge_num*3+2, f_w);
}

int is_edge(){return ((rand() % 100) + 1) < weight_density;}
int rand_weight(){return (rand() % 1000) +1;}

int main(int argc, char** argv) {
    FILE *f_w = NULL;
    f_w = fopen("mycase.in", "w");
    assert(f_w != NULL);
    // w_buf = (int*)malloc( * SIZEOFINT);

    for(int i=0; i<vertex_num-1; i++){
        for(int j = 0; j < vertex_num; j++){
            if(i == j){continue;}
            if(is_edge()){set_graph(i, j, rand_weight());}
        }
    }
    for(int j = 0; j < vertex_num; j++){
        if(vertex_num-1 == j){continue;}
        set_graph(vertex_num-1, j, 0);
    }

    set_graph_info();
    write_file(f_w);
    fclose(f_w);
}