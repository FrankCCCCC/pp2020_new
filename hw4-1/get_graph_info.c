#include <stdio.h>
#include <stdlib.h>

int n, m;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    printf("Vertice Num: %d, Edge Num: %d\n", n, m);
}

int main(int argc, char* argv[]){
    input(argv[1]);
}