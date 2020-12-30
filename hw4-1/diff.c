#include <stdio.h>
#include <stdlib.h>

int n, squr;
int *targraph, *corgraph;
int count;
char is_right = 1;

int main(int argc, char* argv[]) {
    n = atoi(argv[1]);
    count = 0;
    squr = n * n;
    targraph = (int*)malloc(sizeof(int) * squr);
    corgraph = (int*)malloc(sizeof(int) * squr);
    FILE* tarfile = fopen(argv[2], "rb");
    FILE* corfile = fopen(argv[3], "rb");
    fread(targraph, sizeof(int), squr, tarfile);
    fread(corgraph, sizeof(int), squr, corfile);

    printf("Vertice: %d\n", n);
    // printf("%d\n", targraph[0]);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            // printf("%d\n", targraph[count]);
            if(targraph[count] != corgraph[count]){
                is_right = 0;
                printf("(%d %d) %d should be %d\n", i, j, targraph[count], corgraph[count]);
            }
            count++;
        }
    }

    
    // // output(argv[2]);
    // // show_mat(getDistAddr(0, 0, n), n);
    if(is_right){
        printf("Correct\n");
    }else{
        printf("Wrong\n");
    }
    fclose(tarfile);
    fclose(corfile);
    return 0;
}