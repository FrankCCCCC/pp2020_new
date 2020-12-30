#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <time.h>

#include "libs/thread_pool/thread_pool.h"

int cpu_num = 0;

const int vec_scale = 1;
const int vec_gap = 1 << vec_scale;
const double zero_vec[2] = {0};
const double one_vec[2] = {1};
const double two_vec[2] = {2, 2};
const double four_vec[2] = {4, 4};
double iters_vec[2] = {0};

const __m128d zerov = _mm_loadu_pd(zero_vec);
const __m128d onev = _mm_loadu_pd(one_vec);
const __m128d twov = _mm_loadu_pd(two_vec);
const __m128d fourv = _mm_loadu_pd(four_vec);
const __m128d fullv = _mm_or_pd(onev, onev);
__m128d itersv = _mm_loadu_pd(zero_vec);

void assign_iters_vec(int iters){
    iters_vec[0] = (double)iters; iters_vec[1] = (double)iters;
    itersv = _mm_loadu_pd(iters_vec);
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void core_cal_sse2_sig(double *x0, double *y0, int *image){
    __m128d repeatsv = _mm_loadu_pd(zero_vec);
    __m128d xv = _mm_loadu_pd(zero_vec);
    __m128d yv = _mm_loadu_pd(zero_vec);
    __m128d length_squaredv = _mm_loadu_pd(zero_vec);

    const __m128d x0v = _mm_loadu_pd(x0);
    const __m128d y0v = _mm_loadu_pd(y0);

    int count = 0;

    while(1){
        __m128d compv = _mm_and_pd(_mm_cmplt_pd(repeatsv, itersv), _mm_cmplt_pd(length_squaredv, fourv));
        compv = (__m128d)_mm_slli_epi64((__m128i)compv, 54);
        compv = (__m128d)_mm_srli_epi64((__m128i)compv, 2);

        unsigned long int comp_vec[2] = {0, 0};
        _mm_store_pd((double*)comp_vec, compv);
        // printf("Compv0: %lu %lu, Compv: %lu %lu\n", comp_vec0[0], comp_vec0[1], comp_vec[0], comp_vec[1]);
        if((comp_vec[0] == 0) && (comp_vec[1] == 0)){break;}
        
        __m128d tempv = zerov;
        tempv = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(xv, xv), _mm_mul_pd(yv, yv)), x0v) ;
        yv = _mm_add_pd(_mm_mul_pd(twov, _mm_mul_pd(xv, yv)), y0v);
        xv = tempv;
        length_squaredv = _mm_add_pd(_mm_mul_pd(xv, xv), _mm_mul_pd(yv, yv));

        repeatsv = _mm_add_pd(repeatsv, compv);
    }   
    double image_temp[2] = {0, 0};
    _mm_store_pd(image_temp, repeatsv);
    image[0] = (int)(image_temp[0]);
    image[1] = (int)(image_temp[1]);

    // printf("Iters %d, (%d, %d) <- (%lf, %lf)\n", count, image[i], image[i + 1], image_temp[0], image_temp[1]);
}

void core_cal(int iters, double x0, double y0, int *image){
    int repeats = 0;
    double x = 0;
    double y = 0;
    double length_squared = 0;
    while (repeats < iters && length_squared < 4) {
        double temp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = temp;
        length_squared = x * x + y * y;
        ++repeats;
    }

    *image = repeats;
}

void core_cal_sse2(int iters, double *x0, double *y0, int *image, int size){
    const int size_vec = (size >> vec_scale) << vec_scale;
    for(int i = 0; i < size_vec; i+=vec_gap){
        core_cal_sse2_sig(&(x0[i]), &(y0[i]), &(image[i]));
    }

    for(int i = size_vec; i < size; i++){
        core_cal(iters, x0[i], y0[i], &(image[i]));
    }
}

typedef struct {
    // int id;
    int iters;
    double *x0;
    double *y0;
    int *image;
    int size;
}T_Task_Arg;

void make_T_Task_Arg(T_Task_Arg *arg, int iters, double *x0, double *y0, int *image, int size){
    arg->iters = iters;
    arg->x0 = x0;
    arg->y0 = y0;
    arg->image = image;
    arg->size = size;
}

void thread_task(void *arg){
    T_Task_Arg *args = (T_Task_Arg*)arg;
    core_cal_sse2(args->iters, args->x0, args->y0, args->image, args->size);
}

int main(int argc, char** argv) {
    // printf("HI, hw2 is running\n");
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    cpu_num = CPU_COUNT(&cpu_set);
    // printf("%d cpus available\n", cpu_num);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);
    const long int area = width * height;

    assign_iters_vec(iters);

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    ThreadPool *pool = create_thread_pool(cpu_num);
    T_Task_Arg *tasks_arr = (T_Task_Arg*)malloc(sizeof(T_Task_Arg) * area);

    double *x0s = (double*)malloc(sizeof(double) * area);
    double *y0s = (double*)malloc(sizeof(double) * area);
    /* mandelbrot set */
    clock_t s_time = clock();
    for (int j = 0; j < height; ++j) {
        double y0 = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * ((right - left) / width) + left;

            // core_cal_float(iters, x0, y0, &(image[j * width + i]));
            // core_cal(iters, x0, y0, &(image[j * width + i]));
            int serial_id = j * width + i;
            x0s[serial_id] = x0;
            y0s[serial_id] = y0;

            // if(serial_id == cpu_num * cpu_num){start_pool(pool);}

            make_T_Task_Arg(&(tasks_arr[serial_id]), iters, &(x0s[serial_id]), &(y0s[serial_id]), &(image[serial_id]), 4);
            submit((void (*)(void *))thread_task, (void *)(&(tasks_arr[serial_id])), pool);
        }
    }
    clock_t e_time = clock();
    printf("Tasks Submit Time %lf\n", ((double) (e_time - s_time)) * 1000 / CLOCKS_PER_SEC);
    s_time = clock();
    start_pool(pool);
    submit_done(pool);
    // printf("x0s y0s Done\n");
    // core_cal_sse2(iters, x0s, y0s, image, area);
    end_pool(pool);
    e_time = clock();
    printf("Execution Time %lf\n", ((double) (e_time - s_time)) * 1000 / CLOCKS_PER_SEC);
    free(tasks_arr);
    free(x0s);
    free(y0s);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
