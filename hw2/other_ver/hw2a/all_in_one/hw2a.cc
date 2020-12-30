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
#include <pthread.h>

// Queue
typedef struct node{
    struct node *next;
    struct node *prev;
    void *data;
}Node;

typedef struct queue{
    Node *head;
    Node *tail;
    int size;
}Queue;

Queue *create_queue();
int is_q_empty(Queue *);
int get_q_size(Queue *);
void push(Queue *, void *);
void *pop(Queue *);

// Thread Pool
typedef struct task{
    void (*func)(void *);
    void *arg;
}Task;

typedef struct{
    pthread_t *threads;
    Queue *queue;
    pthread_mutex_t *lock;
    int threads_num;
    int is_submit_done;
    int is_finish;
}ThreadPool;

typedef struct{
    ThreadPool *pool;
    int thread_id;
}WorkerArg;

ThreadPool *create_thread_pool(int);
void set_threads_num(ThreadPool*, int);
int get_threads_num(ThreadPool*);
int get_num_tasks(ThreadPool*); // synchronized function
int is_task_queue_empty(ThreadPool*); // synchronized function
int is_finish(ThreadPool*);
void submit(void (*func)(void*), void*, ThreadPool*);  // Need mutex lock
void submit_done(ThreadPool*); // Need mutex lock
int is_submit_done(ThreadPool*); 
Task *get_task(ThreadPool*); // Need mutex lock
void *worker(void*);
void start_pool(ThreadPool*);
void end_pool(ThreadPool*); // Need mutex lock
void resest_pool(ThreadPool*);
void free_pool(ThreadPool*);

// Main Program
int cpu_num = 0;
int granularity = 16;
// double *x0s = NULL;
// double *y0s = NULL;
// int* image = NULL;

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

// void core_cal_float(int iters, double x0, double y0, int *image){
//     int repeats = 0;
//     float x = 0;
//     float y = 0;
//     float length_squared = 0;
//     while (repeats < iters && length_squared < 4) {
//         float temp = x * x - y * y + (float)x0;
//         y = 2 * x * y + (float)y0;
//         x = temp;
//         length_squared = x * x + y * y;
//         ++repeats;
//     }

//     *image = repeats;
// }

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
    for (int j = 0; j < height; ++j) {
        double y0 = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * ((right - left) / width) + left;

            // core_cal_float(iters, x0, y0, &(image[j * width + i]));
            // core_cal(iters, x0, y0, &(image[j * width + i]));
            int serial_id = j * width + i;
            x0s[serial_id] = x0;
            y0s[serial_id] = y0;

            if(serial_id == cpu_num * cpu_num){start_pool(pool);}

            make_T_Task_Arg(&(tasks_arr[serial_id]), iters, &(x0s[serial_id]), &(y0s[serial_id]), &(image[serial_id]), granularity);
            submit((void (*)(void *))thread_task, (void *)(&(tasks_arr[serial_id])), pool);
        }
    }
    submit_done(pool);
    // printf("x0s y0s Done\n");
    // core_cal_sse2(iters, x0s, y0s, image, area);
    end_pool(pool);
    free(tasks_arr);
    free(x0s);
    free(y0s);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}

Queue *create_queue(){
    Queue *q = (Queue*)malloc(sizeof(Queue));
    Node *dum = (Node*)malloc(sizeof(Node));
    dum->data = NULL;
    dum->next = NULL;
    dum->prev = NULL;

    q->tail = dum;
    q->head = dum;
    q->size = 0;
    return q;
}
int is_q_empty(Queue *q){
    return q->head == q->tail;
}

int get_q_size(Queue *q){
    return q->size;
}

// next represent the neighbor node toward head direction, 
// while prev is toward tail.
// Push into the tail
void push(Queue *q, void *data){
    Node *new_node = (Node*)malloc(sizeof(Node));
    Node *temp = q->tail->next;
    // Set up the new node
    new_node->data = data;
    new_node->next = temp;
    new_node->prev = q->tail;
    // Redirect the tail to the new node
    q->tail->next = new_node;
    // printf("New_Node: %d\n", ((Task_Arg *)(new_node->data))->a);
    
    q->size++;

    if(is_q_empty(q)){
        // printf("A\n");
        q->head = new_node;
        // printf("Head: %d, Tail: %d\n", 
        //     ((Task_Arg *)(q->head->data))->a, 
        //     ((Task_Arg *)(q->tail->next->data))->a);
    }else{
        // Redirect the old last one to the new last one
        // printf("B\n");
        temp->prev = new_node;
        // printf("Head: %d, Head Prev: %d, Tail: %d, Tail Next: %d, Temp: %d\n", 
        //     ((Task_Arg *)(q->head->data))->a, 
        //     ((Task_Arg *)(q->head->prev->data))->a, 
        //     ((Task_Arg *)(q->tail->next->data))->a, 
        //     ((Task_Arg *)(q->tail->next->next->data))->a, 
        //     ((Task_Arg *)(temp->data))->a);
    }
}

// Pop from the head
void *pop(Queue *q){
    if(is_q_empty(q)){return NULL;}
    else{
        Node *pop_node = q->head;
        void *pop_data = pop_node->data;
        // Set the head node next link as NULL, prev as the same as old node's prev
        q->head = pop_node->prev;
        q->head->next = NULL;

        // Set popped node link as NULLs
        pop_node->prev = NULL;
        pop_node->next = NULL;
        pop_node->data = NULL;
        free(pop_node);

        q->size--;

        return pop_data;
    }
}

ThreadPool *create_thread_pool(int threads_num){
    ThreadPool *pool = (ThreadPool *)malloc(sizeof(ThreadPool));
    pool->threads = NULL;
    pool->queue = create_queue();
    pool->lock = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(pool->lock, NULL);
    pool->threads_num = threads_num;
    pool->is_submit_done = 0;
    pool->is_finish = 0;

    return pool;
}

void set_threads_num(ThreadPool* pool, int threads_num){
    pool->threads_num = threads_num;
}
int get_threads_num(ThreadPool* pool){
    return pool->threads_num;
}
int get_num_tasks(ThreadPool* pool){
    return get_q_size(pool->queue);
}
int is_task_queue_empty(ThreadPool* pool){
    return is_q_empty(pool->queue);
}
int is_finish(ThreadPool* pool){
    return pool->is_finish;
}

void submit(void (*func)(void *), void *arg, ThreadPool *pool){
    // func(args);
    if(is_submit_done(pool)){return;} // Block task assignment after calling submit_done
    Task *new_task = (Task*)malloc(sizeof(Task));
    new_task->func = func;
    new_task->arg = arg;

    pthread_mutex_lock(pool->lock);
    push(pool->queue, (void *)new_task);
    pthread_mutex_unlock(pool->lock);
}

void submit_done(ThreadPool* pool){
    pool->is_submit_done = 1;
}

int is_submit_done(ThreadPool* pool){
    return pool->is_submit_done;
}

Task *get_task(ThreadPool* pool){
    return (Task*)(pop(pool->queue));
}

void *worker(void *worker_arg_v){
    WorkerArg *worker_arg = (WorkerArg*)worker_arg_v;
    ThreadPool *pool = worker_arg->pool;
    int thread_id = worker_arg->thread_id;
    // printf("Thread %d Created(Self: %d)\n", thread_id, pthread_self());
    Task *task = NULL;
    // int idle_count = 0;

    for(;(!is_submit_done(pool)) || (!is_task_queue_empty(pool));){
        int is_has_task = 0;

        if(pthread_mutex_trylock(pool->lock) == 0){
            if(!is_task_queue_empty(pool)){
                task = get_task(pool);
                is_has_task = 1;
                // idle_count = 0;
            }
            pthread_mutex_unlock(pool->lock);
        }
        // else{
        //     idle_count++;
        // }
        if(is_has_task){
            task->func(task->arg);
        }

        // if(idle_count > 1000){
        //     printf("Thread %d Idling\n", thread_id);
        // }
    }

    pthread_exit(NULL);
}

// Create and start the threads
void start_pool(ThreadPool* pool){
    pool->threads = (pthread_t*)malloc(sizeof(pthread_t) * get_threads_num(pool));
    WorkerArg *worker_args = (WorkerArg*)malloc(sizeof(WorkerArg) * get_threads_num(pool));
    for(int i = 0; i < get_threads_num(pool); i++){
        worker_args[i].pool = pool;
        worker_args[i].thread_id = i;
        pthread_create(&(pool->threads[i]), NULL, worker, (void*)(&(worker_args[i])));
    }
}

// Join the threads
void end_pool(ThreadPool* pool){
    for(int i = 0; i < get_threads_num(pool); i++){
        pthread_join(pool->threads[i], NULL);
    }
    pool->is_finish = 1;
}

// Reset the task queue, is_finish flag, and is_submit_done flag
void resest_pool(ThreadPool* pool){
    if(pool->is_finish){
        free(pool->queue);
        pool->is_finish = 0;
        pool->is_submit_done = 0;
    }
}

// Free the whole ThreadPool object
void free_pool(ThreadPool* pool){
    if(pool->is_finish){
        free(pool->threads);
        free(pool->queue);
        free(pool->lock);
        free(pool);
    }
}