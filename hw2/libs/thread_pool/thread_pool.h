#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "../queue/queue.h"

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