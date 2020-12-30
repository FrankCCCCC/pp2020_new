// Doubly Linked List Queue
#include <stdio.h>
#include <stdlib.h>

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