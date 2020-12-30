// Doubly Linked List Queue
#include "queue.h"

// While initializztion, head and tail points to dummy node.
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