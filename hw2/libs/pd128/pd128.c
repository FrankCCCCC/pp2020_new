#include "pd128.h"

PD128::PD128(){}
PD128::PD128(double *v){this->x = _mm_loadu_pd(v);}
PD128::PD128(const double *v){this->x = _mm_loadu_pd(v);}
PD128::PD128(__m128d v){this->x = v;}
// PD128::PD128(const __m128d v){this->x = v;}
PD128::~PD128(){}


__m128d PD128::m128d()const{return this->x;}
void PD128::print(){
    for(int i = 0; i < this->vec_gap; i++){
        printf("%lf ", this->x[i]);
    }
    printf("\n");
}
void PD128::print_int(){
    for(int i = 0; i < this->vec_gap; i++){
        printf("%lu ", (unsigned long int)(this->x[i]));
    }
    printf("\n");
}
void PD128::load(double *v){this->x = _mm_loadu_pd(v);}
void PD128::store(double *v){_mm_store_pd(v, this->x);}
int PD128::equal(const PD128 a){const __m128d equv = _mm_cmpeq_pd(this->x, a.m128d()); return 1;}

PD128 PD128::operator+(const PD128 a){
    return PD128(_mm_add_pd(this->x, a.m128d()));
}
PD128 PD128::operator==(const PD128 a){
    __m128d res = _mm_cmpeq_pd(this->x, a.m128d());
    res = (__m128d)_mm_slli_epi64((__m128i)res, 54);
    res = (__m128d)_mm_srli_epi64((__m128i)res, 2);

    return PD128(res);
}