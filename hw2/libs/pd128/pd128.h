#ifndef A_H
#define A_H

#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>

class PD128{
private:
    __m128d x;
    const int vec_scale = 1;
    const int vec_gap = 2;
    const int vec_ele_size = 64;

public:
    PD128();
    PD128(double *);
    PD128(const double *);
    PD128(__m128d);
    // PD128(const __m128d);
    ~PD128();

    __m128d m128d() const;
    void print();
    void print_int();
    void load(double *);
    void store(double *);
    int equal(const PD128);
    

    PD128 operator+(const PD128);
    // PD128 operator-(const PD128&);
    // PD128 operator*(const PD128&);
    // PD128 operator/(const PD128&);
    // PD128 operator<(const PD128&);
    PD128 operator==(const PD128);
};

#endif