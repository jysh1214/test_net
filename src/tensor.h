#ifndef TENSOR_H
#define TENSOR_H

#include "gemm.h"
#include "print.h"

#include <assert.h>
#include <stdlib.h>
#include <iostream>

/**
 * NOTE: matrix(row, col) = tensor(row, col, 1)
*/
struct tensor
{
    tensor(size_t row, size_t col, size_t channel) {
        assert(row != 0 && col != 0 && channel != 0);
        this->row = row;
        this->col = col;
        this->channel = channel;
        data = (float *)new float[row * col * channel];
        for (size_t i = 0; i < (row * col * channel); ++i) {
            data[i] = 0.0;
        }
    }
    virtual ~tensor() {
        delete[] data;
    }

    size_t row;
    size_t col;
    size_t channel;
    float *data;
};

static tensor *matrixAdd(tensor *a, tensor *b)
{
    assert(a->row == b->row);
    assert(a->col == b->col);
    assert(a->channel == b->channel);

    tensor *c = new tensor(a->row, a->col, a->channel);

    for (size_t k = 0; k < a->channel; ++k) {
        for (size_t i = 0; i < a->row; ++i) {
            for (size_t j = 0; j < a->col; ++j) {
                c->data[k * (a->row) * (a->col) + i * (a->col) + j] =
                    a->data[k * (a->row) * (a->col) + i * (a->col) + j] + b->data[k * (a->row) * (a->col) + i * (a->col) + j];
            }
        }
    }

    return c;
}

/**
 * matrixMultiplication - matrix a * matrix b = matrix c
 * @a: matrix
 * @TA: transpose matrix a
 * @b: matrix
 * @TB: transpose matrix b
*/
static tensor *matrixMul(tensor *a, int TA, tensor *b, int TB)
{
    assert(a->channel == 1 && b->channel == 1);
    tensor *c = new tensor(a->row, b->col, 1);

    gemm(TA, TB, a->row, b->col, a->col, 1, a->data, a->col, b->data, b->col, 1, c->data, c->col);

    return c;
}

#endif
