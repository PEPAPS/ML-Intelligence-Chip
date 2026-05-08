#ifndef TENSOR_PACKET_H
#define TENSOR_PACKET_H

#include <vector>
#include <iostream>

struct TensorPacket
{
    int c;
    int h;
    int w;
    std::vector<double> data;

    TensorPacket() : c(0), h(0), w(0) {}

    TensorPacket(int c_, int h_, int w_)
        : c(c_), h(h_), w(w_), data(c_ * h_ * w_, 0.0) {}
};

inline std::ostream& operator<<(std::ostream& os, const TensorPacket& t)
{
    os << "TensorPacket(" << t.c << "x" << t.h << "x" << t.w << ")";
    return os;
}

#endif