#ifndef SOFTMAX_FIFO_H
#define SOFTMAX_FIFO_H

#include <systemc.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "TensorPacket.h"

SC_MODULE(SoftmaxFIFO)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    sc_fifo_in<TensorPacket> in_fifo;
    sc_fifo_out<TensorPacket> out_fifo;

    int data_size;
    bool pending;
    TensorPacket out_buf;

    SC_HAS_PROCESS(SoftmaxFIFO);

    SoftmaxFIFO(sc_module_name name, int data_size_)
    : sc_module(name),
      data_size(data_size_),
      pending(false)
    {
        SC_THREAD(run);
        sensitive << clk.pos();
        dont_initialize();
    }

    void compute(const TensorPacket& in_pkt, TensorPacket& out_pkt)
    {
        if ((int)in_pkt.data.size() != data_size)
        {
            std::cerr << name() << ": input size mismatch in SoftmaxFIFO" << std::endl;
            sc_stop();
            return;
        }

        out_pkt = TensorPacket(1, 1, data_size);

        double max_val = in_pkt.data[0];
        for (int i = 1; i < data_size; i++)
        {
            if (in_pkt.data[i] > max_val)
            {
                max_val = in_pkt.data[i];
            }
        }

        std::vector<double> exp_buf(data_size, 0.0);
        double sum_exp = 0.0;

        for (int i = 0; i < data_size; i++)
        {
            double shifted = in_pkt.data[i] - max_val;
            exp_buf[i] = std::exp(shifted);
            sum_exp += exp_buf[i];
        }

        for (int i = 0; i < data_size; i++)
        {
            out_pkt.data[i] = (sum_exp == 0.0) ? 0.0 : (exp_buf[i] / sum_exp);
        }
    }

    void run()
    {
        while (true)
        {
            wait();

            if (rst.read())
            {
                pending = false;
                out_buf = TensorPacket();
                continue;
            }

            if (!pending)
            {
                TensorPacket in_pkt;
                if (in_fifo.nb_read(in_pkt))
                {
                    compute(in_pkt, out_buf);
                    pending = true;
                }
            }

            if (pending)
            {
                if (out_fifo.nb_write(out_buf))
                {
                    pending = false;
                }
            }
        }
    }
};

#endif