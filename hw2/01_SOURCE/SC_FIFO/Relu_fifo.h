#ifndef RELU_FIFO_H
#define RELU_FIFO_H

#include <systemc.h>
#include <iostream>
#include "TensorPacket.h"

SC_MODULE(ReLUFIFO)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    sc_fifo_in<TensorPacket> in_fifo;
    sc_fifo_out<TensorPacket> out_fifo;

    int data_size;
    bool pending;
    TensorPacket out_buf;

    SC_HAS_PROCESS(ReLUFIFO);

    ReLUFIFO(sc_module_name name, int data_size_)
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
            std::cerr << name() << ": input size mismatch in ReLUFIFO" << std::endl;
            sc_stop();
            return;
        }

        out_pkt = in_pkt;

        for (int i = 0; i < data_size; i++)
        {
            if (out_pkt.data[i] < 0.0)
            {
                out_pkt.data[i] = 0.0;
            }
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