#ifndef MAXPOOL_FIFO_H
#define MAXPOOL_FIFO_H

#include <systemc.h>
#include <iostream>
#include "TensorPacket.h"

SC_MODULE(MaxPoolFIFO)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    sc_fifo_in<TensorPacket> in_fifo;
    sc_fifo_out<TensorPacket> out_fifo;

    int in_c;
    int in_h;
    int in_w;
    int kernel;
    int stride;
    int out_h;
    int out_w;

    bool pending;
    TensorPacket out_buf;

    SC_HAS_PROCESS(MaxPoolFIFO);

    MaxPoolFIFO(
        sc_module_name name,
        int in_c_,
        int in_h_,
        int in_w_,
        int kernel_,
        int stride_
    )
    : sc_module(name),
      in_c(in_c_),
      in_h(in_h_),
      in_w(in_w_),
      kernel(kernel_),
      stride(stride_),
      out_h((in_h_ - kernel_) / stride_ + 1),
      out_w((in_w_ - kernel_) / stride_ + 1),
      pending(false)
    {
        SC_THREAD(run);
        sensitive << clk.pos();
        dont_initialize();
    }

    int idx_in(int c, int h, int w) const
    {
        return c * (in_h * in_w) + h * in_w + w;
    }

    int idx_out(int c, int h, int w) const
    {
        return c * (out_h * out_w) + h * out_w + w;
    }

    void compute(const TensorPacket& in_pkt, TensorPacket& out_pkt)
    {
        if ((int)in_pkt.data.size() != in_c * in_h * in_w)
        {
            std::cerr << name() << ": input size mismatch in MaxPoolFIFO" << std::endl;
            sc_stop();
            return;
        }

        out_pkt = TensorPacket(in_c, out_h, out_w);

        for (int c = 0; c < in_c; c++)
        {
            for (int oh = 0; oh < out_h; oh++)
            {
                for (int ow = 0; ow < out_w; ow++)
                {
                    int start_h = oh * stride;
                    int start_w = ow * stride;

                    double max_val = in_pkt.data[idx_in(c, start_h, start_w)];

                    for (int kh = 0; kh < kernel; kh++)
                    {
                        for (int kw = 0; kw < kernel; kw++)
                        {
                            int ih = start_h + kh;
                            int iw = start_w + kw;

                            double v = in_pkt.data[idx_in(c, ih, iw)];
                            if (v > max_val)
                            {
                                max_val = v;
                            }
                        }
                    }

                    out_pkt.data[idx_out(c, oh, ow)] = max_val;
                }
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