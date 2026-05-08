#ifndef PADDING_FIFO_H
#define PADDING_FIFO_H

#include <systemc.h>
#include <iostream>
#include "TensorPacket.h"

SC_MODULE(PaddingFIFO)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    sc_fifo_in<TensorPacket> in_fifo;
    sc_fifo_out<TensorPacket> out_fifo;

    int in_c;
    int in_h;
    int in_w;

    int pad_top;
    int pad_left;
    int pad_bottom;
    int pad_right;

    int out_h;
    int out_w;

    bool pending;
    TensorPacket out_buf;

    SC_HAS_PROCESS(PaddingFIFO);

    PaddingFIFO(
        sc_module_name name,
        int in_c_ = 3,
        int in_h_ = 224,
        int in_w_ = 224,
        int pad_top_ = 2,
        int pad_left_ = 2,
        int pad_bottom_ = 1,
        int pad_right_ = 1
    )
    : sc_module(name),
      in_c(in_c_),
      in_h(in_h_),
      in_w(in_w_),
      pad_top(pad_top_),
      pad_left(pad_left_),
      pad_bottom(pad_bottom_),
      pad_right(pad_right_),
      out_h(in_h_ + pad_top_ + pad_bottom_),
      out_w(in_w_ + pad_left_ + pad_right_),
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
            std::cerr << name() << ": input size mismatch in PaddingFIFO" << std::endl;
            sc_stop();
            return;
        }

        out_pkt = TensorPacket(in_c, out_h, out_w);

        for (int c = 0; c < in_c; c++)
        {
            for (int h = 0; h < in_h; h++)
            {
                for (int w = 0; w < in_w; w++)
                {
                    int oh = h + pad_top;
                    int ow = w + pad_left;
                    out_pkt.data[idx_out(c, oh, ow)] = in_pkt.data[idx_in(c, h, w)];
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