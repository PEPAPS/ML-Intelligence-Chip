#ifndef MAXPOOL_SIGNAL_H
#define MAXPOOL_SIGNAL_H

#include <systemc.h>
#include <iostream>

using namespace std;

SC_MODULE(MaxPoolSignal)
{
    // control ports
    sc_in<bool> clk;
    sc_in<bool> rst;
    sc_in<bool> in_valid;
    sc_out<bool> out_valid;

    // data ports
    sc_vector< sc_in<double> > data_in;
    sc_vector< sc_out<double> > data_out;

    // parameters
    int in_c;
    int in_h;
    int in_w;
    int kernel;
    int stride;
    int out_h;
    int out_w;
    int in_size;
    int out_size;

    SC_HAS_PROCESS(MaxPoolSignal);

    MaxPoolSignal(
        sc_module_name name,
        int in_c_,
        int in_h_,
        int in_w_,
        int kernel_,
        int stride_
    )
    : sc_module(name),
      data_in("data_in", in_c_ * in_h_ * in_w_),
      data_out(
          "data_out",
          in_c_ * ((in_h_ - kernel_) / stride_ + 1) * ((in_w_ - kernel_) / stride_ + 1)
      )
    {
        in_c = in_c_;
        in_h = in_h_;
        in_w = in_w_;
        kernel = kernel_;
        stride = stride_;

        out_h = (in_h - kernel) / stride + 1;
        out_w = (in_w - kernel) / stride + 1;

        in_size = in_c * in_h * in_w;
        out_size = in_c * out_h * out_w;

        SC_METHOD(run);
        sensitive << clk.pos();
        dont_initialize();
    }

    int idx_in(int c, int h, int w)
    {
        return c * (in_h * in_w) + h * in_w + w;
    }

    int idx_out(int c, int h, int w)
    {
        return c * (out_h * out_w) + h * out_w + w;
    }

    void run()
    {
        if (rst.read() == 1)
        {
            for (int i = 0; i < out_size; i++)
            {
                data_out[i].write(0.0);
            }
            out_valid.write(0);
            return;
        }

        if (in_valid.read() == 1)
        {
            for (int c = 0; c < in_c; c++)
            {
                for (int oh = 0; oh < out_h; oh++)
                {
                    for (int ow = 0; ow < out_w; ow++)
                    {
                        int start_h = oh * stride;
                        int start_w = ow * stride;

                        double max_val = data_in[idx_in(c, start_h, start_w)].read();

                        for (int kh = 0; kh < kernel; kh++)
                        {
                            for (int kw = 0; kw < kernel; kw++)
                            {
                                int ih = start_h + kh;
                                int iw = start_w + kw;

                                double v = data_in[idx_in(c, ih, iw)].read();
                                if (v > max_val)
                                {
                                    max_val = v;
                                }
                            }
                        }

                        data_out[idx_out(c, oh, ow)].write(max_val);
                    }
                }
            }

            out_valid.write(1);
        }
        else
        {
            out_valid.write(0);
        }
    }
};

#endif