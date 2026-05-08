#ifndef CONV2D_FIFO_H
#define CONV2D_FIFO_H

#include <systemc.h>
#include <iostream>
#include <fstream>
#include <string>
#include "TensorPacket.h"

SC_MODULE(Conv2DFIFO)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    sc_fifo_in<TensorPacket> in_fifo;
    sc_fifo_out<TensorPacket> out_fifo;

    int in_c;
    int in_h;
    int in_w;
    int out_c;
    int kernel;
    int stride;
    int padding;

    int out_h;
    int out_w;

    double* weight;
    double* bias;

    bool pending;
    TensorPacket out_buf;

    SC_HAS_PROCESS(Conv2DFIFO);

    Conv2DFIFO(
        sc_module_name name,
        int in_c_,
        int in_h_,
        int in_w_,
        int out_c_,
        int kernel_,
        int stride_,
        int padding_,
        const std::string& weight_file,
        const std::string& bias_file
    )
    : sc_module(name),
      in_c(in_c_),
      in_h(in_h_),
      in_w(in_w_),
      out_c(out_c_),
      kernel(kernel_),
      stride(stride_),
      padding(padding_),
      out_h(((in_h_ - kernel_ + 2 * padding_) / stride_) + 1),
      out_w(((in_w_ - kernel_ + 2 * padding_) / stride_) + 1),
      pending(false)
    {
        weight = new double[out_c * in_c * kernel * kernel];
        bias   = new double[out_c];

        load_parameters(weight_file, bias_file);

        SC_THREAD(run);
        sensitive << clk.pos();
        dont_initialize();
    }

    ~Conv2DFIFO()
    {
        delete[] weight;
        delete[] bias;
    }

    int idx_in(int c, int h, int w) const
    {
        return c * (in_h * in_w) + h * in_w + w;
    }

    int idx_out(int c, int h, int w) const
    {
        return c * (out_h * out_w) + h * out_w + w;
    }

    int idx_weight(int oc, int ic, int kh, int kw) const
    {
        return ((oc * in_c + ic) * kernel + kh) * kernel + kw;
    }

    void load_parameters(const std::string& weight_file, const std::string& bias_file)
    {
        std::ifstream wf(weight_file.c_str());
        std::ifstream bf(bias_file.c_str());

        if (!wf.is_open())
        {
            std::cerr << "Error opening weight file: " << weight_file << std::endl;
            exit(1);
        }

        if (!bf.is_open())
        {
            std::cerr << "Error opening bias file: " << bias_file << std::endl;
            exit(1);
        }

        for (int oc = 0; oc < out_c; oc++)
        {
            for (int ic = 0; ic < in_c; ic++)
            {
                for (int kh = 0; kh < kernel; kh++)
                {
                    for (int kw = 0; kw < kernel; kw++)
                    {
                        wf >> weight[idx_weight(oc, ic, kh, kw)];
                    }
                }
            }
        }

        for (int oc = 0; oc < out_c; oc++)
        {
            bf >> bias[oc];
        }

        wf.close();
        bf.close();
    }

    void compute(const TensorPacket& in_pkt, TensorPacket& out_pkt)
    {
        if ((int)in_pkt.data.size() != in_c * in_h * in_w)
        {
            std::cerr << name() << ": input size mismatch in Conv2DFIFO" << std::endl;
            sc_stop();
            return;
        }

        out_pkt = TensorPacket(out_c, out_h, out_w);

        for (int oc = 0; oc < out_c; oc++)
        {
            for (int oh = 0; oh < out_h; oh++)
            {
                for (int ow = 0; ow < out_w; ow++)
                {
                    double sum = bias[oc];

                    for (int ic = 0; ic < in_c; ic++)
                    {
                        for (int kh = 0; kh < kernel; kh++)
                        {
                            for (int kw = 0; kw < kernel; kw++)
                            {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                double in_val = 0.0;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                                {
                                    in_val = in_pkt.data[idx_in(ic, ih, iw)];
                                }

                                sum += in_val * weight[idx_weight(oc, ic, kh, kw)];
                            }
                        }
                    }

                    out_pkt.data[idx_out(oc, oh, ow)] = sum;
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