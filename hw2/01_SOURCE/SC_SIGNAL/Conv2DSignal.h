#ifndef CONV2D_SIGNAL_H
#define CONV2D_SIGNAL_H

#include <systemc.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

SC_MODULE(Conv2DSignal)
{
    sc_in<bool> clk;
    sc_in<bool> rst;
    sc_in<bool> in_valid;
    sc_out<bool> out_valid;

    sc_vector< sc_in<double> > img_in;
    sc_vector< sc_out<double> > img_out;

    int in_c;
    int in_h;
    int in_w;
    int out_c;
    int kernel;
    int stride;
    int padding;

    int out_h;
    int out_w;
    int in_size;
    int out_size;

    double* weight;
    double* bias;

    SC_HAS_PROCESS(Conv2DSignal);

    Conv2DSignal(
        sc_module_name name,
        int in_c_,
        int in_h_,
        int in_w_,
        int out_c_,
        int kernel_,
        int stride_,
        int padding_,
        const string& weight_file,
        const string& bias_file
    )
    : sc_module(name),
      img_in("img_in", in_c_ * in_h_ * in_w_),
      img_out(
          "img_out",
          out_c_ *
          (((in_h_ - kernel_ + 2 * padding_) / stride_) + 1) *
          (((in_w_ - kernel_ + 2 * padding_) / stride_) + 1)
      )
    {
        in_c = in_c_;
        in_h = in_h_;
        in_w = in_w_;
        out_c = out_c_;
        kernel = kernel_;
        stride = stride_;
        padding = padding_;

        out_h = ((in_h - kernel + 2 * padding) / stride) + 1;
        out_w = ((in_w - kernel + 2 * padding) / stride) + 1;

        in_size = in_c * in_h * in_w;
        out_size = out_c * out_h * out_w;

        weight = new double[out_c * in_c * kernel * kernel];
        bias   = new double[out_c];

        load_parameters(weight_file, bias_file);

        SC_METHOD(run);
        sensitive << clk.pos();
        dont_initialize();
    }

    ~Conv2DSignal()
    {
        delete[] weight;
        delete[] bias;
    }

    int idx_in(int c, int h, int w)
    {
        return c * (in_h * in_w) + h * in_w + w;
    }

    int idx_out(int c, int h, int w)
    {
        return c * (out_h * out_w) + h * out_w + w;
    }

    int idx_weight(int oc, int ic, int kh, int kw)
    {
        return ((oc * in_c + ic) * kernel + kh) * kernel + kw;
    }

    void load_parameters(const string& weight_file, const string& bias_file)
    {
        ifstream wf(weight_file.c_str());
        ifstream bf(bias_file.c_str());

        if (!wf.is_open())
        {
            cerr << "Error opening weight file: " << weight_file << endl;
            exit(1);
        }

        if (!bf.is_open())
        {
            cerr << "Error opening bias file: " << bias_file << endl;
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

    void run()
    {
        if (rst.read() == 1)
        {
            for (int i = 0; i < out_size; i++)
            {
                img_out[i].write(0.0);
            }
            out_valid.write(0);
            return;
        }

        if (in_valid.read() == 1)
        {
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
                                        in_val = img_in[idx_in(ic, ih, iw)].read();
                                    }

                                    double w_val = weight[idx_weight(oc, ic, kh, kw)];
                                    sum += in_val * w_val;
                                }
                            }
                        }

                        img_out[idx_out(oc, oh, ow)].write(sum);
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