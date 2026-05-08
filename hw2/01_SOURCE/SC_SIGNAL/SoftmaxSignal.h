#ifndef SOFTMAX_SIGNAL_H
#define SOFTMAX_SIGNAL_H

#include <systemc.h>
#include <iostream>
#include <cmath>

using namespace std;

SC_MODULE(SoftmaxSignal)
{
    // control ports
    sc_in<bool> clk;
    sc_in<bool> rst;
    sc_in<bool> in_valid;
    sc_out<bool> out_valid;

    // data ports
    sc_vector< sc_in<double> > in_data;
    sc_vector< sc_out<double> > out_data;

    int data_size;

    SC_HAS_PROCESS(SoftmaxSignal);

    SoftmaxSignal(sc_module_name name, int data_size_)
    : sc_module(name),
      in_data("in_data", data_size_),
      out_data("out_data", data_size_)
    {
        data_size = data_size_;

        SC_METHOD(run);
        sensitive << clk.pos();
        dont_initialize();
    }

    void run()
    {
        if (rst.read() == 1)
        {
            for (int i = 0; i < data_size; i++)
            {
                out_data[i].write(0.0);
            }
            out_valid.write(0);
            return;
        }

        if (in_valid.read() == 1)
        {
            double max_val = in_data[0].read();

            for (int i = 1; i < data_size; i++)
            {
                double v = in_data[i].read();
                if (v > max_val)
                {
                    max_val = v;
                }
            }

            double* exp_buf = new double[data_size];
            double sum_exp = 0.0;

            for (int i = 0; i < data_size; i++)
            {
                double shifted = in_data[i].read() - max_val;
                exp_buf[i] = exp(shifted);
                sum_exp += exp_buf[i];
            }

            for (int i = 0; i < data_size; i++)
            {
                double prob = 0.0;
                if (sum_exp != 0.0)
                {
                    prob = exp_buf[i] / sum_exp;
                }
                out_data[i].write(prob);
            }

            delete[] exp_buf;
            out_valid.write(1);
        }
        else
        {
            out_valid.write(0);
        }
    }
};

#endif