#ifndef RELU_SIGNAL_H
#define RELU_SIGNAL_H

#include <systemc.h>
#include <iostream>

using namespace std;

SC_MODULE(ReLUSignal)
{
    sc_in<bool> clk;
    sc_in<bool> rst;
    sc_in<bool> in_valid;
    sc_out<bool> out_valid;

    sc_vector< sc_in<double> > data_in;
    sc_vector< sc_out<double> > data_out;

    int data_size;

    SC_HAS_PROCESS(ReLUSignal);

    ReLUSignal(sc_module_name name, int data_size_)
    : sc_module(name),
      data_in("data_in", data_size_),
      data_out("data_out", data_size_)
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
                data_out[i].write(0.0);
            }
            out_valid.write(0);
            return;
        }

        if (in_valid.read() == 1)
        {
            for (int i = 0; i < data_size; i++)
            {
                double v = data_in[i].read();

                if (v < 0.0)
                    v = 0.0;

                data_out[i].write(v);
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