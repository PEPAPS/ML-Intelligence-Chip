#ifndef FULLYCONNECTED_SIGNAL_H
#define FULLYCONNECTED_SIGNAL_H

#include <systemc.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

SC_MODULE(FullyConnectedSignal)
{
    // control
    sc_in<bool> clk;
    sc_in<bool> rst;
    sc_in<bool> in_valid;
    sc_out<bool> out_valid;

    // data ports
    sc_vector< sc_in<double> > in_data;
    sc_vector< sc_out<double> > out_data;

    // dimensions
    int in_size;
    int out_size;

    // parameters
    double* weight; // [out_size][in_size]
    double* bias;   // [out_size]

    SC_HAS_PROCESS(FullyConnectedSignal);

    FullyConnectedSignal(
        sc_module_name name,
        int in_size_,
        int out_size_,
        const string& weight_file,
        const string& bias_file
    )
    : sc_module(name),
      in_data("in_data", in_size_),
      out_data("out_data", out_size_)
    {
        in_size = in_size_;
        out_size = out_size_;

        weight = new double[out_size * in_size];
        bias   = new double[out_size];

        load_parameters(weight_file, bias_file);

        SC_METHOD(run);
        sensitive << clk.pos();
        dont_initialize();
    }

    ~FullyConnectedSignal()
    {
        delete[] weight;
        delete[] bias;
    }

    int idx_weight(int o, int i)
    {
        return o * in_size + i;
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

        for (int o = 0; o < out_size; o++)
        {
            for (int i = 0; i < in_size; i++)
            {
                wf >> weight[idx_weight(o, i)];
            }
        }

        for (int o = 0; o < out_size; o++)
        {
            bf >> bias[o];
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
                out_data[i].write(0.0);
            }

            out_valid.write(0);
            return;
        }

        if (in_valid.read() == 1)
        {
            for (int o = 0; o < out_size; o++)
            {
                double sum = bias[o];

                for (int i = 0; i < in_size; i++)
                {
                    sum += in_data[i].read() * weight[idx_weight(o, i)];
                }

                out_data[o].write(sum);
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