#ifndef FULLYCONNECTED_FIFO_H
#define FULLYCONNECTED_FIFO_H

#include <systemc.h>
#include <iostream>
#include <fstream>
#include <string>
#include "TensorPacket.h"

SC_MODULE(FullyConnectedFIFO)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    sc_fifo_in<TensorPacket> in_fifo;
    sc_fifo_out<TensorPacket> out_fifo;

    int in_size;
    int out_size;

    double* weight;
    double* bias;

    bool pending;
    TensorPacket out_buf;

    SC_HAS_PROCESS(FullyConnectedFIFO);

    FullyConnectedFIFO(
        sc_module_name name,
        int in_size_,
        int out_size_,
        const std::string& weight_file,
        const std::string& bias_file
    )
    : sc_module(name),
      in_size(in_size_),
      out_size(out_size_),
      pending(false)
    {
        weight = new double[out_size * in_size];
        bias   = new double[out_size];

        load_parameters(weight_file, bias_file);

        SC_THREAD(run);
        sensitive << clk.pos();
        dont_initialize();
    }

    ~FullyConnectedFIFO()
    {
        delete[] weight;
        delete[] bias;
    }

    int idx_weight(int o, int i) const
    {
        return o * in_size + i;
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

    void compute(const TensorPacket& in_pkt, TensorPacket& out_pkt)
    {
        if ((int)in_pkt.data.size() != in_size)
        {
            std::cerr << name() << ": input size mismatch in FullyConnectedFIFO" << std::endl;
            sc_stop();
            return;
        }

        out_pkt = TensorPacket(1, 1, out_size);

        for (int o = 0; o < out_size; o++)
        {
            double sum = bias[o];

            for (int i = 0; i < in_size; i++)
            {
                sum += in_pkt.data[i] * weight[idx_weight(o, i)];
            }

            out_pkt.data[o] = sum;
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