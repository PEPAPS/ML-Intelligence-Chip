#define SC_INCLUDE_FX
#include <systemc.h>
#include <vector>
#include <iostream>
using namespace std;

SC_MODULE(FullyConnected)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    // Parameters
    int in_size;
    int out_size;

    // Data
    vector<double> input_feature;
    vector<double> output_feature;
    vector<double> weight;
    vector<double> bias;

    SC_CTOR(FullyConnected)
    {
        SC_METHOD(run);
        sensitive << clk.pos();
    }

    //////////////////////////////////////////////////////
    // CONFIGURE
    //////////////////////////////////////////////////////
    void configure(int in_s, int out_s)
    {
        in_size = in_s;
        out_size = out_s;

        input_feature.resize(in_size);
        output_feature.resize(out_size);
        weight.resize(out_size * in_size);
        bias.resize(out_size);
    }

    //////////////////////////////////////////////////////
    //  FORWARD FUNCTION (IMPORTANT)
    //////////////////////////////////////////////////////
    vector<double> forward(vector<double>& input)
    {
        input_feature = input;

        for(int o=0; o<out_size; o++)
        {
            double sum = bias[o];

            for(int i=0; i<in_size; i++)
            {
                sum += input_feature[i] * weight[o*in_size + i];
            }

            output_feature[o] = sum;
        }

        return output_feature;
    }

    //////////////////////////////////////////////////////
    // SystemC run (not used)
    //////////////////////////////////////////////////////
    void run()
    {
        if(rst.read()) return;
    }
};