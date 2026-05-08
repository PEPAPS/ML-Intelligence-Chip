#define SC_INCLUDE_FX
#include <systemc.h>
#include <vector>
#include <iostream>
using namespace std;

SC_MODULE(Conv2D)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    int in_channels;
    int out_channels;
    int in_h, in_w;
    int kernel;
    int stride;
    int padding;

    vector<double> input_feature;
    vector<double> output_feature;
    vector<double> weight;
    vector<double> bias;

    SC_CTOR(Conv2D)
    {
        SC_METHOD(run);   // you can keep this (not used)
        sensitive << clk.pos();
    }

    //  CONFIG
    void configure(int ic, int oc, int ih, int iw,
                   int k, int s, int p)
    {
        in_channels = ic;
        out_channels = oc;
        in_h = ih;
        in_w = iw;
        kernel = k;
        stride = s;
        padding = p;

        int out_h = (in_h - kernel + 2*padding)/stride + 1;
        int out_w = (in_w - kernel + 2*padding)/stride + 1;

        input_feature.resize(ic * ih * iw);
        output_feature.resize(oc * out_h * out_w);
        weight.resize(oc * ic * kernel * kernel);
        bias.resize(oc);
    }

    //  LOAD WEIGHT
    void load_weights(string w_file, string b_file)
    {
        ifstream wf(w_file), bf(b_file);

        for(int i=0;i<weight.size();i++)
            wf >> weight[i];

        for(int i=0;i<bias.size();i++)
            bf >> bias[i];
    }

    //  FORWARD (THIS IS WHAT YOU ADD)
    vector<double> forward(vector<double>& input)
    {
        input_feature = input;

        int out_h = (in_h - kernel + 2*padding)/stride + 1;
        int out_w = (in_w - kernel + 2*padding)/stride + 1;

        for(int oc=0; oc<out_channels; oc++)
        {
            for(int oy=0; oy<out_h; oy++)
            {
                for(int ox=0; ox<out_w; ox++)
                {
                    double sum = bias[oc];

                    for(int ic=0; ic<in_channels; ic++)
                    {
                        for(int ky=0; ky<kernel; ky++)
                        {
                            for(int kx=0; kx<kernel; kx++)
                            {
                                int iy = oy*stride + ky - padding;
                                int ix = ox*stride + kx - padding;

                                if(iy>=0 && iy<in_h && ix>=0 && ix<in_w)
                                {
                                    sum += input_feature[input_index(ic,iy,ix)]
                                         * weight[weight_index(oc,ic,ky,kx)];
                                }
                            }
                        }
                    }

                    output_feature[output_index(oc,oy,ox,out_h,out_w)] = sum;
                }
            }
        }

        return output_feature;
    }

    // (optional, not used)
    void run()
    {
        if(rst.read()) return;
    }

    // index helpers (keep yours)
    int input_index(int c, int y, int x)
    {
        return c*in_h*in_w + y*in_w + x;
    }

    int weight_index(int oc, int ic, int ky, int kx)
    {
        return oc*in_channels*kernel*kernel
             + ic*kernel*kernel
             + ky*kernel
             + kx;
    }

    int output_index(int oc, int y, int x, int out_h, int out_w)
    {
        return oc*out_h*out_w + y*out_w + x;
    }
};