#define SC_INCLUDE_FX
#include <systemc.h>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

SC_MODULE(MaxPooling)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    int channels;
    int in_h;
    int in_w;
    int kernel;
    int stride;

    int out_h;
    int out_w;

    vector<double> input_feature;
    vector<double> output_feature;

    SC_CTOR(MaxPooling)
    {
        SC_METHOD(run);
        sensitive << clk.pos();
    }

    //////////////////////////////////////////////////////
    // CONFIGURE
    //////////////////////////////////////////////////////
    void configure(int c, int h, int w, int k, int s)
    {
        channels = c;
        in_h = h;
        in_w = w;
        kernel = k;
        stride = s;

        out_h = (in_h - kernel)/stride + 1;
        out_w = (in_w - kernel)/stride + 1;

        input_feature.resize(c * in_h * in_w);
        output_feature.resize(c * out_h * out_w);
    }

    int input_index(int c, int y, int x)
    {
        return c*in_h*in_w + y*in_w + x;
    }

    int output_index(int c, int y, int x)
    {
        return c*out_h*out_w + y*out_w + x;
    }

    //////////////////////////////////////////////////////
    //  FORWARD FUNCTION 
    //////////////////////////////////////////////////////
    vector<double> forward(vector<double>& input)
    {
        input_feature = input;

        for(int c=0; c<channels; c++)
        {
            for(int oy=0; oy<out_h; oy++)
            {
                for(int ox=0; ox<out_w; ox++)
                {
                    double max_val = input_feature[input_index(c, oy*stride, ox*stride)];

                    for(int ky=0; ky<kernel; ky++)
                    {
                        for(int kx=0; kx<kernel; kx++)
                        {
                            int iy = oy*stride + ky;
                            int ix = ox*stride + kx;

                            double val = input_feature[input_index(c,iy,ix)];
                            max_val = max(max_val, val);
                        }
                    }

                    output_feature[output_index(c,oy,ox)] = max_val;
                }
            }
        }

        return output_feature;
    }

    //////////////////////////////////////////////////////
    // (optional SystemC run)
    //////////////////////////////////////////////////////
    void run()
    {
        if(rst.read()) return;
    }
};