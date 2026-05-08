#define SC_INCLUDE_FX
#include <systemc.h>
#include <vector>
#include <iostream>
using namespace std;

SC_MODULE(ReLU)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    int channels;
    int height;
    int width;

    vector<double> input_feature;
    vector<double> output_feature;

    SC_CTOR(ReLU)
    {
        SC_METHOD(run);
        sensitive << clk.pos();
    }

    // CONFIGURE
    void configure(int c, int h, int w)
    {
        channels = c;
        height = h;
        width = w;

        input_feature.resize(c*h*w);
        output_feature.resize(c*h*w);
    }

    int index(int c, int y, int x)
    {
        return c*height*width + y*width + x;
    }

    //////////////////////////////////////////////////////
    //  FORWARD FUNCTION 
    //////////////////////////////////////////////////////
    vector<double> forward(vector<double>& input)
    {
        input_feature = input;

        for(int i=0;i<input_feature.size();i++)
        {
            if(input_feature[i] < 0)
                output_feature[i] = 0;
            else
                output_feature[i] = input_feature[i];
        }

        return output_feature;
    }

    
    void run()
    {
        if(rst.read()) return;
        
    }
};