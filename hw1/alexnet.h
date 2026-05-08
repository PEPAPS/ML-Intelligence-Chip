#ifndef ALEXNET_H
#define ALEXNET_H

#define SC_INCLUDE_FX
#include <systemc.h>
#include <vector>
#include <fstream>
#include <iostream>
#include "conv2d.h"
#include "relu.h"
#include "maxpool.h"
#include "fc_layer.h"
#include "Pattern.H"
using namespace std;

SC_MODULE(AlexNet)
{
    sc_in_clk clk;
    sc_in<bool> rst;
    sc_in<bool> in_valid;

    sc_out<bool> out_valid;
    // INPUT IMAGE FROM PATTERN
    sc_vector<sc_in<double>> img{"img", 150528};

    // OUTPUT TO PATTERN
    sc_vector<sc_out<double>> output_softmax{"output_softmax", 1000};
    sc_vector<sc_out<double>> output_linear{"output_linear", 1000};
    // Buffers
    vector<double> image;
    vector<double> padded_image;
    Conv2D conv1, conv2, conv3, conv4, conv5;
    ReLU relu1, relu2, relu3, relu4, relu5, relu6, relu7;
    MaxPooling pool1, pool2, pool3;
    FullyConnected fc6, fc7, fc8;


   SC_CTOR(AlexNet) : 
        conv1("conv1"),                     
        relu1("relu1"),
        pool1("pool1"),
        conv2("conv2"),
        relu2("relu2"),
        pool2("pool2"),
        conv3("conv3"),
        relu3("relu3"),
        conv4("conv4"),
        relu4("relu4"),
        conv5("conv5"),
        relu5("relu5"),
        pool3("pool3"),
        fc6("fc6"),
        relu6("relu6"),
        fc7("fc7"),
        relu7("relu7"),
        fc8("fc8")
        
    {
    SC_METHOD(run);
    sensitive << clk.pos();
    }

    // Function declarations ONLY
    void load_image(string filename);
    void run();
};

#endif