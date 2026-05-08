#define SC_INCLUDE_FX
#include <systemc.h>
#include <iostream>
#include "alexnet.h"
#include <cmath>
#include <algorithm>

using namespace std;
//////////////////////////////////////////////////////
// softmax FUNCTION
//////////////////////////////////////////////////////

vector<double> softmax(const vector<double>& x)
{
    vector<double> y(x.size());

    double max_val = *max_element(x.begin(), x.end());

    double sum = 0.0;

    for(int i = 0; i < x.size(); i++)
    {
        y[i] = exp(x[i] - max_val);
        sum += y[i];
    }

    for(int i = 0; i < x.size(); i++)
    {
        y[i] /= sum;
    }

    return y;
}
void AlexNet::run()
{
    if(rst.read())
    {
        out_valid.write(0);
        return;
    }

    if(in_valid.read())
    {
        vector<double> raw_img(3 * 224 * 224);
        for(int i = 0; i < 3 * 224 * 224; i++)
        {
            raw_img[i] = img[i].read();
        }
        int in_c = 3;
        int in_h = 224;
        int in_w = 224;

        int pad_top = 2;
        int pad_left = 2;
        int pad_bottom = 1;
        int pad_right = 1;

        int out_h = in_h + pad_top + pad_bottom; // 227
        int out_w = in_w + pad_left + pad_right; // 227

        padded_image.assign(in_c * out_h * out_w, 0);

        for(int c = 0; c < in_c; c++)
        {
            for(int y = 0; y < in_h; y++)
            {
                for(int x = 0; x < in_w; x++)
                {
                    int in_idx  = c*in_h*in_w + y*in_w + x;
                    int out_idx = c*out_h*out_w + (y + pad_top)*out_w + (x + pad_left);

                    padded_image[out_idx] = raw_img[in_idx];
                }
            }
        }
        // // AFTER padding
        // vector<double> x(3 * 224 * 224);

        // // read from Pattern ports
        // for(int i = 0; i < 3*224*224; i++)
        // {
        //     x[i] = img[i].read();
        // }
        // Conv1
        vector<double> x = conv1.forward(padded_image);

        // ReLU
        x = relu1.forward(x);

        // MaxPool1
        x = pool1.forward(x);

        // Conv2  
        x = conv2.forward(x);

        // ReLU2  
        x = relu2.forward(x);

        // MaxPool2
        x = pool2.forward(x);

        // Conv3  
        x = conv3.forward(x);

        // ReLU3  
        x = relu3.forward(x);

        // Conv4  
        x = conv4.forward(x);

        // ReLU4  
        x = relu4.forward(x);
        
        // Conv5  
        x = conv5.forward(x);
        // ReLU5  
        x = relu5.forward(x);

        // MaxPool3
        x = pool3.forward(x);

        // FC6
        x = fc6.forward(x);

        // ReLU6
        x = relu6.forward(x);

        // FC7
        x = fc7.forward(x);

        // ReLU7
        x = relu7.forward(x);

        // FC8
        x = fc8.forward(x);

        vector<double> prob = softmax(x);
        // send raw FC8 output
        for(int i = 0; i < 1000; i++)
        {
            output_linear[i].write(x[i]);
        }

        // send softmax output
        for(int i = 0; i < 1000; i++)
        {
            output_softmax[i].write(prob[i]);
        }

        out_valid.write(1);  // notify Pattern        
    }
    else
    {
        out_valid.write(0);
    }
}

//////////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////////
int sc_main(int argc, char *argv[])
{
    AlexNet model("alexnet");
    sc_clock clk("clk", 1, SC_NS);
    sc_signal<bool> rst, in_valid, out_valid;
    Pattern pattern("pattern", argv[1]);

    // sc_signal<double> img_signals[150528];
    // sc_signal<double> softmax_sigs[1000];
    // sc_signal<double> linear_sigs[1000];
    // 1. Heap Allocation for large signal arrays
    auto img_signals = new sc_signal<double>[150528];
    auto softmax_sigs = new sc_signal<double>[1000];
    auto linear_sigs = new sc_signal<double>[1000];

    // 2. Control Signal Binding
    pattern.clock(clk);         // DON'T FORGET THIS
    pattern.rst(rst);
    pattern.in_valid(in_valid);
    pattern.out_valid(out_valid);


    // 3. Data Signal Binding (Bus)
    for(int i = 0; i < 150528; i++) {
        pattern.img[i](img_signals[i]); 
        model.img[i](img_signals[i]);
    }
    
    for(int i = 0; i < 1000; i++) {
        model.output_softmax[i](softmax_sigs[i]);
        pattern.output_softmax[i](softmax_sigs[i]);
        model.output_linear[i](linear_sigs[i]);
        pattern.output_linear[i](linear_sigs[i]);
    }
    //  Configure 
    model.conv1.configure
    (
    3,    // in_channels
    64,   // out_channels (your assignment)
    227,  // input height (after padding)
    227,  // input width
    11,   // kernel
    4,    // stride
    0     // padding (already padded before)
    );
    model.relu1.configure
    (
    64,  // channels
    55,  // height
    55   // width
    );
    model.pool1.configure(
    64,  // channels
    55,  // input height
    55,  // input width
    3,   // kernel
    2    // stride
    );
    model.conv2.configure(
    64,   // input channels
    192,  // output channels
    27,   // height
    27,   // width
    5,    // kernel
    1,    // stride
    2     // padding
    );
    model.relu2.configure(
    192,
    27,
    27
    );
    model.pool2.configure(
    192,  // channels
    27,  // input height
    27,  // input width
    3,   // kernel
    2    // stride
    );
    model.conv3.configure(
    192,   // input channels
    384,  // output channels
    13,   // height
    13,   // width
    3,    // kernel
    1,    // stride
    1     // padding
    );
    model.relu3.configure(
    384,
    13,
    13
    );
    model.conv4.configure(
    384,   // input channels
    256,  // output channels
    13,   // height
    13,   // width
    3,    // kernel
    1,    // stride
    1     // padding
    );
    model.relu4.configure(
    256,
    13,
    13
    );
    model.conv5.configure(
    256,   // input channels
    256,  // output channels
    13,   // height
    13,   // width
    3,    // kernel
    1,    // stride
    1     // padding
    );
    model.relu5.configure(
    256,
    13,
    13
    );
    model.pool3.configure(
    256,  // channels
    13,  // input height
    13,  // input width
    3,   // kernel
    2    // stride
    );
    // FC6
    model.fc6.configure(256*6*6, 4096);
    model.relu6.configure(1,1,4096);
    // FC7
    model.fc7.configure(4096, 4096);
    model.relu7.configure(1,1,4096);
    // FC8
    model.fc8.configure(4096, 1000);
    // Load weights
    model.conv1.load_weights(
        "/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/conv1_weight.txt",
        "/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/conv1_bias.txt");
    model.conv2.load_weights(
        "/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/conv2_weight.txt",
        "/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/conv2_bias.txt");
    model.conv3.load_weights(
        "/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/conv3_weight.txt",
        "/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/conv3_bias.txt");
    model.conv4.load_weights(
        "/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/conv4_weight.txt",
        "/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/conv4_bias.txt");
    model.conv5.load_weights(
        "/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/conv5_weight.txt",
        "/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/conv5_bias.txt");

    ifstream w6("/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/fc6_weight.txt");
    for(int i=0;i<model.fc6.weight.size();i++)
        w6 >> model.fc6.weight[i];
    ifstream b6("/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/fc6_bias.txt");
    for(int i=0;i<model.fc6.bias.size();i++)
        b6 >> model.fc6.bias[i];

    ifstream w7("/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/fc7_weight.txt");
    for(int i=0;i<model.fc7.weight.size();i++)
        w7 >> model.fc7.weight[i];
    ifstream b7("/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/fc7_bias.txt");
    for(int i=0;i<model.fc7.bias.size();i++)
        b7 >> model.fc7.bias[i];

    ifstream w8("/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/fc8_weight.txt");
    for(int i=0;i<model.fc8.weight.size();i++)
        w8 >> model.fc8.weight[i];
    ifstream b8("/RAID2/COURSE/2026_Spring/es26mlchip/es26mlchip081/mliccourse/hw1/data/fc8_bias.txt");
    for(int i=0;i<model.fc8.bias.size();i++)
        b8 >> model.fc8.bias[i];

    /////////
    /////////
    ////////
    model.clk(clk);
    model.rst(rst);
    model.in_valid(in_valid);
    model.out_valid(out_valid);
    model.conv1.clk(clk);
    model.conv1.rst(rst);
    model.relu1.clk(clk);
    model.relu1.rst(rst);
    model.pool1.clk(clk);
    model.pool1.rst(rst);
    model.conv2.clk(clk);
    model.conv2.rst(rst);
    model.relu2.clk(clk);
    model.relu2.rst(rst);
    model.pool2.clk(clk);
    model.pool2.rst(rst);
    model.conv3.clk(clk);
    model.conv3.rst(rst);
    model.relu3.clk(clk);
    model.relu3.rst(rst);
    model.conv4.clk(clk);
    model.conv4.rst(rst);
    model.relu4.clk(clk);
    model.relu4.rst(rst);
    model.conv5.clk(clk);
    model.conv5.rst(rst);
    model.relu5.clk(clk);
    model.relu5.rst(rst);
    model.pool3.clk(clk);
    model.pool3.rst(rst);
    model.fc6.clk(clk);
    model.fc6.rst(rst);
    model.relu6.clk(clk);
    model.relu6.rst(rst);
    model.fc7.clk(clk);
    model.fc7.rst(rst);
    model.relu7.clk(clk);
    model.relu7.rst(rst);
    model.fc8.clk(clk);
    model.fc8.rst(rst);
    
    sc_start(100, SC_NS); 

    
    // // Simulation
    // rst.write(1);
    // in_valid.write(0);
    // sc_start(1, SC_NS);

    // rst.write(0);
    // in_valid.write(1);
    // sc_start(1, SC_NS);

    return 0;
}