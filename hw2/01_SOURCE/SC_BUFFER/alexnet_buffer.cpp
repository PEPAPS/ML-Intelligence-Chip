#include <systemc.h>
#include <iostream>
#include <string>

#include "Pattern.h"
#include "PaddingSignal.h"
#include "Conv2DSignal.h"
#include "ReLUSignal.h"
#include "MaxPoolSignal.h"
#include "FullyConnectedSignal.h"
#include "SoftmaxSignal.h"

using namespace std;

int sc_main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: ./run <cat.txt|dog.txt>" << endl;
        return 1;
    }

    string img_name = string(argv[1]);

    sc_clock clk("clk", 1, SC_NS);

    // control channels: use sc_buffer instead of sc_signal
    sc_buffer<bool> rst_sig;
    sc_buffer<bool> in_valid_sig;
    sc_buffer<bool> pad_out_valid_sig;
    sc_buffer<bool> conv1_out_valid_sig;
    sc_buffer<bool> relu1_out_valid_sig;
    sc_buffer<bool> pool1_out_valid_sig;
    sc_buffer<bool> conv2_out_valid_sig;
    sc_buffer<bool> relu2_out_valid_sig;
    sc_buffer<bool> pool2_out_valid_sig;
    sc_buffer<bool> conv3_out_valid_sig;
    sc_buffer<bool> relu3_out_valid_sig;
    sc_buffer<bool> conv4_out_valid_sig;
    sc_buffer<bool> relu4_out_valid_sig;
    sc_buffer<bool> conv5_out_valid_sig;
    sc_buffer<bool> relu5_out_valid_sig;
    sc_buffer<bool> pool5_out_valid_sig;

    sc_buffer<bool> fc6_out_valid_sig;
    sc_buffer<bool> relu6_out_valid_sig;
    sc_buffer<bool> fc7_out_valid_sig;
    sc_buffer<bool> relu7_out_valid_sig;
    sc_buffer<bool> fc8_out_valid_sig;
    sc_buffer<bool> softmax_out_valid_sig;

    const int IMG_SIZE       = 224 * 224 * 3;
    const int PAD_SIZE       = 227 * 227 * 3;

    const int CONV1_OUT_SIZE = 64  * 55 * 55;
    const int POOL1_OUT_SIZE = 64  * 27 * 27;

    const int CONV2_OUT_SIZE = 192 * 27 * 27;
    const int POOL2_OUT_SIZE = 192 * 13 * 13;

    const int CONV3_OUT_SIZE = 384 * 13 * 13;
    const int CONV4_OUT_SIZE = 256 * 13 * 13;
    const int CONV5_OUT_SIZE = 256 * 13 * 13;
    const int POOL5_OUT_SIZE = 256 * 6  * 6;

    const int FC6_OUT_SIZE   = 4096;
    const int FC7_OUT_SIZE   = 4096;
    const int FC8_OUT_SIZE   = 1000;

    // data channels: use sc_buffer 
    sc_buffer<double>* img_sig     = new sc_buffer<double>[IMG_SIZE];
    sc_buffer<double>* padded_sig  = new sc_buffer<double>[PAD_SIZE];

    sc_buffer<double>* conv1_sig   = new sc_buffer<double>[CONV1_OUT_SIZE];
    sc_buffer<double>* relu1_sig   = new sc_buffer<double>[CONV1_OUT_SIZE];
    sc_buffer<double>* pool1_sig   = new sc_buffer<double>[POOL1_OUT_SIZE];

    sc_buffer<double>* conv2_sig   = new sc_buffer<double>[CONV2_OUT_SIZE];
    sc_buffer<double>* relu2_sig   = new sc_buffer<double>[CONV2_OUT_SIZE];
    sc_buffer<double>* pool2_sig   = new sc_buffer<double>[POOL2_OUT_SIZE];

    sc_buffer<double>* conv3_sig   = new sc_buffer<double>[CONV3_OUT_SIZE];
    sc_buffer<double>* relu3_sig   = new sc_buffer<double>[CONV3_OUT_SIZE];

    sc_buffer<double>* conv4_sig   = new sc_buffer<double>[CONV4_OUT_SIZE];
    sc_buffer<double>* relu4_sig   = new sc_buffer<double>[CONV4_OUT_SIZE];

    sc_buffer<double>* conv5_sig   = new sc_buffer<double>[CONV5_OUT_SIZE];
    sc_buffer<double>* relu5_sig   = new sc_buffer<double>[CONV5_OUT_SIZE];
    sc_buffer<double>* pool5_sig   = new sc_buffer<double>[POOL5_OUT_SIZE];

    sc_buffer<double>* fc6_sig     = new sc_buffer<double>[FC6_OUT_SIZE];
    sc_buffer<double>* relu6_sig   = new sc_buffer<double>[FC6_OUT_SIZE];

    sc_buffer<double>* fc7_sig     = new sc_buffer<double>[FC7_OUT_SIZE];
    sc_buffer<double>* relu7_sig   = new sc_buffer<double>[FC7_OUT_SIZE];

    sc_buffer<double>* fc8_sig     = new sc_buffer<double>[FC8_OUT_SIZE];
    sc_buffer<double>* softmax_sig = new sc_buffer<double>[FC8_OUT_SIZE];

    // modules
    Pattern* pattern = new Pattern("pattern", img_name);
    PaddingSignal* padding = new PaddingSignal("padding");

    Conv2DSignal* conv1 = new Conv2DSignal(
        "conv1",
        3, 227, 227,
        64, 11, 4, 0,
        "../../00_TESTBED/data/conv1_weight.txt",
        "../../00_TESTBED/data/conv1_bias.txt"
    );
    ReLUSignal* relu1 = new ReLUSignal("relu1", CONV1_OUT_SIZE);
    MaxPoolSignal* pool1 = new MaxPoolSignal("pool1", 64, 55, 55, 3, 2);

    Conv2DSignal* conv2 = new Conv2DSignal(
        "conv2",
        64, 27, 27,
        192, 5, 1, 2,
        "../../00_TESTBED/data/conv2_weight.txt",
        "../../00_TESTBED/data/conv2_bias.txt"
    );
    ReLUSignal* relu2 = new ReLUSignal("relu2", CONV2_OUT_SIZE);
    MaxPoolSignal* pool2 = new MaxPoolSignal("pool2", 192, 27, 27, 3, 2);

    Conv2DSignal* conv3 = new Conv2DSignal(
        "conv3",
        192, 13, 13,
        384, 3, 1, 1,
        "../../00_TESTBED/data/conv3_weight.txt",
        "../../00_TESTBED/data/conv3_bias.txt"
    );
    ReLUSignal* relu3 = new ReLUSignal("relu3", CONV3_OUT_SIZE);

    Conv2DSignal* conv4 = new Conv2DSignal(
        "conv4",
        384, 13, 13,
        256, 3, 1, 1,
        "../../00_TESTBED/data/conv4_weight.txt",
        "../../00_TESTBED/data/conv4_bias.txt"
    );
    ReLUSignal* relu4 = new ReLUSignal("relu4", CONV4_OUT_SIZE);

    Conv2DSignal* conv5 = new Conv2DSignal(
        "conv5",
        256, 13, 13,
        256, 3, 1, 1,
        "../../00_TESTBED/data/conv5_weight.txt",
        "../../00_TESTBED/data/conv5_bias.txt"
    );
    ReLUSignal* relu5 = new ReLUSignal("relu5", CONV5_OUT_SIZE);
    MaxPoolSignal* pool5 = new MaxPoolSignal("pool5", 256, 13, 13, 3, 2);

    FullyConnectedSignal* fc6 = new FullyConnectedSignal(
        "fc6",
        POOL5_OUT_SIZE,
        FC6_OUT_SIZE,
        "../../00_TESTBED/data/fc6_weight.txt",
        "../../00_TESTBED/data/fc6_bias.txt"
    );
    ReLUSignal* relu6 = new ReLUSignal("relu6", FC6_OUT_SIZE);

    FullyConnectedSignal* fc7 = new FullyConnectedSignal(
        "fc7",
        FC6_OUT_SIZE,
        FC7_OUT_SIZE,
        "../../00_TESTBED/data/fc7_weight.txt",
        "../../00_TESTBED/data/fc7_bias.txt"
    );
    ReLUSignal* relu7 = new ReLUSignal("relu7", FC7_OUT_SIZE);

    FullyConnectedSignal* fc8 = new FullyConnectedSignal(
        "fc8",
        FC7_OUT_SIZE,
        FC8_OUT_SIZE,
        "../../00_TESTBED/data/fc8_weight.txt",
        "../../00_TESTBED/data/fc8_bias.txt"
    );

    SoftmaxSignal* softmax = new SoftmaxSignal("softmax", FC8_OUT_SIZE);

    // ===== Pattern =====
    pattern->clock(clk);
    pattern->rst(rst_sig);
    pattern->in_valid(in_valid_sig);
    pattern->out_valid(softmax_out_valid_sig);

    for (int i = 0; i < IMG_SIZE; i++)
        pattern->img[i](img_sig[i]);

    for (int i = 0; i < FC8_OUT_SIZE; i++)
    {
        pattern->output_linear[i](fc8_sig[i]);
        pattern->output_softmax[i](softmax_sig[i]);
    }

    // ===== Padding =====
    padding->clk(clk);
    padding->rst(rst_sig);
    padding->in_valid(in_valid_sig);
    padding->out_valid(pad_out_valid_sig);

    for (int i = 0; i < IMG_SIZE; i++)
        padding->img_in[i](img_sig[i]);

    for (int i = 0; i < PAD_SIZE; i++)
        padding->img_out[i](padded_sig[i]);

    // ===== Conv1 =====
    conv1->clk(clk);
    conv1->rst(rst_sig);
    conv1->in_valid(pad_out_valid_sig);
    conv1->out_valid(conv1_out_valid_sig);

    for (int i = 0; i < PAD_SIZE; i++)
        conv1->img_in[i](padded_sig[i]);

    for (int i = 0; i < CONV1_OUT_SIZE; i++)
        conv1->img_out[i](conv1_sig[i]);

    // ===== ReLU1 =====
    relu1->clk(clk);
    relu1->rst(rst_sig);
    relu1->in_valid(conv1_out_valid_sig);
    relu1->out_valid(relu1_out_valid_sig);

    for (int i = 0; i < CONV1_OUT_SIZE; i++)
        relu1->data_in[i](conv1_sig[i]);

    for (int i = 0; i < CONV1_OUT_SIZE; i++)
        relu1->data_out[i](relu1_sig[i]);

    // ===== Pool1 =====
    pool1->clk(clk);
    pool1->rst(rst_sig);
    pool1->in_valid(relu1_out_valid_sig);
    pool1->out_valid(pool1_out_valid_sig);

    for (int i = 0; i < CONV1_OUT_SIZE; i++)
        pool1->data_in[i](relu1_sig[i]);

    for (int i = 0; i < POOL1_OUT_SIZE; i++)
        pool1->data_out[i](pool1_sig[i]);

    // ===== Conv2 =====
    conv2->clk(clk);
    conv2->rst(rst_sig);
    conv2->in_valid(pool1_out_valid_sig);
    conv2->out_valid(conv2_out_valid_sig);

    for (int i = 0; i < POOL1_OUT_SIZE; i++)
        conv2->img_in[i](pool1_sig[i]);

    for (int i = 0; i < CONV2_OUT_SIZE; i++)
        conv2->img_out[i](conv2_sig[i]);

    // ===== ReLU2 =====
    relu2->clk(clk);
    relu2->rst(rst_sig);
    relu2->in_valid(conv2_out_valid_sig);
    relu2->out_valid(relu2_out_valid_sig);

    for (int i = 0; i < CONV2_OUT_SIZE; i++)
        relu2->data_in[i](conv2_sig[i]);

    for (int i = 0; i < CONV2_OUT_SIZE; i++)
        relu2->data_out[i](relu2_sig[i]);

    // ===== Pool2 =====
    pool2->clk(clk);
    pool2->rst(rst_sig);
    pool2->in_valid(relu2_out_valid_sig);
    pool2->out_valid(pool2_out_valid_sig);

    for (int i = 0; i < CONV2_OUT_SIZE; i++)
        pool2->data_in[i](relu2_sig[i]);

    for (int i = 0; i < POOL2_OUT_SIZE; i++)
        pool2->data_out[i](pool2_sig[i]);

    // ===== Conv3 =====
    conv3->clk(clk);
    conv3->rst(rst_sig);
    conv3->in_valid(pool2_out_valid_sig);
    conv3->out_valid(conv3_out_valid_sig);

    for (int i = 0; i < POOL2_OUT_SIZE; i++)
        conv3->img_in[i](pool2_sig[i]);

    for (int i = 0; i < CONV3_OUT_SIZE; i++)
        conv3->img_out[i](conv3_sig[i]);

    // ===== ReLU3 =====
    relu3->clk(clk);
    relu3->rst(rst_sig);
    relu3->in_valid(conv3_out_valid_sig);
    relu3->out_valid(relu3_out_valid_sig);

    for (int i = 0; i < CONV3_OUT_SIZE; i++)
        relu3->data_in[i](conv3_sig[i]);

    for (int i = 0; i < CONV3_OUT_SIZE; i++)
        relu3->data_out[i](relu3_sig[i]);

    // ===== Conv4 =====
    conv4->clk(clk);
    conv4->rst(rst_sig);
    conv4->in_valid(relu3_out_valid_sig);
    conv4->out_valid(conv4_out_valid_sig);

    for (int i = 0; i < CONV3_OUT_SIZE; i++)
        conv4->img_in[i](relu3_sig[i]);

    for (int i = 0; i < CONV4_OUT_SIZE; i++)
        conv4->img_out[i](conv4_sig[i]);

    // ===== ReLU4 =====
    relu4->clk(clk);
    relu4->rst(rst_sig);
    relu4->in_valid(conv4_out_valid_sig);
    relu4->out_valid(relu4_out_valid_sig);

    for (int i = 0; i < CONV4_OUT_SIZE; i++)
        relu4->data_in[i](conv4_sig[i]);

    for (int i = 0; i < CONV4_OUT_SIZE; i++)
        relu4->data_out[i](relu4_sig[i]);

    // ===== Conv5 =====
    conv5->clk(clk);
    conv5->rst(rst_sig);
    conv5->in_valid(relu4_out_valid_sig);
    conv5->out_valid(conv5_out_valid_sig);

    for (int i = 0; i < CONV4_OUT_SIZE; i++)
        conv5->img_in[i](relu4_sig[i]);

    for (int i = 0; i < CONV5_OUT_SIZE; i++)
        conv5->img_out[i](conv5_sig[i]);

    // ===== ReLU5 =====
    relu5->clk(clk);
    relu5->rst(rst_sig);
    relu5->in_valid(conv5_out_valid_sig);
    relu5->out_valid(relu5_out_valid_sig);

    for (int i = 0; i < CONV5_OUT_SIZE; i++)
        relu5->data_in[i](conv5_sig[i]);

    for (int i = 0; i < CONV5_OUT_SIZE; i++)
        relu5->data_out[i](relu5_sig[i]);

    // ===== Pool5 =====
    pool5->clk(clk);
    pool5->rst(rst_sig);
    pool5->in_valid(relu5_out_valid_sig);
    pool5->out_valid(pool5_out_valid_sig);

    for (int i = 0; i < CONV5_OUT_SIZE; i++)
        pool5->data_in[i](relu5_sig[i]);

    for (int i = 0; i < POOL5_OUT_SIZE; i++)
        pool5->data_out[i](pool5_sig[i]);

    // ===== FC6 =====
    fc6->clk(clk);
    fc6->rst(rst_sig);
    fc6->in_valid(pool5_out_valid_sig);
    fc6->out_valid(fc6_out_valid_sig);

    for (int i = 0; i < POOL5_OUT_SIZE; i++)
        fc6->in_data[i](pool5_sig[i]);

    for (int i = 0; i < FC6_OUT_SIZE; i++)
        fc6->out_data[i](fc6_sig[i]);

    // ===== ReLU6 =====
    relu6->clk(clk);
    relu6->rst(rst_sig);
    relu6->in_valid(fc6_out_valid_sig);
    relu6->out_valid(relu6_out_valid_sig);

    for (int i = 0; i < FC6_OUT_SIZE; i++)
        relu6->data_in[i](fc6_sig[i]);

    for (int i = 0; i < FC6_OUT_SIZE; i++)
        relu6->data_out[i](relu6_sig[i]);

    // ===== FC7 =====
    fc7->clk(clk);
    fc7->rst(rst_sig);
    fc7->in_valid(relu6_out_valid_sig);
    fc7->out_valid(fc7_out_valid_sig);

    for (int i = 0; i < FC6_OUT_SIZE; i++)
        fc7->in_data[i](relu6_sig[i]);

    for (int i = 0; i < FC7_OUT_SIZE; i++)
        fc7->out_data[i](fc7_sig[i]);

    // ===== ReLU7 =====
    relu7->clk(clk);
    relu7->rst(rst_sig);
    relu7->in_valid(fc7_out_valid_sig);
    relu7->out_valid(relu7_out_valid_sig);

    for (int i = 0; i < FC7_OUT_SIZE; i++)
        relu7->data_in[i](fc7_sig[i]);

    for (int i = 0; i < FC7_OUT_SIZE; i++)
        relu7->data_out[i](relu7_sig[i]);

    // ===== FC8 =====
    fc8->clk(clk);
    fc8->rst(rst_sig);
    fc8->in_valid(relu7_out_valid_sig);
    fc8->out_valid(fc8_out_valid_sig);

    for (int i = 0; i < FC7_OUT_SIZE; i++)
        fc8->in_data[i](relu7_sig[i]);

    for (int i = 0; i < FC8_OUT_SIZE; i++)
        fc8->out_data[i](fc8_sig[i]);

    // ===== Softmax =====
    softmax->clk(clk);
    softmax->rst(rst_sig);
    softmax->in_valid(fc8_out_valid_sig);
    softmax->out_valid(softmax_out_valid_sig);

    for (int i = 0; i < FC8_OUT_SIZE; i++)
        softmax->in_data[i](fc8_sig[i]);

    for (int i = 0; i < FC8_OUT_SIZE; i++)
        softmax->out_data[i](softmax_sig[i]);

    sc_start();
    return 0;
}
