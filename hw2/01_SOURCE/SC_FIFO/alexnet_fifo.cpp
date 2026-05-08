#include <systemc.h>
#include <iostream>
#include <string>

#include "Pattern.h"
#include "TensorPacket.h"
#include "Padding_fifo.h"
#include "Conv2D_fifo.h"
#include "Relu_fifo.h"
#include "MaxPool_fifo.h"
#include "FullyConnected_fifo.h"
#include "Softmax_fifo.h"

using namespace std;

// ============================================================
// Pattern (signal) -> FIFO bridge
// ============================================================
SC_MODULE(PatternToFifo)
{
    sc_in<bool> clk;
    sc_in<bool> rst;
    sc_in<bool> in_valid;

    sc_vector< sc_in<double> > img_in;
    sc_fifo_out<TensorPacket> out_fifo;

    bool pending;
    TensorPacket buf;

    SC_HAS_PROCESS(PatternToFifo);

    PatternToFifo(sc_module_name name)
    : sc_module(name),
      img_in("img_in", 224 * 224 * 3),
      pending(false),
      buf(3, 224, 224)
    {
        SC_THREAD(run);
        sensitive << clk.pos();
        dont_initialize();
    }

    void run()
    {
        while (true)
        {
            wait();

            if (rst.read())
            {
                pending = false;
                buf = TensorPacket(3, 224, 224);
                continue;
            }

            if (!pending && in_valid.read())
            {
                buf = TensorPacket(3, 224, 224);

                for (int i = 0; i < 224 * 224 * 3; i++)
                {
                    buf.data[i] = img_in[i].read();
                }

                pending = true;
            }

            if (pending)
            {
                if (out_fifo.nb_write(buf))
                {
                    pending = false;
                }
            }
        }
    }
};

// ============================================================
// FIFO fork: duplicate FC8 output
//   one path -> output_linear
//   one path -> softmax
// ============================================================
SC_MODULE(ForkFifo)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    sc_fifo_in<TensorPacket> in_fifo;
    sc_fifo_out<TensorPacket> out_a;
    sc_fifo_out<TensorPacket> out_b;

    bool has_data;
    bool sent_a;
    bool sent_b;
    TensorPacket buf;

    SC_HAS_PROCESS(ForkFifo);

    ForkFifo(sc_module_name name)
    : sc_module(name),
      has_data(false),
      sent_a(false),
      sent_b(false)
    {
        SC_THREAD(run);
        sensitive << clk.pos();
        dont_initialize();
    }

    void run()
    {
        while (true)
        {
            wait();

            if (rst.read())
            {
                has_data = false;
                sent_a = false;
                sent_b = false;
                buf = TensorPacket();
                continue;
            }

            if (!has_data)
            {
                if (in_fifo.nb_read(buf))
                {
                    has_data = true;
                    sent_a = false;
                    sent_b = false;
                }
            }

            if (has_data)
            {
                if (!sent_a && out_a.nb_write(buf))
                {
                    sent_a = true;
                }

                if (!sent_b && out_b.nb_write(buf))
                {
                    sent_b = true;
                }

                if (sent_a && sent_b)
                {
                    has_data = false;
                }
            }
        }
    }
};

// ============================================================
// FIFO -> Pattern (signal) bridge
// ============================================================
SC_MODULE(FifoToPattern)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    sc_fifo_in<TensorPacket> linear_fifo;
    sc_fifo_in<TensorPacket> softmax_fifo;

    sc_out<bool> out_valid;
    sc_vector< sc_out<double> > output_linear;
    sc_vector< sc_out<double> > output_softmax;

    bool have_linear;
    bool have_softmax;
    TensorPacket linear_buf;
    TensorPacket softmax_buf;

    SC_HAS_PROCESS(FifoToPattern);

    FifoToPattern(sc_module_name name)
    : sc_module(name),
      output_linear("output_linear", 1000),
      output_softmax("output_softmax", 1000),
      have_linear(false),
      have_softmax(false),
      linear_buf(1, 1, 1000),
      softmax_buf(1, 1, 1000)
    {
        SC_THREAD(run);
        sensitive << clk.pos();
        dont_initialize();
    }

    void clear_outputs()
    {
        for (int i = 0; i < 1000; i++)
        {
            output_linear[i].write(0.0);
            output_softmax[i].write(0.0);
        }
    }

    void run()
    {
        while (true)
        {
            wait();

            if (rst.read())
            {
                clear_outputs();
                out_valid.write(0);
                have_linear = false;
                have_softmax = false;
                continue;
            }

            if (!have_linear)
            {
                have_linear = linear_fifo.nb_read(linear_buf);
            }

            if (!have_softmax)
            {
                have_softmax = softmax_fifo.nb_read(softmax_buf);
            }

            if (have_linear && have_softmax)
            {
                for (int i = 0; i < 1000; i++)
                {
                    output_linear[i].write(linear_buf.data[i]);
                    output_softmax[i].write(softmax_buf.data[i]);
                }

                out_valid.write(1);
                have_linear = false;
                have_softmax = false;
            }
            else
            {
                out_valid.write(0);
            }
        }
    }
};

// ============================================================
// sc_main
// ============================================================
int sc_main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <dog.txt|cat.txt>" << endl;
        return 1;
    }

    string img_name = argv[1];

    sc_clock clk("clk", 1, SC_NS);

    // Pattern-side signals
    sc_signal<bool> rst_sig("rst_sig");
    sc_signal<bool> in_valid_sig("in_valid_sig");
    sc_signal<bool> out_valid_sig("out_valid_sig");

    sc_vector< sc_signal<double> > img_sig("img_sig", 224 * 224 * 3);
    sc_vector< sc_signal<double> > linear_sig("linear_sig", 1000);
    sc_vector< sc_signal<double> > softmax_sig("softmax_sig", 1000);

    // FIFO channels
    sc_fifo<TensorPacket> fifo_input_to_pad(1);
    sc_fifo<TensorPacket> fifo_pad_to_conv1(1);
    sc_fifo<TensorPacket> fifo_conv1_to_relu1(1);
    sc_fifo<TensorPacket> fifo_relu1_to_pool1(1);

    sc_fifo<TensorPacket> fifo_pool1_to_conv2(1);
    sc_fifo<TensorPacket> fifo_conv2_to_relu2(1);
    sc_fifo<TensorPacket> fifo_relu2_to_pool2(1);

    sc_fifo<TensorPacket> fifo_pool2_to_conv3(1);
    sc_fifo<TensorPacket> fifo_conv3_to_relu3(1);

    sc_fifo<TensorPacket> fifo_relu3_to_conv4(1);
    sc_fifo<TensorPacket> fifo_conv4_to_relu4(1);

    sc_fifo<TensorPacket> fifo_relu4_to_conv5(1);
    sc_fifo<TensorPacket> fifo_conv5_to_relu5(1);
    sc_fifo<TensorPacket> fifo_relu5_to_pool5(1);

    sc_fifo<TensorPacket> fifo_pool5_to_fc6(1);
    sc_fifo<TensorPacket> fifo_fc6_to_relu6(1);

    sc_fifo<TensorPacket> fifo_relu6_to_fc7(1);
    sc_fifo<TensorPacket> fifo_fc7_to_relu7(1);

    sc_fifo<TensorPacket> fifo_relu7_to_fc8(1);
    sc_fifo<TensorPacket> fifo_fc8_raw(1);
    sc_fifo<TensorPacket> fifo_fc8_to_linear(1);
    sc_fifo<TensorPacket> fifo_fc8_to_softmax(1);
    sc_fifo<TensorPacket> fifo_softmax_to_output(1);

    // --------------------------------------------------------
    // Pattern
    // --------------------------------------------------------
    Pattern pattern("pattern", img_name);
    pattern.clock(clk);
    pattern.rst(rst_sig);
    pattern.in_valid(in_valid_sig);
    pattern.out_valid(out_valid_sig);

    for (int i = 0; i < 224 * 224 * 3; i++)
    {
        pattern.img[i](img_sig[i]);
    }

    for (int i = 0; i < 1000; i++)
    {
        pattern.output_linear[i](linear_sig[i]);
        pattern.output_softmax[i](softmax_sig[i]);
    }

    // --------------------------------------------------------
    // Bridges
    // --------------------------------------------------------
    PatternToFifo input_bridge("input_bridge");
    input_bridge.clk(clk);
    input_bridge.rst(rst_sig);
    input_bridge.in_valid(in_valid_sig);
    input_bridge.out_fifo(fifo_input_to_pad);

    for (int i = 0; i < 224 * 224 * 3; i++)
    {
        input_bridge.img_in[i](img_sig[i]);
    }

    ForkFifo fc8_fork("fc8_fork");
    fc8_fork.clk(clk);
    fc8_fork.rst(rst_sig);
    fc8_fork.in_fifo(fifo_fc8_raw);
    fc8_fork.out_a(fifo_fc8_to_linear);
    fc8_fork.out_b(fifo_fc8_to_softmax);

    FifoToPattern output_bridge("output_bridge");
    output_bridge.clk(clk);
    output_bridge.rst(rst_sig);
    output_bridge.linear_fifo(fifo_fc8_to_linear);
    output_bridge.softmax_fifo(fifo_softmax_to_output);
    output_bridge.out_valid(out_valid_sig);

    for (int i = 0; i < 1000; i++)
    {
        output_bridge.output_linear[i](linear_sig[i]);
        output_bridge.output_softmax[i](softmax_sig[i]);
    }

    // --------------------------------------------------------
    // AlexNet FIFO modules
    // --------------------------------------------------------
    PaddingFIFO pad("pad", 3, 224, 224, 2, 2, 1, 1);
    pad.clk(clk);
    pad.rst(rst_sig);
    pad.in_fifo(fifo_input_to_pad);
    pad.out_fifo(fifo_pad_to_conv1);

    Conv2DFIFO conv1(
        "conv1",
        3, 227, 227, 64, 11, 4, 0,
        "../../00_TESTBED/data/conv1_weight.txt",
        "../../00_TESTBED/data/conv1_bias.txt"
    );
    conv1.clk(clk);
    conv1.rst(rst_sig);
    conv1.in_fifo(fifo_pad_to_conv1);
    conv1.out_fifo(fifo_conv1_to_relu1);

    ReLUFIFO relu1("relu1", 55 * 55 * 64);
    relu1.clk(clk);
    relu1.rst(rst_sig);
    relu1.in_fifo(fifo_conv1_to_relu1);
    relu1.out_fifo(fifo_relu1_to_pool1);

    MaxPoolFIFO pool1("pool1", 64, 55, 55, 3, 2);
    pool1.clk(clk);
    pool1.rst(rst_sig);
    pool1.in_fifo(fifo_relu1_to_pool1);
    pool1.out_fifo(fifo_pool1_to_conv2);

    Conv2DFIFO conv2(
        "conv2",
        64, 27, 27, 192, 5, 1, 2,
        "../../00_TESTBED/data/conv2_weight.txt",
        "../../00_TESTBED/data/conv2_bias.txt"
    );
    conv2.clk(clk);
    conv2.rst(rst_sig);
    conv2.in_fifo(fifo_pool1_to_conv2);
    conv2.out_fifo(fifo_conv2_to_relu2);

    ReLUFIFO relu2("relu2", 27 * 27 * 192);
    relu2.clk(clk);
    relu2.rst(rst_sig);
    relu2.in_fifo(fifo_conv2_to_relu2);
    relu2.out_fifo(fifo_relu2_to_pool2);

    MaxPoolFIFO pool2("pool2", 192, 27, 27, 3, 2);
    pool2.clk(clk);
    pool2.rst(rst_sig);
    pool2.in_fifo(fifo_relu2_to_pool2);
    pool2.out_fifo(fifo_pool2_to_conv3);

    Conv2DFIFO conv3(
        "conv3",
        192, 13, 13, 384, 3, 1, 1,
        "../../00_TESTBED/data/conv3_weight.txt",
        "../../00_TESTBED/data/conv3_bias.txt"
    );
    conv3.clk(clk);
    conv3.rst(rst_sig);
    conv3.in_fifo(fifo_pool2_to_conv3);
    conv3.out_fifo(fifo_conv3_to_relu3);

    ReLUFIFO relu3("relu3", 13 * 13 * 384);
    relu3.clk(clk);
    relu3.rst(rst_sig);
    relu3.in_fifo(fifo_conv3_to_relu3);
    relu3.out_fifo(fifo_relu3_to_conv4);

    Conv2DFIFO conv4(
        "conv4",
        384, 13, 13, 256, 3, 1, 1,
        "../../00_TESTBED/data/conv4_weight.txt",
        "../../00_TESTBED/data/conv4_bias.txt"
    );
    conv4.clk(clk);
    conv4.rst(rst_sig);
    conv4.in_fifo(fifo_relu3_to_conv4);
    conv4.out_fifo(fifo_conv4_to_relu4);

    ReLUFIFO relu4("relu4", 13 * 13 * 256);
    relu4.clk(clk);
    relu4.rst(rst_sig);
    relu4.in_fifo(fifo_conv4_to_relu4);
    relu4.out_fifo(fifo_relu4_to_conv5);

    Conv2DFIFO conv5(
        "conv5",
        256, 13, 13, 256, 3, 1, 1,
        "../../00_TESTBED/data/conv5_weight.txt",
        "../../00_TESTBED/data/conv5_bias.txt"
    );
    conv5.clk(clk);
    conv5.rst(rst_sig);
    conv5.in_fifo(fifo_relu4_to_conv5);
    conv5.out_fifo(fifo_conv5_to_relu5);

    ReLUFIFO relu5("relu5", 13 * 13 * 256);
    relu5.clk(clk);
    relu5.rst(rst_sig);
    relu5.in_fifo(fifo_conv5_to_relu5);
    relu5.out_fifo(fifo_relu5_to_pool5);

    MaxPoolFIFO pool5("pool5", 256, 13, 13, 3, 2);
    pool5.clk(clk);
    pool5.rst(rst_sig);
    pool5.in_fifo(fifo_relu5_to_pool5);
    pool5.out_fifo(fifo_pool5_to_fc6);

    FullyConnectedFIFO fc6(
        "fc6",
        6 * 6 * 256, 4096,
        "../../00_TESTBED/data/fc6_weight.txt",
        "../../00_TESTBED/data/fc6_bias.txt"
    );
    fc6.clk(clk);
    fc6.rst(rst_sig);
    fc6.in_fifo(fifo_pool5_to_fc6);
    fc6.out_fifo(fifo_fc6_to_relu6);

    ReLUFIFO relu6("relu6", 4096);
    relu6.clk(clk);
    relu6.rst(rst_sig);
    relu6.in_fifo(fifo_fc6_to_relu6);
    relu6.out_fifo(fifo_relu6_to_fc7);

    FullyConnectedFIFO fc7(
        "fc7",
        4096, 4096,
        "../../00_TESTBED/data/fc7_weight.txt",
        "../../00_TESTBED/data/fc7_bias.txt"
    );
    fc7.clk(clk);
    fc7.rst(rst_sig);
    fc7.in_fifo(fifo_relu6_to_fc7);
    fc7.out_fifo(fifo_fc7_to_relu7);

    ReLUFIFO relu7("relu7", 4096);
    relu7.clk(clk);
    relu7.rst(rst_sig);
    relu7.in_fifo(fifo_fc7_to_relu7);
    relu7.out_fifo(fifo_relu7_to_fc8);

    FullyConnectedFIFO fc8(
        "fc8",
        4096, 1000,
        "../../00_TESTBED/data/fc8_weight.txt",
        "../../00_TESTBED/data/fc8_bias.txt"
    );
    fc8.clk(clk);
    fc8.rst(rst_sig);
    fc8.in_fifo(fifo_relu7_to_fc8);
    fc8.out_fifo(fifo_fc8_raw);

    SoftmaxFIFO softmax("softmax", 1000);
    softmax.clk(clk);
    softmax.rst(rst_sig);
    softmax.in_fifo(fifo_fc8_to_softmax);
    softmax.out_fifo(fifo_softmax_to_output);

    sc_start();
    return 0;
}