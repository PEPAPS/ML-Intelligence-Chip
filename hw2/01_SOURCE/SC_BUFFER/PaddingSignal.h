#ifndef PADDING_SIGNAL_H
#define PADDING_SIGNAL_H

#include <systemc.h>

// ===== Dimensions =====
#define IN_H 224
#define IN_W 224
#define IN_C 3

#define PAD_TOP    2
#define PAD_LEFT   2
#define PAD_BOTTOM 1
#define PAD_RIGHT  1

#define OUT_H (IN_H + PAD_TOP + PAD_BOTTOM)   // 227
#define OUT_W (IN_W + PAD_LEFT + PAD_RIGHT)   // 227
#define OUT_C IN_C

#define IN_SIZE  (IN_H * IN_W * IN_C)
#define OUT_SIZE (OUT_H * OUT_W * OUT_C)

SC_MODULE(PaddingSignal)
{
    // ===== Ports =====
    sc_in<bool> clk;
    sc_in<bool> rst;
    sc_in<bool> in_valid;

    sc_out<bool> out_valid;

    // Input image (flattened)
    sc_in<double> img_in[IN_SIZE];

    // Output padded image
    sc_out<double> img_out[OUT_SIZE];

    // ===== Constructor =====
    SC_CTOR(PaddingSignal)
    {
        SC_METHOD(run);
        sensitive << clk.pos();
        dont_initialize();
    }

    // ===== Index helpers =====
    int idx_in(int c, int h, int w)
    {
        return c * (IN_H * IN_W) + h * IN_W + w;
    }

    int idx_out(int c, int h, int w)
    {
        return c * (OUT_H * OUT_W) + h * OUT_W + w;
    }

    // ===== Main Logic =====
    void run()
    {
        if (rst.read() == 1)
        {
            // reset output
            for (int i = 0; i < OUT_SIZE; i++)
            {
                img_out[i].write(0.0);
            }
            out_valid.write(0);
            return;
        }

        if (in_valid.read() == 1)
        {
            // Step 1: initialize all output to zero
            for (int i = 0; i < OUT_SIZE; i++)
            {
                img_out[i].write(0.0);
            }

            // Step 2: copy input into padded position
            for (int c = 0; c < IN_C; c++)
            {
                for (int h = 0; h < IN_H; h++)
                {
                    for (int w = 0; w < IN_W; w++)
                    {
                        double val = img_in[idx_in(c, h, w)].read();

                        int out_h = h + PAD_TOP;
                        int out_w = w + PAD_LEFT;

                        img_out[idx_out(c, out_h, out_w)].write(val);
                    }
                }
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