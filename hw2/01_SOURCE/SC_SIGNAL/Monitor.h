#ifndef MONITOR_H
#define MONITOR_H

#include <systemc.h>
#include <iostream>
#include <iomanip>

using namespace std;

#define MON_OUT_C 256
#define MON_OUT_H 6
#define MON_OUT_W 6
#define MON_OUT_SIZE (MON_OUT_C * MON_OUT_H * MON_OUT_W)

SC_MODULE(Monitor)
{
    sc_in_clk clock;
    sc_in<bool> in_valid;

    sc_vector<sc_in<double> > pool5_out;

    SC_CTOR(Monitor)
        : pool5_out("pool5_out", MON_OUT_SIZE)
    {
        SC_METHOD(run);
        sensitive << clock.pos();
        dont_initialize();
    }

    int idx(int c, int h, int w)
    {
        return c * (MON_OUT_H * MON_OUT_W) + h * MON_OUT_W + w;
    }

    void run()
    {
        if (in_valid.read() == 1)
        {
            cout << "\n[Monitor] Pool5 output ready" << endl;
            cout << "Shape = 256 x 6 x 6" << endl;

            for (int c = 0; c < 3; c++)
            {
                cout << "Channel " << c << " (6x6):" << endl;
                for (int h = 0; h < 6; h++)
                {
                    for (int w = 0; w < 6; w++)
                    {
                        cout << fixed << setprecision(5)
                             << pool5_out[idx(c, h, w)].read() << " ";
                    }
                    cout << endl;
                }
                cout << endl;
            }

            double min_v = pool5_out[0].read();
            double max_v = pool5_out[0].read();
            double sum = 0.0;

            for (int i = 0; i < MON_OUT_SIZE; i++)
            {
                double v = pool5_out[i].read();
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
                sum += v;
            }

            cout << "Stats:" << endl;
            cout << "min  = " << min_v << endl;
            cout << "max  = " << max_v << endl;
            cout << "mean = " << sum / MON_OUT_SIZE << endl;

            
            exit(0);
            return;
        }
    }
};

#endif