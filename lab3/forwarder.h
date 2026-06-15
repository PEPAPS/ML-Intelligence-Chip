#ifndef FORWARDER_ADDR_SC_THREAD_V2_H
#define FORWARDER_ADDR_SC_THREAD_V2_H

#include <systemc.h>
#include <iostream>
#include <vector>

SC_MODULE(Forwarder)
{
    // Ports
    sc_in<bool> ACLK;
    sc_in<bool> ARESETn; // Active-low reset

    // --- Request Interface (Master to Producer) ---
    sc_out<unsigned int> REQ_ID_TO_PROD;
    sc_out<bool> REQ_VALID_TO_PROD;
    sc_in<bool> REQ_READY_FROM_PROD;

    // --- AXI Stream Data Interface (Slave from Producer) ---
    sc_in<bool> S_AXIS_TVALID;
    sc_out<bool> S_AXIS_TREADY;
    sc_in<char> S_AXIS_TDATA;
    sc_in<bool> S_AXIS_TLAST;

    // --- AXI Stream Data Interface (Master to Checker) ---
    sc_out<bool> M_AXIS_TVALID;
    sc_in<bool> M_AXIS_TREADY;
    sc_out<char> M_AXIS_TDATA;
    sc_out<bool> M_AXIS_TLAST;

    const unsigned int TOTAL_STRINGS_TO_REQUEST; // Define how many strings to request

    // hint : You can directly pull S_AXIS_TREADY to 1
    // hint : You can directly set REQ_ID_TO_PROD to 0
    void forwarder_thread_logic()
{
    REQ_ID_TO_PROD.write(0);
    REQ_VALID_TO_PROD.write(false);

    S_AXIS_TREADY.write(false);

    M_AXIS_TVALID.write(false);
    M_AXIS_TDATA.write('\0');
    M_AXIS_TLAST.write(false);

    bool requested = false;
    bool request_done = false;
    bool finished = false;

    while (true) {
        wait();

        if (!ARESETn.read()) {
            REQ_ID_TO_PROD.write(0);
            REQ_VALID_TO_PROD.write(false);

            S_AXIS_TREADY.write(false);

            M_AXIS_TVALID.write(false);
            M_AXIS_TDATA.write('\0');
            M_AXIS_TLAST.write(false);

            requested = false;
            request_done = false;
            finished = false;
            continue;
        }

        if (finished) {
            REQ_VALID_TO_PROD.write(false);
            S_AXIS_TREADY.write(false);
            M_AXIS_TVALID.write(false);
            M_AXIS_TDATA.write('\0');
            M_AXIS_TLAST.write(false);
            continue;
        }

        // ------------------------------------------------------------
        // 1. Send one request to Producer
        // ------------------------------------------------------------
        if (!request_done) {
            REQ_ID_TO_PROD.write(0);

            if (!requested) {
                REQ_VALID_TO_PROD.write(true);
                requested = true;
            }

            if (REQ_VALID_TO_PROD.read() && REQ_READY_FROM_PROD.read()) {
                REQ_VALID_TO_PROD.write(false);
                request_done = true;

                // Lab hint: directly pull S_AXIS_TREADY to 1
                S_AXIS_TREADY.write(true);
            }

            M_AXIS_TVALID.write(false);
            M_AXIS_TDATA.write('\0');
            M_AXIS_TLAST.write(false);
            continue;
        }

        // ------------------------------------------------------------
        // 2. Forward Producer data to Checker
        // ------------------------------------------------------------
        S_AXIS_TREADY.write(true);

        if (S_AXIS_TVALID.read()) {
            M_AXIS_TDATA.write(S_AXIS_TDATA.read());
            M_AXIS_TLAST.write(S_AXIS_TLAST.read());
            M_AXIS_TVALID.write(true);

            if (S_AXIS_TLAST.read()) {
                finished = true;
            }
        } else {
            M_AXIS_TVALID.write(false);
            M_AXIS_TDATA.write('\0');
            M_AXIS_TLAST.write(false);
        }
    }
}

    // if you need to add more functions, please add them here

    SC_CTOR(Forwarder) : TOTAL_STRINGS_TO_REQUEST(1) // Initialize const member in initializer list
    {
        SC_THREAD(forwarder_thread_logic);
        sensitive << ACLK.pos();
        sensitive << ARESETn.neg();
    }
};

#endif // FORWARDER_ADDR_SC_THREAD_V2_H