#ifndef FORWARDER_H
#define FORWARDER_H

#include <systemc.h>

SC_MODULE(Forwarder) {
    // Ports
    sc_in<bool> ACLK;
    sc_in<bool> ARESETn;

    // Slave Interface (from Producer)
    sc_in<bool>         S_AXIS_TVALID;
    sc_out<bool>        S_AXIS_TREADY;
    sc_in<char>         S_AXIS_TDATA; 
    sc_in<bool>         S_AXIS_TLAST;

    // Master Interface (to Checker)
    sc_out<bool>        M_AXIS_TVALID;
    sc_in<bool>         M_AXIS_TREADY;
    sc_out<char>        M_AXIS_TDATA;
    sc_out<bool>        M_AXIS_TLAST;

    //hint : You can directly pull S_AXIS_TREADY to 1
     void forward_data_proc() {
        if (!ARESETn.read()) {
            S_AXIS_TREADY.write(true);

            M_AXIS_TVALID.write(false);
            M_AXIS_TDATA.write('\0');
            M_AXIS_TLAST.write(false);

            cout << sc_time_stamp() << ": Forwarder: Resetting..." << endl;
        } 
        else {
            // Forwarder is always ready to receive data from Producer
            S_AXIS_TREADY.write(true);

            // Transfer from Producer to Forwarder happens when:
            // S_AXIS_TVALID && S_AXIS_TREADY
            // Transfer from Forwarder to Checker happens when:
            // M_AXIS_TVALID && M_AXIS_TREADY

            if (S_AXIS_TVALID.read() && M_AXIS_TREADY.read()) {
                M_AXIS_TDATA.write(S_AXIS_TDATA.read());
                M_AXIS_TVALID.write(true);
                M_AXIS_TLAST.write(S_AXIS_TLAST.read());

                cout << sc_time_stamp() << ": Forwarder: Forwarded '"
                     << S_AXIS_TDATA.read()
                     << "' (TLAST=" << S_AXIS_TLAST.read() << ")" << endl;
            } 
            else {
                M_AXIS_TVALID.write(false);
                M_AXIS_TLAST.write(false);
            }
        }
    }

    SC_CTOR(Forwarder)
    {
        SC_METHOD(forward_data_proc);
        sensitive << ACLK.pos() << ARESETn.neg();
    }
};

#endif // FORWARDER_H