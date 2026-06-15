#ifndef ROUTER_H
#define ROUTER_H

#include "systemc.h"
#include <queue>

#define EAST      0
#define SOUTH     1
#define WEST      2
#define NORTH     3
#define LOCAL     4
#define CTRL_PORT 5
#define IDLE      6

#define ROUTER_PORTS 6

using namespace std;

SC_MODULE( Router ) {
    sc_in<bool> rst;
    sc_in<bool> clk;

    sc_out<sc_lv<34> > out_flit[ROUTER_PORTS];
    sc_out<bool>       out_req[ROUTER_PORTS];
    sc_in<bool>        in_ack[ROUTER_PORTS];

    sc_in<sc_lv<34> >  in_flit[ROUTER_PORTS];
    sc_in<bool>        in_req[ROUTER_PORTS];
    sc_out<bool>       out_ack[ROUTER_PORTS];

    int id;

    queue<sc_lv<34> > input_buffer[ROUTER_PORTS];

    // Packet-atomic output arbitration.
    // Once a HEAD flit chooses an output port, that output port is locked
    // to the same input port until the TAIL flit passes.
    // This prevents HEAD/BODY/TAIL from different packets from interleaving
    // at the same destination core.
    bool out_port_busy[ROUTER_PORTS];
    int out_port_owner[ROUTER_PORTS];

    void tx_process();
    void rx_process();

    int get_dest_id(sc_lv<34> flit);
    int get_packet_type(sc_lv<34> flit);
    int get_flit_kind(sc_lv<34> flit);
    int get_output_port(int dest_id, int packet_type);

    SC_HAS_PROCESS(Router);

    Router(sc_module_name name, int router_id)
        : sc_module(name), id(router_id)
    {
        SC_METHOD(rx_process);
        sensitive << clk.pos() << rst.pos();

        SC_METHOD(tx_process);
        sensitive << clk.pos() << rst.pos();
    }
};

#endif
