#ifndef ROUTER_H
#define ROUTER_H

#include "systemc.h"
#include <queue>

SC_MODULE(Router) {
    sc_in<bool> rst;
    sc_in<bool> clk;

    sc_out<sc_lv<34> > out_flit[5];
    sc_out<bool> out_req[5];
    sc_in<bool> in_ack[5];

    sc_in<sc_lv<34> > in_flit[5];
    sc_in<bool> in_req[5];
    sc_out<bool> out_ack[5];

    std::queue<sc_lv<34> > in_q[5];
    std::queue<sc_lv<34> > out_q[5];

    int in_port_state[5];
    int out_port_lock[5];
    int router_id;

    enum PortId {
        PORT_NORTH = 0,
        PORT_SOUTH = 1,
        PORT_EAST = 2,
        PORT_WEST = 3,
        PORT_LOCAL = 4,
        PORT_COUNT = 5
    };

    enum FlitType {
        FLIT_BODY = 0,
        FLIT_TAIL = 1,
        FLIT_HEAD = 2
    };

    void init(int id) {
        router_id = id;
    }

    int get_xy_route(int current_id, int dest_id) {
        int cx = current_id % 4;
        int cy = current_id / 4;
        int dx = dest_id % 4;
        int dy = dest_id / 4;

        if (dx > cx) return PORT_EAST;
        if (dx < cx) return PORT_WEST;
        if (dy > cy) return PORT_SOUTH;
        if (dy < cy) return PORT_NORTH;
        return PORT_LOCAL;
    }

    void clear_queues() {
        for (int i = 0; i < PORT_COUNT; ++i) {
            while (!in_q[i].empty()) in_q[i].pop();
            while (!out_q[i].empty()) out_q[i].pop();
            in_port_state[i] = -1;
            out_port_lock[i] = -1;
        }
    }

    void sample_inputs() {
        for (int i = 0; i < PORT_COUNT; ++i) {
            out_ack[i].write(true);
            if (in_req[i].read()) {
                in_q[i].push(in_flit[i].read());
            }
        }
    }

    void route_flits() {
        for (int i = 0; i < PORT_COUNT; ++i) {
            if (in_q[i].empty()) {
                continue;
            }

            sc_lv<34> flit = in_q[i].front();
            unsigned int type = flit.range(33, 32).to_uint();

            if (in_port_state[i] == -1) {
                if (type != FLIT_HEAD) {
                    in_q[i].pop();
                    continue;
                }

                int dest_id = flit.range(31, 16).to_uint();
                int target_port = get_xy_route(router_id, dest_id);

                if (out_port_lock[target_port] != -1) {
                    continue;
                }

                out_port_lock[target_port] = i;
                in_port_state[i] = target_port;
            }

            int target_port = in_port_state[i];
            if (target_port >= 0 && out_port_lock[target_port] == i) {
                out_q[target_port].push(flit);
                in_q[i].pop();

                if (type == FLIT_TAIL) {
                    out_port_lock[target_port] = -1;
                    in_port_state[i] = -1;
                }
            }
        }
    }

    void drive_outputs() {
        for (int i = 0; i < PORT_COUNT; ++i) {
            if (!out_q[i].empty()) {
                out_flit[i].write(out_q[i].front());
                out_req[i].write(true);
                out_q[i].pop();
            } else {
                out_req[i].write(false);
            }
        }
    }

    void router_thread() {
        clear_queues();
        for (int i = 0; i < PORT_COUNT; ++i) {
            out_ack[i].write(true);
            out_req[i].write(false);
            out_flit[i].write(0);
        }
        wait();

        while (true) {
            sample_inputs();
            route_flits();
            drive_outputs();
            wait();
        }
    }

    SC_HAS_PROCESS(Router);

    Router(sc_module_name name) : sc_module(name), router_id(0) {
        for (int i = 0; i < PORT_COUNT; ++i) {
            in_port_state[i] = -1;
            out_port_lock[i] = -1;
        }

        SC_THREAD(router_thread);
        sensitive << clk.pos();
    }
};

#endif
