#include "router.h"

// ============================================================
// New flit format:
//
// [33:32] flit kind
// [31:28] destination ID for HEAD
// [27:24] packet type for HEAD
// [23:0]  payload/count for HEAD
//
// BODY flit:
// [33:32] flit kind
// [31:0]  exact float bits / integer bits
//
// TAIL flit:
// [33:32] flit kind
// ============================================================

#define FLIT_HEAD 0
#define FLIT_BODY 1
#define FLIT_TAIL 2

int Router::get_dest_id(sc_lv<34> flit)
{
    sc_lv<4> dest_bits = flit.range(31, 28);
    return dest_bits.to_uint();
}

int Router::get_flit_kind(sc_lv<34> flit)
{
    sc_lv<2> kind_bits = flit.range(33, 32);
    return kind_bits.to_uint();
}

int Router::get_output_port(int dest_id)
{
    if (dest_id == id) {
        return LOCAL;
    }

    int curr_x = id % 4;
    int curr_y = id / 4;

    int dest_x = dest_id % 4;
    int dest_y = dest_id / 4;

    if (dest_x > curr_x) {
        return EAST;
    }

    if (dest_x < curr_x) {
        return WEST;
    }

    if (dest_y > curr_y) {
        return SOUTH;
    }

    if (dest_y < curr_y) {
        return NORTH;
    }

    return LOCAL;
}

void Router::rx_process()
{
    if (rst.read()) {
        for (int p = 0; p < 5; p++) {
            while (!input_buffer[p].empty()) {
                input_buffer[p].pop();
            }

            out_ack[p].write(false);
        }

        return;
    }

    for (int p = 0; p < 5; p++) {
        if (in_req[p].read()) {
            input_buffer[p].push(in_flit[p].read());
            out_ack[p].write(true);
        } else {
            out_ack[p].write(false);
        }
    }
}

void Router::tx_process()
{
    static bool packet_active[16][5] = {{false}};
    static int saved_out_port[16][5] = {{LOCAL}};

    if (rst.read()) {
        for (int p = 0; p < 5; p++) {
            out_flit[p].write(0);
            out_req[p].write(false);
            packet_active[id][p] = false;
            saved_out_port[id][p] = LOCAL;
        }

        return;
    }

    for (int p = 0; p < 5; p++) {
        out_req[p].write(false);
    }

    for (int in_port = 0; in_port < 5; in_port++) {
        if (!input_buffer[in_port].empty()) {
            sc_lv<34> flit = input_buffer[in_port].front();

            int flit_kind = get_flit_kind(flit);
            int out_port = LOCAL;

            if (flit_kind == FLIT_HEAD) {
                int dest_id = get_dest_id(flit);
                out_port = get_output_port(dest_id);

                packet_active[id][in_port] = true;
                saved_out_port[id][in_port] = out_port;
            }
            else {
                out_port = saved_out_port[id][in_port];
            }

            out_flit[out_port].write(flit);
            out_req[out_port].write(true);

            input_buffer[in_port].pop();

            if (flit_kind == FLIT_TAIL) {
                packet_active[id][in_port] = false;
            }

            break;
        }
    }
}