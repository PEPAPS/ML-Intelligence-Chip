#include "router.h"

// Flit format:
// HEAD [33:32] kind, [31:28] destination ID, [27:24] packet type, [23:0] payload/count
// BODY [33:32] kind, [31:0] exact float/integer bits
// TAIL [33:32] kind

#define FLIT_HEAD 0
#define FLIT_BODY 1
#define FLIT_TAIL 2
#define PKT_RESULT_FOR_ROUTER 4

int Router::get_dest_id(sc_lv<34> flit)
{
    sc_lv<4> dest_bits = flit.range(31, 28);
    return dest_bits.to_uint();
}

int Router::get_packet_type(sc_lv<34> flit)
{
    sc_lv<4> type_bits = flit.range(27, 24);
    return type_bits.to_uint();
}

int Router::get_flit_kind(sc_lv<34> flit)
{
    sc_lv<2> kind_bits = flit.range(33, 32);
    return kind_bits.to_uint();
}

int Router::get_output_port(int dest_id, int packet_type)
{
    // Optimized architecture: router/core IDs 0-15 are all compute PEs.
    // The external controller/global-SRAM adapter is attached to router 0's CTRL_PORT.
    // A result packet to destination 0 is interpreted by router 0 as a packet for
    // the external controller, not for Core 0. Non-result packets to destination 0
    // still go to Core 0 through LOCAL.
    if (id == 0 && dest_id == 0 && packet_type == PKT_RESULT_FOR_ROUTER) {
        return CTRL_PORT;
    }

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
        for (int p = 0; p < ROUTER_PORTS; p++) {
            while (!input_buffer[p].empty()) {
                input_buffer[p].pop();
            }

            out_ack[p].write(false);
        }

        return;
    }

    for (int p = 0; p < ROUTER_PORTS; p++) {
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
    if (rst.read()) {
        for (int p = 0; p < ROUTER_PORTS; p++) {
            out_flit[p].write(0);
            out_req[p].write(false);
            out_port_busy[p] = false;
            out_port_owner[p] = -1;
        }
        return;
    }

    for (int p = 0; p < ROUTER_PORTS; p++) {
        out_req[p].write(false);
    }

    // ------------------------------------------------------------
    // 1) Continue an already-started packet first.
    //    This is the important fix for Design B PE-to-PE forwarding:
    //    a destination core expects packet flits to arrive as:
    //        HEAD, BODY..., TAIL
    //    If two packets going to the same LOCAL output are interleaved,
    //    the core can see BODY before HEAD. Therefore each output port
    //    is locked from HEAD until TAIL.
    // ------------------------------------------------------------
    for (int out_port = 0; out_port < ROUTER_PORTS; out_port++) {
        if (!out_port_busy[out_port]) {
            continue;
        }

        int in_port = out_port_owner[out_port];
        if (in_port < 0 || in_port >= ROUTER_PORTS) {
            out_port_busy[out_port] = false;
            out_port_owner[out_port] = -1;
            continue;
        }

        if (!input_buffer[in_port].empty()) {
            sc_lv<34> flit = input_buffer[in_port].front();
            int flit_kind = get_flit_kind(flit);

            out_flit[out_port].write(flit);
            out_req[out_port].write(true);
            input_buffer[in_port].pop();

            if (flit_kind == FLIT_TAIL) {
                out_port_busy[out_port] = false;
                out_port_owner[out_port] = -1;
            }

            // This router model sends one flit per cycle.
            return;
        }
    }

    // ------------------------------------------------------------
    // 2) Start a new packet only from a HEAD flit, and lock its
    //    selected output port until TAIL.
    // ------------------------------------------------------------
    for (int in_port = 0; in_port < ROUTER_PORTS; in_port++) {
        if (input_buffer[in_port].empty()) {
            continue;
        }

        sc_lv<34> flit = input_buffer[in_port].front();
        int flit_kind = get_flit_kind(flit);

        // A BODY/TAIL with no active packet lock means the stream is already
        // corrupted. Hold it rather than forwarding it into a core and causing
        // BODY-before-HEAD spam.
        if (flit_kind != FLIT_HEAD) {
            continue;
        }

        int dest_id = get_dest_id(flit);
        int packet_type = get_packet_type(flit);
        int out_port = get_output_port(dest_id, packet_type);

        if (out_port < 0 || out_port >= ROUTER_PORTS) {
            continue;
        }

        if (out_port_busy[out_port]) {
            continue;
        }

        out_port_busy[out_port] = true;
        out_port_owner[out_port] = in_port;

        out_flit[out_port].write(flit);
        out_req[out_port].write(true);
        input_buffer[in_port].pop();

        // HEAD-only packets are not used here, but keep the guard correct.
        if (flit_kind == FLIT_TAIL) {
            out_port_busy[out_port] = false;
            out_port_owner[out_port] = -1;
        }

        return;
    }
}
