#ifndef CORE_H
#define CORE_H

#include "systemc.h"
#include "pe.h"

#include <iostream>
#include <vector>
#include <queue>
#include <cmath>

using namespace std;

// ============================================================
// Same flit format as controller/router
// ============================================================
//
// [33:30] destination ID
// [29:28] flit kind
// [27:24] packet type
// [23:0]  payload
// ============================================================

#ifndef FLIT_HEAD
#define FLIT_HEAD 0
#endif

#ifndef FLIT_BODY
#define FLIT_BODY 1
#endif

#ifndef FLIT_TAIL
#define FLIT_TAIL 2
#endif

SC_MODULE(Core) {
    sc_in<bool> rst;
    sc_in<bool> clk;

    // ========================================================
    // Router -> Core
    // ========================================================
    sc_in<sc_lv<34> > flit_rx;
    sc_in<bool>       req_rx;
    sc_out<bool>      ack_rx;

    // ========================================================
    // Core -> Router
    // ========================================================
    sc_out<sc_lv<34> > flit_tx;
    sc_out<bool>       req_tx;
    sc_in<bool>        ack_tx;

    // ========================================================
    // Core ID
    // ========================================================
    int id;

    // Local Processing Element
    PE pe;

    // ========================================================
    // Original fields
    // ========================================================
    int tx_cnt;
    int rx_cnt;
    int tx_state;

    Packet* tx_packet;
    Packet* rx_packet;

    // ========================================================
    // Packet rebuild state
    // ========================================================
    bool rebuilding_packet;

    int current_packet_type;
    int expected_payload_count;
    int received_payload_count;

    vector<int> current_config_fields;
    vector<float> current_float_payload;

    // ========================================================
    // Stored PE-side data
    // ========================================================
    LayerConfig current_config;

    vector<float> input_data;
    vector<float> weight_data;
    vector<float> bias_data;

    bool has_config;
    bool has_data;
    bool has_weight;
    bool has_bias;

    bool compute_started;
    bool compute_done;
    bool waiting_printed;

    // ========================================================
    // TX flit queue
    // ========================================================
    queue<sc_lv<34> > tx_flit_queue;

    // ========================================================
    // Processes
    // ========================================================
    void rx_process();
    void tx_process();
    void CAL_process();

    // ========================================================
    // Helper functions
    // ========================================================
    int get_flit_dest(sc_lv<34> flit);
    int get_flit_kind(sc_lv<34> flit);
    int get_flit_type(sc_lv<34> flit);
    int get_flit_payload(sc_lv<34> flit);

    float fixed24_to_float(int payload);
    int float_to_fixed24(float value);

    sc_lv<34> make_flit(int dest_id, int flit_kind, int packet_type, int payload);

    void reset_packet_rebuild();
    void finish_packet();

    void apply_config_fields(const vector<int>& fields);
    bool ready_to_compute();

    void enqueue_data_packet(int dest_id, int packet_type, const vector<float>& values);

    SC_HAS_PROCESS(Core);

    Core(sc_module_name name, int core_id)
        : sc_module(name), id(core_id), pe()
    {
        tx_cnt = 0;
        rx_cnt = 0;
        tx_state = 0;

        tx_packet = NULL;
        rx_packet = NULL;

        rebuilding_packet = false;

        current_packet_type = -1;
        expected_payload_count = 0;
        received_payload_count = 0;

        has_config = false;
        has_data = false;
        has_weight = false;
        has_bias = false;

        compute_started = false;
        compute_done = false;
        waiting_printed = false;

        SC_METHOD(rx_process);
        sensitive << clk.pos() << rst.pos();

        SC_METHOD(tx_process);
        sensitive << clk.pos() << rst.pos();

        SC_METHOD(CAL_process);
        sensitive << clk.pos() << rst.pos();
    }
};

#endif