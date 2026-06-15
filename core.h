#ifndef CORE_H
#define CORE_H

#include "systemc.h"
#include "pe.h"
#include <iostream>
#include <vector>
#include <queue>

using namespace std;

#define FLIT_HEAD 0
#define FLIT_BODY 1
#define FLIT_TAIL 2

SC_MODULE( Core ) {
    sc_in<bool> rst;
    sc_in<bool> clk;

    sc_in<sc_lv<34> > flit_rx;
    sc_in<bool>       req_rx;
    sc_out<bool>      ack_rx;

    sc_out<sc_lv<34> > flit_tx;
    sc_out<bool>       req_tx;
    sc_in<bool>        ack_tx;

    int id;
    int tx_cnt;
    int rx_cnt;

    PE pe;
    PEProfile profile;

    vector<float> input_data;
    vector<float> weight_data;
    vector<float> bias_data;

    bool has_data;
    bool has_weight;
    bool has_bias;

    bool compute_started;
    bool compute_done;

    int received_input_chunks;

    queue<sc_lv<34> > tx_flit_queue;

    bool rebuilding_packet;
    int current_packet_type;
    unsigned int expected_payload_count;
    unsigned int received_payload_count;
    vector<float> current_float_payload;

    sc_lv<34> make_flit(int dest_id, int flit_kind, int packet_type, int payload);
    int float_to_fixed24(float value);
    float fixed24_to_float(int payload);

    int get_flit_kind(sc_lv<34> flit);
    int get_flit_type(sc_lv<34> flit);
    int get_flit_payload(sc_lv<34> flit);

    int input_tensor_size();
    void reset_packet_rebuild();
    void reset_local_input_buffer();
    void merge_forwarded_input_chunk(int output_start, int output_count, const vector<float>& chunk);
    void finish_packet();
    bool ready_to_compute();
    void enqueue_data_packet(int dest_id, int packet_type, const vector<float>& values);
    void enqueue_result_to_next_stage(const vector<float>& result);

    void rx_process();
    void tx_process();
    void CAL_process();

    SC_HAS_PROCESS(Core);

    Core(sc_module_name name, int core_id)
        : sc_module(name), id(core_id)
    {
        tx_cnt = 0;
        rx_cnt = 0;

        profile = PE::get_profile(core_id);

        has_data = false;
        has_weight = false;
        has_bias = false;

        compute_started = false;
        compute_done = false;

        received_input_chunks = 0;

        rebuilding_packet = false;
        current_packet_type = -1;
        expected_payload_count = 0;
        received_payload_count = 0;

        reset_local_input_buffer();

        SC_METHOD(rx_process);
        sensitive << clk.pos() << rst.pos();

        SC_METHOD(tx_process);
        sensitive << clk.pos() << rst.pos();

        SC_METHOD(CAL_process);
        sensitive << clk.pos() << rst.pos();
    }
};

#endif
