#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "systemc.h"
#include <vector>
#include <queue>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <string>
#include "pe.h"
#include "AXI4DMA.h"
#include "MemoryMap.h"

using namespace std;

// ============================================================
// Controller states for Design B with no role/setup packet.
// Controller/global SRAM is used only for input image and final output.
// Intermediate activations are forwarded PE-to-PE as PKT_RESULT chunks.
// PE mapping is defined in pe.cpp via PE::get_profile().
// ============================================================
#define CTRL_RESET_STATE       0
#define CTRL_PRELOAD_PE        1
#define CTRL_REQUEST_IMAGE     2
#define CTRL_SEND_INPUT        3
#define CTRL_WAIT_FINAL        4
#define CTRL_SOFTMAX_PRINT     5
#define CTRL_DONE              6

#define FLIT_HEAD 0
#define FLIT_BODY 1
#define FLIT_TAIL 2

SC_MODULE( Controller ) {
    sc_in<bool> rst;
    sc_in<bool> clk;

    // External controller/global-SRAM adapter port connected to router0 CTRL_PORT.
    sc_out<sc_lv<34> > flit_tx;
    sc_out<bool>       req_tx;
    sc_in<bool>        ack_tx;

    sc_in<sc_lv<34> > flit_rx;
    sc_in<bool>       req_rx;
    sc_out<bool>      ack_rx;

    AXI4DMA* dma;

    int id;
    int tx_cnt;

    int ctrl_state;
    int preload_pe_index;

    bool image_request_sent;
    bool input_sent;

    int expected_final_chunks;
    int received_final_chunks;

    // Lists are queried from PE-owned profile data. Controller does not define the mapping.
    vector<int> preload_pe_ids;
    vector<int> input_pe_ids;

    // Global SRAM behavior model: Design B uses these only for system boundary data.
    vector<float> global_input_buffer;
    vector<float> global_output_buffer;

    unsigned long long global_sram_read_words;
    unsigned long long global_sram_write_words;

    // Execution cycle counter. Counts positive clock edges after reset is deasserted.
    unsigned long long cycle_count;

    queue<sc_lv<34> > tx_flit_queue;

    bool rx_rebuilding_packet;
    int rx_packet_type;
    unsigned int rx_expected_payload_count;
    unsigned int rx_received_payload_count;
    vector<float> rx_float_payload;

    void bind_dma(AXI4DMA* d)
    {
        dma = d;
    }

    void tx_process();
    void rx_process();
    void DMA_request();

    void init_controller_plan_from_pe_profiles();

    unsigned int get_weight_base(int layer_id);
    unsigned int get_bias_base(int layer_id);

    sc_lv<34> make_flit(int dest_id, int flit_kind, int packet_type, int payload);
    int float_to_fixed24(float value);
    float fixed24_to_float(int payload);

    int get_flit_kind(sc_lv<34> flit);
    int get_flit_type(sc_lv<34> flit);
    int get_flit_payload(sc_lv<34> flit);

    void enqueue_data_packet(int dest_id, int packet_type, const vector<float>& values);
    void enqueue_preload_for_current_pe();
    void send_input_to_first_stage_group();
    void merge_final_output_chunk(int output_start, int output_count, const vector<float>& chunk);

    void reset_rx_packet();
    void finish_rx_packet();

    void print_softmax_top100(const vector<float>& logits);

    SC_CTOR( Controller )
    {
        dma = NULL;
        id = -1;
        tx_cnt = 0;
        ctrl_state = CTRL_RESET_STATE;
        preload_pe_index = 0;
        image_request_sent = false;
        input_sent = false;
        expected_final_chunks = 1;
        received_final_chunks = 0;
        global_sram_read_words = 0;
        global_sram_write_words = 0;
        cycle_count = 0;
        rx_rebuilding_packet = false;
        rx_packet_type = -1;
        rx_expected_payload_count = 0;
        rx_received_payload_count = 0;

        init_controller_plan_from_pe_profiles();

        SC_METHOD(tx_process);
        sensitive << clk.pos() << rst.pos();

        SC_METHOD(rx_process);
        sensitive << clk.pos() << rst.pos();

        SC_METHOD(DMA_request);
        sensitive << clk.pos() << rst.pos();
        dont_initialize();
    }
};

#endif
