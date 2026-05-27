#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "systemc.h"
#include <vector>
#include <queue>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include "pe.h"
#include <fstream>
#include <string>
using namespace std;

// ============================================================
// Controller states
// ============================================================
#define CTRL_RESET_STATE     0
#define CTRL_REQUEST_IMAGE   1
#define CTRL_READ_IMAGE      2
#define CTRL_REQUEST_WEIGHT  3
#define CTRL_READ_WEIGHT     4
#define CTRL_REQUEST_BIAS    5
#define CTRL_READ_BIAS       6
#define CTRL_SEND_LAYER      7
#define CTRL_WAIT_RESULT     8
#define CTRL_SOFTMAX_PRINT   9
#define CTRL_DONE            10

// ============================================================
// Flit format
//
// [33:30] destination ID
// [29:28] flit kind
// [27:24] packet type
// [23:0]  payload
// ============================================================
#define FLIT_HEAD 0
#define FLIT_BODY 1
#define FLIT_TAIL 2

// ============================================================
// LayerSpec
// ============================================================
// rom_id:
//   - conv1 to conv5 use ROM layer id 1 to 5
//   - fc6 to fc8 use ROM layer id 6 to 8
//   - pooling layers do not use ROM, so rom_id = -1
// ============================================================
struct LayerSpec {
    int rom_id;
    int dest_pe;
    int op_type;

    int in_w;
    int in_h;
    int in_c;

    int out_w;
    int out_h;
    int out_c;

    int kernel;
    int stride;
    int padding;

    LayerSpec()
    {
        rom_id = -1;
        dest_pe = CTRL_ID;
        op_type = OP_MAXPOOL;

        in_w = 0;
        in_h = 0;
        in_c = 0;

        out_w = 0;
        out_h = 0;
        out_c = 0;

        kernel = 0;
        stride = 0;
        padding = 0;
    }
};

SC_MODULE( Controller ) {
    sc_in<bool> rst;
    sc_in<bool> clk;

    // ========================================================
    // To ROM
    // ========================================================
    sc_out<int>  layer_id;
    sc_out<bool> layer_id_type;   // 0 = weight, 1 = bias
    sc_out<bool> layer_id_valid;

    // ========================================================
    // From ROM
    // ========================================================
    sc_in<float> data;
    sc_in<bool>  data_valid;

    // ========================================================
    // To Router 0
    // ========================================================
    sc_out<sc_lv<34> > flit_tx;
    sc_out<bool>       req_tx;
    sc_in<bool>        ack_tx;

    // ========================================================
    // From Router 0
    // ========================================================
    sc_in<sc_lv<34> > flit_rx;
    sc_in<bool>       req_rx;
    sc_out<bool>      ack_rx;

    int id;
    int tx_cnt;
    int tx_state;

    Packet* rx_packet;
    Packet* tx_packet;

    // ========================================================
    // Controller state
    // ========================================================
    int ctrl_state;
    int layer_index;

    bool image_request_sent;
    bool weight_request_sent;
    bool bias_request_sent;
    bool layer_sent;

    bool rom_seen_valid;

    int image_count;
    int weight_count;
    int bias_count;

    // ========================================================
    // Data buffers
    // ========================================================
    vector<float> image_data;
    vector<float> padded_image;

    vector<float> current_input;
    vector<float> current_weight;
    vector<float> current_bias;
    vector<float> current_result;

    vector<LayerSpec> layer_plan;

    // ========================================================
    // TX/RX packet buffers
    // ========================================================
    queue<sc_lv<34> > tx_flit_queue;

    bool rx_rebuilding_packet;
    int rx_packet_type;
    int rx_expected_payload_count;
    int rx_received_payload_count;
    vector<float> rx_float_payload;

    // ========================================================
    // Processes
    // ========================================================
    void tx_process();
    void rx_process();
    void ROM_request();
    void ROM_receive();

    // ========================================================
    // Helper functions
    // ========================================================
    void init_layer_plan();

    LayerSpec make_layer_spec(
        int rom_id,
        int dest_pe,
        int op_type,
        int in_w,
        int in_h,
        int in_c,
        int out_w,
        int out_h,
        int out_c,
        int kernel,
        int stride,
        int padding
    );

    bool layer_needs_weight(const LayerSpec& spec);

    LayerConfig make_config_from_spec(const LayerSpec& spec);

    vector<float> pad_input_224_to_227(const vector<float>& input);

    sc_lv<34> make_flit(int dest_id, int flit_kind, int packet_type, int payload);

    int float_to_fixed24(float value);
    float fixed24_to_float(int payload);

    int get_flit_dest(sc_lv<34> flit);
    int get_flit_kind(sc_lv<34> flit);
    int get_flit_type(sc_lv<34> flit);
    int get_flit_payload(sc_lv<34> flit);

    void enqueue_config_packet(int dest_id, const LayerConfig& config);
    void enqueue_data_packet(int dest_id, int packet_type, const vector<float>& values);

    void reset_rx_packet();
    void finish_rx_packet();

    void prepare_next_layer();
    void print_softmax_top100(const vector<float>& logits);

    SC_CTOR( Controller )
    {
        id = CTRL_ID;

        tx_cnt = 0;
        tx_state = 0;

        rx_packet = NULL;
        tx_packet = NULL;

        ctrl_state = CTRL_RESET_STATE;
        layer_index = 0;

        image_request_sent = false;
        weight_request_sent = false;
        bias_request_sent = false;
        layer_sent = false;

        rom_seen_valid = false;

        image_count = 0;
        weight_count = 0;
        bias_count = 0;

        rx_rebuilding_packet = false;
        rx_packet_type = -1;
        rx_expected_payload_count = 0;
        rx_received_payload_count = 0;

        init_layer_plan();

        SC_METHOD(tx_process);
        sensitive << clk.pos() << rst.pos();

        SC_METHOD(rx_process);
        sensitive << clk.pos() << rst.pos();

        SC_METHOD(ROM_request);
        sensitive << clk.pos() << rst.pos();
        dont_initialize();

        SC_METHOD(ROM_receive);
        sensitive << clk.pos();
        dont_initialize();
    }
};

#endif