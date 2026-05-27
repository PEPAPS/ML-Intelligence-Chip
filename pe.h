#ifndef PE_H
#define PE_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cmath>
#include "systemc.h"

using namespace std;

// #define DEBUG_MODE false
#define TOTAL_PACKET_NUM 1

// ============================================================
// Router/Core/PE ID mapping
// ============================================================
#define CTRL_ID        0
#define PE_CONV_RELU   1
#define PE_MAXPOOL     2
#define PE_FC_RELU     3
#define PE_FC8         4

// ============================================================
// Packet type
// ============================================================
enum PacketType {
    PKT_CONFIG = 0,
    PKT_DATA   = 1,
    PKT_WEIGHT = 2,
    PKT_BIAS   = 3,
    PKT_RESULT = 4
};

// ============================================================
// Operation type
// ============================================================
enum OpType {
    OP_CONV_RELU = 0,
    OP_MAXPOOL   = 1,
    OP_FC_RELU   = 2,
    OP_FC_ONLY   = 3
};

// ============================================================
// Layer configuration
// ============================================================
struct LayerConfig {
    int layer_id;
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

    int next_dest_id;

    LayerConfig()
    {
        layer_id = -1;
        op_type = -1;

        in_w = 0;
        in_h = 0;
        in_c = 0;

        out_w = 0;
        out_h = 0;
        out_c = 0;

        kernel = 0;
        stride = 0;
        padding = 0;

        next_dest_id = CTRL_ID;
    }
};

// ============================================================
// Packet format
// ============================================================
struct Packet {
    int source_id;
    int dest_id;
    int packet_type;

    LayerConfig config;
    vector<float> datas;

    Packet()
    {
        source_id = CTRL_ID;
        dest_id = CTRL_ID;
        packet_type = PKT_DATA;
    }
};

// ============================================================
// PE class
// ============================================================
// This is the actual Processing Element.
// Core calls this class after it rebuilds packets from flits.
// ============================================================
class PE {
public:
    PE();

    vector<float> compute(
        const vector<float>& input,
        const vector<float>& weight,
        const vector<float>& bias,
        const LayerConfig& config,
        int pe_id
    );

private:
    vector<float> conv_relu(
        const vector<float>& input,
        const vector<float>& weight,
        const vector<float>& bias,
        const LayerConfig& config,
        int pe_id
    );

    vector<float> maxpool(
        const vector<float>& input,
        const LayerConfig& config,
        int pe_id
    );

    vector<float> fc_relu(
        const vector<float>& input,
        const vector<float>& weight,
        const vector<float>& bias,
        const LayerConfig& config,
        int pe_id
    );

    vector<float> fc_only(
        const vector<float>& input,
        const vector<float>& weight,
        const vector<float>& bias,
        const LayerConfig& config,
        int pe_id
    );
};

#endif