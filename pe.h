#ifndef PE_H
#define PE_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cmath>
#include <string>
#include "systemc.h"

using namespace std;

#define TOTAL_PACKET_NUM 1

// Router/core IDs 0-15 are all compute PEs.
// The external controller/global-SRAM adapter is attached to router0 CTRL_PORT.
// A PKT_RESULT packet to destination 0 is routed by router0 to the controller.
#define CTRL_ID 0
#define MAX_NEXT_PE 4
#define NUM_COMPUTE_PE 16

// ============================================================
// Packet types.
// IMPORTANT: Controller does not send any role/setup packet in this version.
// Controller sends only weights, biases, input data, and receives result.
// ============================================================
enum PacketType {
    PKT_DATA   = 1,
    PKT_WEIGHT = 2,
    PKT_BIAS   = 3,
    PKT_RESULT = 4
};

// ============================================================
// PEProfile: fixed PE mapping for Design B.
// This is the PE's own role. It is NOT sent by controller.
// ============================================================
struct PEProfile {
    int pe_id;
    int layer_id;

    // Hardcoded PE execution style owned by this PE profile.
    bool is_conv;
    bool is_fc;
    bool use_relu;
    bool use_pool;

    int in_w;
    int in_h;
    int in_c;

    int out_w;
    int out_h;
    int out_c;

    int kernel;
    int stride;
    int padding;

    // Chunk assigned to this PE.
    int output_start;
    int output_count;
    int full_out_c;

    // For fused Conv/ReLU/Pool.
    int conv_out_w;
    int conv_out_h;
    int pool_kernel;
    int pool_stride;

    // For implicit Conv1 input padding.
    // Original input remains 224x224x3, but Conv1 address generation sees
    // an effective padded input with a top/left offset.
    int effective_in_w;
    int effective_in_h;
    int input_pad_top;
    int input_pad_left;

    // PE-to-PE forwarding.
    int expected_input_chunks;
    int next_pe_count;
    int next_pe_ids[MAX_NEXT_PE];

    bool is_first_stage;
    bool is_final_stage;

    PEProfile();
};

class PE {
public:
    PE();

    void preload_weight(const vector<float>& weight);
    void preload_bias(const vector<float>& bias);
    void clear_preload();

    bool has_weight() const;
    bool has_bias() const;

    // The actual AlexNet PE mapping is declared here and implemented in pe.cpp.
    static PEProfile get_profile(int pe_id);
    static vector<int> get_all_compute_pe_ids();
    static vector<int> get_first_stage_pe_ids();
    static int get_final_output_chunk_count();

    static bool profile_needs_weight(const PEProfile& profile);
    static int profile_input_size_words(const PEProfile& profile);
    static unsigned int profile_weight_offset_words(const PEProfile& profile);
    static unsigned int profile_weight_count_words(const PEProfile& profile);
    static string role_for_pe_static(int pe_id);

    string role_for_pe(int pe_id) const;

    vector<float> compute(
        const vector<float>& input,
        const vector<float>& weight,
        const vector<float>& bias,
        const PEProfile& profile,
        int pe_id
    );

private:
    // PE-local SRAM model for stationary model parameters.
    vector<float> local_weight;
    vector<float> local_bias;

    bool weight_preloaded;
    bool bias_preloaded;

    float read_input_value(
        const vector<float>& input,
        const PEProfile& profile,
        int ic,
        int y,
        int x
    );

    vector<float> conv_relu(
        const vector<float>& input,
        const vector<float>& weight,
        const vector<float>& bias,
        const PEProfile& profile,
        int pe_id
    );

    vector<float> conv_relu_pool(
        const vector<float>& input,
        const vector<float>& weight,
        const vector<float>& bias,
        const PEProfile& profile,
        int pe_id
    );

    vector<float> fc_relu(
        const vector<float>& input,
        const vector<float>& weight,
        const vector<float>& bias,
        const PEProfile& profile,
        int pe_id
    );

    vector<float> fc_only(
        const vector<float>& input,
        const vector<float>& weight,
        const vector<float>& bias,
        const PEProfile& profile,
        int pe_id
    );
};

#endif
