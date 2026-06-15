#include "pe.h"

PEProfile::PEProfile()
{
    pe_id = -1;
    layer_id = -1;

    is_conv = false;
    is_fc = false;
    use_relu = false;
    use_pool = false;

    in_w = 0;
    in_h = 0;
    in_c = 0;

    out_w = 0;
    out_h = 0;
    out_c = 0;

    kernel = 0;
    stride = 0;
    padding = 0;

    output_start = 0;
    output_count = 0;
    full_out_c = 0;

    conv_out_w = 0;
    conv_out_h = 0;
    pool_kernel = 0;
    pool_stride = 0;

    effective_in_w = 0;
    effective_in_h = 0;
    input_pad_top = 0;
    input_pad_left = 0;

    expected_input_chunks = 1;
    next_pe_count = 0;
    for (int i = 0; i < MAX_NEXT_PE; i++) {
        next_pe_ids[i] = -1;
    }

    is_first_stage = false;
    is_final_stage = false;
}

PE::PE()
{
    weight_preloaded = false;
    bias_preloaded = false;
}

void PE::preload_weight(const vector<float>& weight)
{
    local_weight = weight;
    weight_preloaded = true;
}

void PE::preload_bias(const vector<float>& bias)
{
    local_bias = bias;
    bias_preloaded = true;
}

void PE::clear_preload()
{
    local_weight.clear();
    local_bias.clear();
    weight_preloaded = false;
    bias_preloaded = false;
}

bool PE::has_weight() const
{
    return weight_preloaded;
}

bool PE::has_bias() const
{
    return bias_preloaded;
}

// ============================================================
// Fixed 16-PE AlexNet mapping.
// This is intentionally defined in pe.cpp, not in controller.cpp.
// Controller never sends a role/setup packet in this version.
//
// PE0-PE1    Conv1 + ReLU + Pool1, split by output filters
// PE2-PE5    Conv2 + ReLU + Pool2, split by output filters
// PE6-PE7    Conv3 + ReLU, split by output filters
// PE8-PE9    Conv4 + ReLU, split by output filters
// PE10-PE11  Conv5 + ReLU + Pool5, split by output filters
// PE12-PE13  FC6 + ReLU, split by output neurons
// PE14       FC7 + ReLU
// PE15       FC8 logits
// ============================================================
static void set_next_list(PEProfile& p, const int* ids, int count)
{
    p.next_pe_count = count;
    for (int i = 0; i < MAX_NEXT_PE; i++) {
        p.next_pe_ids[i] = -1;
    }
    for (int i = 0; i < count && i < MAX_NEXT_PE; i++) {
        p.next_pe_ids[i] = ids[i];
    }
}

static PEProfile make_profile_base(
    int pe_id,
    int layer_id,
    bool is_conv,
    bool is_fc,
    bool use_relu,
    bool use_pool,
    int in_w,
    int in_h,
    int in_c,
    int out_w,
    int out_h,
    int out_c,
    int kernel,
    int stride,
    int padding,
    int output_start,
    int output_count,
    int conv_out_w,
    int conv_out_h,
    int pool_kernel,
    int pool_stride,
    int expected_input_chunks,
    const int* next_ids,
    int next_count,
    bool first_stage,
    bool final_stage
)
{
    PEProfile p;

    p.pe_id = pe_id;
    p.layer_id = layer_id;
    p.is_conv = is_conv;
    p.is_fc = is_fc;
    p.use_relu = use_relu;
    p.use_pool = use_pool;

    p.in_w = in_w;
    p.in_h = in_h;
    p.in_c = in_c;

    p.out_w = out_w;
    p.out_h = out_h;
    p.out_c = out_c;

    p.kernel = kernel;
    p.stride = stride;
    p.padding = padding;

    p.output_start = output_start;
    p.output_count = output_count;
    p.full_out_c = out_c;

    p.conv_out_w = conv_out_w;
    p.conv_out_h = conv_out_h;
    p.pool_kernel = pool_kernel;
    p.pool_stride = pool_stride;

    p.expected_input_chunks = expected_input_chunks;
    set_next_list(p, next_ids, next_count);

    p.is_first_stage = first_stage;
    p.is_final_stage = final_stage;

    return p;
}

PEProfile PE::get_profile(int pe_id)
{
    static const int NEXT_CONV2[4] = {2, 3, 4, 5};
    static const int NEXT_CONV3[2] = {6, 7};
    static const int NEXT_CONV4[2] = {8, 9};
    static const int NEXT_CONV5[2] = {10, 11};
    static const int NEXT_FC6[2]   = {12, 13};
    static const int NEXT_FC7[1]   = {14};
    static const int NEXT_FC8[1]   = {15};
    static const int NEXT_NONE[1]  = {-1};

    switch (pe_id) {
        // Conv1: 64 filters split into 32 + 32.
        // Input is stored as 224x224x3. PE address logic performs implicit
        // zero-padding so Conv1 sees the same effective padded input as baseline.
        case 0: {
            PEProfile p = make_profile_base(0, 1, true, false, true, true,
                224, 224, 3, 27, 27, 64, 11, 4, 0,
                0, 32, 55, 55, 3, 2, 1, NEXT_CONV2, 4, true, false);
            p.effective_in_w = 227;
            p.effective_in_h = 227;
            p.input_pad_top = 2;
            p.input_pad_left = 2;
            return p;
        }
        case 1: {
            PEProfile p = make_profile_base(1, 1, true, false, true, true,
                224, 224, 3, 27, 27, 64, 11, 4, 0,
                32, 32, 55, 55, 3, 2, 1, NEXT_CONV2, 4, true, false);
            p.effective_in_w = 227;
            p.effective_in_h = 227;
            p.input_pad_top = 2;
            p.input_pad_left = 2;
            return p;
        }

        // Conv2: 192 filters split into 4 chunks of 48.
        // Each Conv2 PE waits for both Conv1 chunks.
        case 2: return make_profile_base(2, 2, true, false, true, true, 27, 27, 64, 13, 13, 192, 5, 1, 2,   0, 48, 27, 27, 3, 2, 2, NEXT_CONV3, 2, false, false);
        case 3: return make_profile_base(3, 2, true, false, true, true, 27, 27, 64, 13, 13, 192, 5, 1, 2,  48, 48, 27, 27, 3, 2, 2, NEXT_CONV3, 2, false, false);
        case 4: return make_profile_base(4, 2, true, false, true, true, 27, 27, 64, 13, 13, 192, 5, 1, 2,  96, 48, 27, 27, 3, 2, 2, NEXT_CONV3, 2, false, false);
        case 5: return make_profile_base(5, 2, true, false, true, true, 27, 27, 64, 13, 13, 192, 5, 1, 2, 144, 48, 27, 27, 3, 2, 2, NEXT_CONV3, 2, false, false);

        // Conv3: 384 filters split into 192 + 192. Each waits for 4 Conv2 chunks.
        case 6: return make_profile_base(6, 3, true, false, true, false, 13, 13, 192, 13, 13, 384, 3, 1, 1,   0, 192, 0, 0, 0, 0, 4, NEXT_CONV4, 2, false, false);
        case 7: return make_profile_base(7, 3, true, false, true, false, 13, 13, 192, 13, 13, 384, 3, 1, 1, 192, 192, 0, 0, 0, 0, 4, NEXT_CONV4, 2, false, false);

        // Conv4: 256 filters split into 128 + 128. Each waits for 2 Conv3 chunks.
        case 8: return make_profile_base(8, 4, true, false, true, false, 13, 13, 384, 13, 13, 256, 3, 1, 1,   0, 128, 0, 0, 0, 0, 2, NEXT_CONV5, 2, false, false);
        case 9: return make_profile_base(9, 4, true, false, true, false, 13, 13, 384, 13, 13, 256, 3, 1, 1, 128, 128, 0, 0, 0, 0, 2, NEXT_CONV5, 2, false, false);

        // Conv5: 256 filters split into 128 + 128, with local ReLU+Pool5.
        case 10: return make_profile_base(10, 5, true, false, true, true, 13, 13, 256, 6, 6, 256, 3, 1, 1,   0, 128, 13, 13, 3, 2, 2, NEXT_FC6, 2, false, false);
        case 11: return make_profile_base(11, 5, true, false, true, true, 13, 13, 256, 6, 6, 256, 3, 1, 1, 128, 128, 13, 13, 3, 2, 2, NEXT_FC6, 2, false, false);

        // FC6: 4096 neurons split into 2048 + 2048. Each waits for both Conv5 chunks.
        case 12: return make_profile_base(12, 6, false, true, true, false, 1, 1, 9216, 1, 1, 4096, 1, 1, 0,    0, 2048, 0, 0, 0, 0, 2, NEXT_FC7, 1, false, false);
        case 13: return make_profile_base(13, 6, false, true, true, false, 1, 1, 9216, 1, 1, 4096, 1, 1, 0, 2048, 2048, 0, 0, 0, 0, 2, NEXT_FC7, 1, false, false);

        // FC7 waits for both FC6 chunks. FC8 waits for the FC7 chunk.
        case 14: return make_profile_base(14, 7, false, true, true, false, 1, 1, 4096, 1, 1, 4096, 1, 1, 0, 0, 4096, 0, 0, 0, 0, 2, NEXT_FC8, 1, false, false);
        case 15: return make_profile_base(15, 8, false, true, false, false, 1, 1, 4096, 1, 1, 1000, 1, 1, 0, 0, 1000, 0, 0, 0, 0, 1, NEXT_NONE, 0, false, true);

        default:
            return PEProfile();
    }
}

vector<int> PE::get_all_compute_pe_ids()
{
    vector<int> ids;
    for (int i = 0; i < NUM_COMPUTE_PE; i++) {
        ids.push_back(i);
    }
    return ids;
}

vector<int> PE::get_first_stage_pe_ids()
{
    vector<int> ids;
    for (int i = 0; i < NUM_COMPUTE_PE; i++) {
        PEProfile p = PE::get_profile(i);
        if (p.is_first_stage) {
            ids.push_back(i);
        }
    }
    return ids;
}

int PE::get_final_output_chunk_count()
{
    int count = 0;
    for (int i = 0; i < NUM_COMPUTE_PE; i++) {
        PEProfile p = PE::get_profile(i);
        if (p.is_final_stage) {
            count++;
        }
    }
    return count > 0 ? count : 1;
}

bool PE::profile_needs_weight(const PEProfile& p)
{
    return p.is_conv || p.is_fc;
}

int PE::profile_input_size_words(const PEProfile& p)
{
    return p.in_w * p.in_h * p.in_c;
}

unsigned int PE::profile_weight_offset_words(const PEProfile& p)
{
    if (p.is_conv) {
        return p.output_start * p.in_c * p.kernel * p.kernel;
    }

    if (p.is_fc) {
        return p.output_start * profile_input_size_words(p);
    }

    return 0;
}

unsigned int PE::profile_weight_count_words(const PEProfile& p)
{
    if (p.is_conv) {
        return p.output_count * p.in_c * p.kernel * p.kernel;
    }

    if (p.is_fc) {
        return p.output_count * profile_input_size_words(p);
    }

    return 0;
}

string PE::role_for_pe_static(int pe_id)
{
    switch (pe_id) {
        case 0:  return "PE0  Conv1+ReLU+Pool1, output channels 0-31";
        case 1:  return "PE1  Conv1+ReLU+Pool1, output channels 32-63";
        case 2:  return "PE2  Conv2+ReLU+Pool2, output channels 0-47";
        case 3:  return "PE3  Conv2+ReLU+Pool2, output channels 48-95";
        case 4:  return "PE4  Conv2+ReLU+Pool2, output channels 96-143";
        case 5:  return "PE5  Conv2+ReLU+Pool2, output channels 144-191";
        case 6:  return "PE6  Conv3+ReLU, output channels 0-191";
        case 7:  return "PE7  Conv3+ReLU, output channels 192-383";
        case 8:  return "PE8  Conv4+ReLU, output channels 0-127";
        case 9:  return "PE9  Conv4+ReLU, output channels 128-255";
        case 10: return "PE10 Conv5+ReLU+Pool5, output channels 0-127";
        case 11: return "PE11 Conv5+ReLU+Pool5, output channels 128-255";
        case 12: return "PE12 FC6+ReLU, output neurons 0-2047";
        case 13: return "PE13 FC6+ReLU, output neurons 2048-4095";
        case 14: return "PE14 FC7+ReLU, output neurons 0-4095";
        case 15: return "PE15 FC8 logits, output neurons 0-999";
        default: return "Invalid PE";
    }
}

string PE::role_for_pe(int pe_id) const
{
    return PE::role_for_pe_static(pe_id);
}

vector<float> PE::compute(
    const vector<float>& input,
    const vector<float>& weight,
    const vector<float>& bias,
    const PEProfile& profile,
    int pe_id
)
{
    const vector<float>& effective_weight =
        weight.empty() && weight_preloaded ? local_weight : weight;

    const vector<float>& effective_bias =
        bias.empty() && bias_preloaded ? local_bias : bias;

    cout << "[PE " << pe_id << "] " << role_for_pe(pe_id)
         << " | layer=" << profile.layer_id
         << " start=" << profile.output_start
         << " count=" << profile.output_count << endl;

    if (profile.is_conv && profile.use_pool) {
        return conv_relu_pool(input, effective_weight, effective_bias, profile, pe_id);
    }

    if (profile.is_conv) {
        return conv_relu(input, effective_weight, effective_bias, profile, pe_id);
    }

    if (profile.is_fc && profile.use_relu) {
        return fc_relu(input, effective_weight, effective_bias, profile, pe_id);
    }

    if (profile.is_fc) {
        return fc_only(input, effective_weight, effective_bias, profile, pe_id);
    }

    return vector<float>();
}

float PE::read_input_value(
    const vector<float>& input,
    const PEProfile& profile,
    int ic,
    int y,
    int x
)
{
    int real_y = y;
    int real_x = x;

    if (profile.effective_in_w > 0 && profile.effective_in_h > 0) {
        real_y = y - profile.input_pad_top;
        real_x = x - profile.input_pad_left;
    }

    if (real_y < 0 || real_y >= profile.in_h ||
        real_x < 0 || real_x >= profile.in_w ||
        ic < 0 || ic >= profile.in_c) {
        return 0.0f;
    }

    int input_idx = ic * profile.in_h * profile.in_w + real_y * profile.in_w + real_x;

    if (input_idx < 0 || input_idx >= (int)input.size()) {
        return 0.0f;
    }

    return input[input_idx];
}

vector<float> PE::conv_relu(
    const vector<float>& input,
    const vector<float>& weight,
    const vector<float>& bias,
    const PEProfile& profile,
    int pe_id
)
{
    int in_c = profile.in_c;
    int out_w = profile.out_w;
    int out_h = profile.out_h;
    int output_count = profile.output_count > 0 ? profile.output_count : profile.out_c;
    int kernel = profile.kernel;
    int stride = profile.stride;
    int padding = profile.padding;

    vector<float> output(out_w * out_h * output_count, 0.0f);

    for (int local_oc = 0; local_oc < output_count; local_oc++) {
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                float sum = 0.0f;

                if (local_oc < (int)bias.size()) {
                    sum = bias[local_oc];
                }

                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = 0; ky < kernel; ky++) {
                        for (int kx = 0; kx < kernel; kx++) {
                            int in_y = oy * stride + ky - padding;
                            int in_x = ox * stride + kx - padding;
                            float in_val = read_input_value(input, profile, ic, in_y, in_x);

                            int weight_idx =
                                local_oc * in_c * kernel * kernel +
                                ic * kernel * kernel +
                                ky * kernel +
                                kx;

                            if (weight_idx < (int)weight.size()) {
                                sum += in_val * weight[weight_idx];
                            }
                        }
                    }
                }

                if (sum < 0.0f) {
                    sum = 0.0f;
                }

                int output_idx = local_oc * out_h * out_w + oy * out_w + ox;
                output[output_idx] = sum;
            }
        }
    }

    return output;
}

vector<float> PE::conv_relu_pool(
    const vector<float>& input,
    const vector<float>& weight,
    const vector<float>& bias,
    const PEProfile& profile,
    int pe_id
)
{
    PEProfile conv_profile = profile;
    conv_profile.out_w = profile.conv_out_w;
    conv_profile.out_h = profile.conv_out_h;
    conv_profile.out_c = profile.out_c;

    vector<float> conv_output = conv_relu(input, weight, bias, conv_profile, pe_id);

    PEProfile pool_profile;
    pool_profile.in_w = profile.conv_out_w;
    pool_profile.in_h = profile.conv_out_h;
    pool_profile.in_c = profile.output_count > 0 ? profile.output_count : profile.out_c;
    pool_profile.out_w = profile.out_w;
    pool_profile.out_h = profile.out_h;
    pool_profile.out_c = pool_profile.in_c;
    pool_profile.kernel = profile.pool_kernel;
    pool_profile.stride = profile.pool_stride;
    pool_profile.padding = 0;
    pool_profile.output_start = 0;
    pool_profile.output_count = pool_profile.in_c;
    pool_profile.full_out_c = pool_profile.in_c;

    int in_w = pool_profile.in_w;
    int in_h = pool_profile.in_h;
    int in_c = pool_profile.in_c;
    int out_w = pool_profile.out_w;
    int out_h = pool_profile.out_h;
    int pool = pool_profile.kernel;
    int pool_stride = pool_profile.stride;

    vector<float> pooled_output(out_w * out_h * in_c, 0.0f);

    for (int c = 0; c < in_c; c++) {
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                float max_val = -1.0e30f;

                for (int ky = 0; ky < pool; ky++) {
                    for (int kx = 0; kx < pool; kx++) {
                        int in_y = oy * pool_stride + ky;
                        int in_x = ox * pool_stride + kx;

                        if (in_y < 0 || in_y >= in_h || in_x < 0 || in_x >= in_w) {
                            continue;
                        }

                        int input_idx = c * in_h * in_w + in_y * in_w + in_x;
                        if (input_idx < (int)conv_output.size()) {
                            max_val = max(max_val, conv_output[input_idx]);
                        }
                    }
                }

                int output_idx = c * out_h * out_w + oy * out_w + ox;
                pooled_output[output_idx] = max_val;
            }
        }
    }

    return pooled_output;
}

vector<float> PE::fc_relu(
    const vector<float>& input,
    const vector<float>& weight,
    const vector<float>& bias,
    const PEProfile& profile,
    int pe_id
)
{
    int input_size = profile.in_w * profile.in_h * profile.in_c;
    int output_size = profile.output_count > 0 ? profile.output_count : profile.out_c;
    vector<float> output(output_size, 0.0f);

    for (int o = 0; o < output_size; o++) {
        float sum = 0.0f;

        if (o < (int)bias.size()) {
            sum = bias[o];
        }

        for (int i = 0; i < input_size; i++) {
            int weight_idx = o * input_size + i;
            if (i < (int)input.size() && weight_idx < (int)weight.size()) {
                sum += input[i] * weight[weight_idx];
            }
        }

        if (sum < 0.0f) {
            sum = 0.0f;
        }

        output[o] = sum;
    }

    return output;
}

vector<float> PE::fc_only(
    const vector<float>& input,
    const vector<float>& weight,
    const vector<float>& bias,
    const PEProfile& profile,
    int pe_id
)
{
    int input_size = profile.in_w * profile.in_h * profile.in_c;
    int output_size = profile.output_count > 0 ? profile.output_count : profile.out_c;
    vector<float> output(output_size, 0.0f);

    for (int o = 0; o < output_size; o++) {
        float sum = 0.0f;

        if (o < (int)bias.size()) {
            sum = bias[o];
        }

        for (int i = 0; i < input_size; i++) {
            int weight_idx = o * input_size + i;
            if (i < (int)input.size() && weight_idx < (int)weight.size()) {
                sum += input[i] * weight[weight_idx];
            }
        }

        output[o] = sum;
    }

    return output;
}
