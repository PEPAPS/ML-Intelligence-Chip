#include "pe.h"

PE::PE()
{
}

// ============================================================
// Main compute dispatcher
// ============================================================
vector<float> PE::compute(
    const vector<float>& input,
    const vector<float>& weight,
    const vector<float>& bias,
    const LayerConfig& config,
    int pe_id
)
{
    if (config.op_type == OP_CONV_RELU) {
        return conv_relu(input, weight, bias, config, pe_id);
    }

    if (config.op_type == OP_MAXPOOL) {
        return maxpool(input, config, pe_id);
    }

    if (config.op_type == OP_FC_RELU) {
        return fc_relu(input, weight, bias, config, pe_id);
    }

    if (config.op_type == OP_FC_ONLY) {
        return fc_only(input, weight, bias, config, pe_id);
    }

    // cout << "[PE " << pe_id << "] ERROR: unknown op_type "
    //      << config.op_type << endl;

    return vector<float>();
}

// ============================================================
// Conv + ReLU
// ============================================================
//
// Data layout:
// input  index = c * H * W + y * W + x
// output index = oc * OH * OW + oy * OW + ox
// weight index = oc * IC * K * K + ic * K * K + ky * K + kx
// ============================================================
vector<float> PE::conv_relu(
    const vector<float>& input,
    const vector<float>& weight,
    const vector<float>& bias,
    const LayerConfig& config,
    int pe_id
)
{
    int in_w = config.in_w;
    int in_h = config.in_h;
    int in_c = config.in_c;

    int out_w = config.out_w;
    int out_h = config.out_h;
    int out_c = config.out_c;

    int kernel = config.kernel;
    int stride = config.stride;
    int padding = config.padding;

    vector<float> output(out_w * out_h * out_c, 0.0f);

    // cout << "[PE " << pe_id << "] Conv+ReLU start" << endl;

    for (int oc = 0; oc < out_c; oc++) {
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                float sum = 0.0f;

                if (oc < (int)bias.size()) {
                    sum = bias[oc];
                }

                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = 0; ky < kernel; ky++) {
                        for (int kx = 0; kx < kernel; kx++) {
                            int in_y = oy * stride + ky - padding;
                            int in_x = ox * stride + kx - padding;

                            if (in_y < 0 || in_y >= in_h ||
                                in_x < 0 || in_x >= in_w) {
                                continue;
                            }

                            int input_idx =
                                ic * in_h * in_w +
                                in_y * in_w +
                                in_x;

                            int weight_idx =
                                oc * in_c * kernel * kernel +
                                ic * kernel * kernel +
                                ky * kernel +
                                kx;

                            if (input_idx < (int)input.size() &&
                                weight_idx < (int)weight.size()) {
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }

                if (sum < 0.0f) {
                    sum = 0.0f;
                }

                int output_idx =
                    oc * out_h * out_w +
                    oy * out_w +
                    ox;

                output[output_idx] = sum;
            }
        }

        // if (DEBUG_MODE && oc % 16 == 0) {
        //     cout << "[PE " << pe_id << "] Conv output channel "
        //          << oc << " done" << endl;
        // }
    }

    // cout << "[PE " << pe_id << "] Conv+ReLU finished, output size = "
    //      << output.size() << endl;

    return output;
}

// ============================================================
// MaxPool
// ============================================================
vector<float> PE::maxpool(
    const vector<float>& input,
    const LayerConfig& config,
    int pe_id
)
{
    int in_w = config.in_w;
    int in_h = config.in_h;
    int in_c = config.in_c;

    int out_w = config.out_w;
    int out_h = config.out_h;
    int out_c = config.out_c;

    int pool = config.kernel;
    int stride = config.stride;

    vector<float> output(out_w * out_h * out_c, 0.0f);

    // cout << "[PE " << pe_id << "] MaxPool start" << endl;

    for (int c = 0; c < out_c; c++) {
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                float max_val = -1.0e30f;

                for (int ky = 0; ky < pool; ky++) {
                    for (int kx = 0; kx < pool; kx++) {
                        int in_y = oy * stride + ky;
                        int in_x = ox * stride + kx;

                        if (in_y < 0 || in_y >= in_h ||
                            in_x < 0 || in_x >= in_w) {
                            continue;
                        }

                        int input_idx =
                            c * in_h * in_w +
                            in_y * in_w +
                            in_x;

                        if (input_idx < (int)input.size()) {
                            max_val = max(max_val, input[input_idx]);
                        }
                    }
                }

                int output_idx =
                    c * out_h * out_w +
                    oy * out_w +
                    ox;

                output[output_idx] = max_val;
            }
        }
    }

    // cout << "[PE " << pe_id << "] MaxPool finished, output size = "
    //      << output.size() << endl;

    return output;
}

// ============================================================
// FC + ReLU
// ============================================================
vector<float> PE::fc_relu(
    const vector<float>& input,
    const vector<float>& weight,
    const vector<float>& bias,
    const LayerConfig& config,
    int pe_id
)
{
    int input_size = config.in_w * config.in_h * config.in_c;
    int output_size = config.out_c;

    vector<float> output(output_size, 0.0f);

    // cout << "[PE " << pe_id << "] FC+ReLU start" << endl;

    for (int o = 0; o < output_size; o++) {
        float sum = 0.0f;

        if (o < (int)bias.size()) {
            sum = bias[o];
        }

        for (int i = 0; i < input_size; i++) {
            int weight_idx = o * input_size + i;

            if (i < (int)input.size() &&
                weight_idx < (int)weight.size()) {
                sum += input[i] * weight[weight_idx];
            }
        }

        if (sum < 0.0f) {
            sum = 0.0f;
        }

        output[o] = sum;
    }

    // cout << "[PE " << pe_id << "] FC+ReLU finished, output size = "
    //      << output.size() << endl;

    return output;
}

// ============================================================
// FC only
// ============================================================
vector<float> PE::fc_only(
    const vector<float>& input,
    const vector<float>& weight,
    const vector<float>& bias,
    const LayerConfig& config,
    int pe_id
)
{
    int input_size = config.in_w * config.in_h * config.in_c;
    int output_size = config.out_c;

    vector<float> output(output_size, 0.0f);

    // cout << "[PE " << pe_id << "] FC only start" << endl;

    for (int o = 0; o < output_size; o++) {
        float sum = 0.0f;

        if (o < (int)bias.size()) {
            sum = bias[o];
        }

        for (int i = 0; i < input_size; i++) {
            int weight_idx = o * input_size + i;

            if (i < (int)input.size() &&
                weight_idx < (int)weight.size()) {
                sum += input[i] * weight[weight_idx];
            }
        }

        output[o] = sum;
    }

    // cout << "[PE " << pe_id << "] FC only finished, output size = "
    //      << output.size() << endl;

    return output;
}