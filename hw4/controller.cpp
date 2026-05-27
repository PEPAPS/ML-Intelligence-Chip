#include "controller.h"

// ============================================================
// Layer plan initialization
// ============================================================
LayerSpec Controller::make_layer_spec(
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
)
{
    LayerSpec spec;

    spec.rom_id = rom_id;
    spec.dest_pe = dest_pe;
    spec.op_type = op_type;

    spec.in_w = in_w;
    spec.in_h = in_h;
    spec.in_c = in_c;

    spec.out_w = out_w;
    spec.out_h = out_h;
    spec.out_c = out_c;

    spec.kernel = kernel;
    spec.stride = stride;
    spec.padding = padding;

    return spec;
}

void Controller::init_layer_plan()
{
    layer_plan.clear();

    // Layer 1: Conv1 + ReLU
    // Input is manually padded from 224x224x3 to 227x227x3 before this.
    layer_plan.push_back(make_layer_spec(
        1, PE_CONV_RELU, OP_CONV_RELU,
        227, 227, 3,
        55, 55, 64,
        11, 4, 0
    ));

    // Layer 1b: MaxPool1
    layer_plan.push_back(make_layer_spec(
        -1, PE_MAXPOOL, OP_MAXPOOL,
        55, 55, 64,
        27, 27, 64,
        3, 2, 0
    ));

    // Layer 2: Conv2 + ReLU
    layer_plan.push_back(make_layer_spec(
        2, PE_CONV_RELU, OP_CONV_RELU,
        27, 27, 64,
        27, 27, 192,
        5, 1, 2
    ));

    // Layer 2b: MaxPool2
    layer_plan.push_back(make_layer_spec(
        -1, PE_MAXPOOL, OP_MAXPOOL,
        27, 27, 192,
        13, 13, 192,
        3, 2, 0
    ));

    // Layer 3: Conv3 + ReLU
    layer_plan.push_back(make_layer_spec(
        3, PE_CONV_RELU, OP_CONV_RELU,
        13, 13, 192,
        13, 13, 384,
        3, 1, 1
    ));

    // Layer 4: Conv4 + ReLU
    layer_plan.push_back(make_layer_spec(
        4, PE_CONV_RELU, OP_CONV_RELU,
        13, 13, 384,
        13, 13, 256,
        3, 1, 1
    ));

    // Layer 5: Conv5 + ReLU
    layer_plan.push_back(make_layer_spec(
        5, PE_CONV_RELU, OP_CONV_RELU,
        13, 13, 256,
        13, 13, 256,
        3, 1, 1
    ));

    // Layer 5b: MaxPool5
    layer_plan.push_back(make_layer_spec(
        -1, PE_MAXPOOL, OP_MAXPOOL,
        13, 13, 256,
        6, 6, 256,
        3, 2, 0
    ));

    // Layer 6: FC6 + ReLU
    layer_plan.push_back(make_layer_spec(
        6, PE_FC_RELU, OP_FC_RELU,
        1, 1, 9216,
        1, 1, 4096,
        1, 1, 0
    ));

    // Layer 7: FC7 + ReLU
    layer_plan.push_back(make_layer_spec(
        7, PE_FC_RELU, OP_FC_RELU,
        1, 1, 4096,
        1, 1, 4096,
        1, 1, 0
    ));

    // Layer 8: FC8
    layer_plan.push_back(make_layer_spec(
        8, PE_FC8, OP_FC_ONLY,
        1, 1, 4096,
        1, 1, 1000,
        1, 1, 0
    ));
}

bool Controller::layer_needs_weight(const LayerSpec& spec)
{
    return spec.op_type == OP_CONV_RELU ||
           spec.op_type == OP_FC_RELU ||
           spec.op_type == OP_FC_ONLY;
}

// ============================================================
// Convert LayerSpec to LayerConfig
// ============================================================
LayerConfig Controller::make_config_from_spec(const LayerSpec& spec)
{
    LayerConfig config;

    config.layer_id = spec.rom_id;
    config.op_type = spec.op_type;

    config.in_w = spec.in_w;
    config.in_h = spec.in_h;
    config.in_c = spec.in_c;

    config.out_w = spec.out_w;
    config.out_h = spec.out_h;
    config.out_c = spec.out_c;

    config.kernel = spec.kernel;
    config.stride = spec.stride;
    config.padding = spec.padding;

    // Controller-orchestrated design:
    // every PE sends result back to Controller.
    config.next_dest_id = CTRL_ID;

    return config;
}

// ============================================================
// Exact float-bit conversion
// ============================================================
int Controller::float_to_fixed24(float value)
{
    union {
        float f;
        unsigned int u;
    } converter;

    converter.f = value;
    return (int)converter.u;
}

float Controller::fixed24_to_float(int payload)
{
    union {
        float f;
        unsigned int u;
    } converter;

    converter.u = (unsigned int)payload;
    return converter.f;
}

// ============================================================
// New flit helpers
// ============================================================
sc_lv<34> Controller::make_flit(
    int dest_id,
    int flit_kind,
    int packet_type,
    int payload
)
{
    sc_lv<34> flit;
    flit = 0;

    sc_uint<2> kind_bits = flit_kind & 0x3;

    flit.range(33, 32) = kind_bits;

    if (flit_kind == FLIT_HEAD) {
        sc_uint<4> dest_bits = dest_id & 0xF;
        sc_uint<4> type_bits = packet_type & 0xF;
        sc_uint<24> payload_bits = payload & 0x00FFFFFF;

        flit.range(31, 28) = dest_bits;
        flit.range(27, 24) = type_bits;
        flit.range(23, 0) = payload_bits;
    }
    else if (flit_kind == FLIT_BODY) {
        sc_uint<32> body_bits = (unsigned int)payload;
        flit.range(31, 0) = body_bits;
    }

    return flit;
}

int Controller::get_flit_dest(sc_lv<34> flit)
{
    sc_lv<4> bits = flit.range(31, 28);
    return bits.to_uint();
}

int Controller::get_flit_kind(sc_lv<34> flit)
{
    sc_lv<2> bits = flit.range(33, 32);
    return bits.to_uint();
}

int Controller::get_flit_type(sc_lv<34> flit)
{
    sc_lv<4> bits = flit.range(27, 24);
    return bits.to_uint();
}

int Controller::get_flit_payload(sc_lv<34> flit)
{
    int flit_kind = get_flit_kind(flit);

    if (flit_kind == FLIT_HEAD) {
        sc_lv<24> bits = flit.range(23, 0);
        return bits.to_uint();
    }

    sc_lv<32> bits = flit.range(31, 0);
    return (int)bits.to_uint();
}

// ============================================================
// Padding 224x224x3 input image to 227x227x3
// ============================================================
vector<float> Controller::pad_input_224_to_227(const vector<float>& input)
{
    const int old_w = 224;
    const int old_h = 224;
    const int channels = 3;

    const int new_w = 227;
    const int new_h = 227;

    vector<float> padded(new_w * new_h * channels, 0.0f);

    // Padding 224 -> 227
    // top    = 2
    // bottom = 1
    // left   = 2
    // right  = 1
    int y_offset = 2;
    int x_offset = 2;

    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < old_h; y++) {
            for (int x = 0; x < old_w; x++) {
                int old_idx =
                    c * old_h * old_w +
                    y * old_w +
                    x;

                int new_idx =
                    c * new_h * new_w +
                    (y + y_offset) * new_w +
                    (x + x_offset);

                if (old_idx < (int)input.size() &&
                    new_idx < (int)padded.size()) {
                    padded[new_idx] = input[old_idx];
                }
            }
        }
    }

    return padded;
}

// ============================================================
// Enqueue config packet
// ============================================================
void Controller::enqueue_config_packet(int dest_id, const LayerConfig& config)
{
    vector<int> fields;

    fields.push_back(config.layer_id);
    fields.push_back(config.op_type);

    fields.push_back(config.in_w);
    fields.push_back(config.in_h);
    fields.push_back(config.in_c);

    fields.push_back(config.out_w);
    fields.push_back(config.out_h);
    fields.push_back(config.out_c);

    fields.push_back(config.kernel);
    fields.push_back(config.stride);
    fields.push_back(config.padding);

    fields.push_back(config.next_dest_id);

    tx_flit_queue.push(make_flit(
        dest_id,
        FLIT_HEAD,
        PKT_CONFIG,
        fields.size()
    ));

    for (int i = 0; i < (int)fields.size(); i++) {
        tx_flit_queue.push(make_flit(
            dest_id,
            FLIT_BODY,
            PKT_CONFIG,
            fields[i]
        ));
    }

    tx_flit_queue.push(make_flit(
        dest_id,
        FLIT_TAIL,
        PKT_CONFIG,
        0
    ));
}

// ============================================================
// Enqueue data packet
// ============================================================
void Controller::enqueue_data_packet(
    int dest_id,
    int packet_type,
    const vector<float>& values
)
{
    tx_flit_queue.push(make_flit(
        dest_id,
        FLIT_HEAD,
        packet_type,
        values.size()
    ));

    for (int i = 0; i < (int)values.size(); i++) {
        int raw_float_bits = float_to_fixed24(values[i]);

        tx_flit_queue.push(make_flit(
            dest_id,
            FLIT_BODY,
            packet_type,
            raw_float_bits
        ));
    }

    tx_flit_queue.push(make_flit(
        dest_id,
        FLIT_TAIL,
        packet_type,
        0
    ));
}

// ============================================================
// TX process
// ============================================================
void Controller::tx_process()
{
    if (rst.read()) {
        flit_tx.write(0);
        req_tx.write(false);

        while (!tx_flit_queue.empty()) {
            tx_flit_queue.pop();
        }

        tx_cnt = 0;
        tx_state = 0;

        return;
    }

    if (!tx_flit_queue.empty()) {
        sc_lv<34> flit = tx_flit_queue.front();
        tx_flit_queue.pop();

        flit_tx.write(flit);
        req_tx.write(true);

        tx_cnt++;

        // if (tx_cnt % 200000 == 0) {
        //     cout << "[Controller TX] Sent "
        //          << tx_cnt
        //          << " flits at "
        //          << sc_time_stamp()
        //          << endl;
        // }
    } else {
        req_tx.write(false);
    }
}

// ============================================================
// RX packet helpers
// ============================================================
void Controller::reset_rx_packet()
{
    rx_rebuilding_packet = false;
    rx_packet_type = -1;
    rx_expected_payload_count = 0;
    rx_received_payload_count = 0;
    rx_float_payload.clear();
}

void Controller::finish_rx_packet()
{
    if (rx_packet_type == PKT_RESULT) {
        current_result = rx_float_payload;

        // cout << "[Controller] Received RESULT for layer "
        //      << layer_index
        //      << ", size = "
        //      << current_result.size()
        //      << endl;

        prepare_next_layer();
    }

    reset_rx_packet();
}

// ============================================================
// RX process
// ============================================================
void Controller::rx_process()
{
    if (rst.read()) {
        ack_rx.write(false);
        reset_rx_packet();
        return;
    }

    if (req_rx.read()) {
        sc_lv<34> flit = flit_rx.read();

        int dest_id = get_flit_dest(flit);
        int flit_kind = get_flit_kind(flit);
        int packet_type = get_flit_type(flit);
        int payload = get_flit_payload(flit);

        ack_rx.write(true);

        // if (dest_id != CTRL_ID) {
        //     cout << "[Controller] WARNING: received flit for dest "
        //          << dest_id
        //          << endl;
        // }

        if (flit_kind == FLIT_HEAD) {
            reset_rx_packet();

            rx_rebuilding_packet = true;
            rx_packet_type = packet_type;
            rx_expected_payload_count = payload;
            rx_received_payload_count = 0;
        }
        else if (flit_kind == FLIT_BODY) {
            if (!rx_rebuilding_packet) {
                cout << "[Controller] ERROR: BODY received before HEAD"
                    << endl;
                return;
            }

            int raw_bits = get_flit_payload(flit);
            rx_float_payload.push_back(fixed24_to_float(raw_bits));
            rx_received_payload_count++;
        }
        else if (flit_kind == FLIT_TAIL) {
            if (!rx_rebuilding_packet) {
                cout << "[Controller] ERROR: TAIL received before HEAD"
                     << endl;
                return;
            }

            // if (rx_received_payload_count != rx_expected_payload_count) {
            //     cout << "[Controller] WARNING: result expected "
            //          << rx_expected_payload_count
            //          << " values, received "
            //          << rx_received_payload_count
            //          << endl;
            // }

            finish_rx_packet();
        }
    } else {
        ack_rx.write(false);
    }
}

// ============================================================
// Prepare next layer after receiving result
// ============================================================
void Controller::prepare_next_layer()
{
    current_input = current_result;

    layer_index++;

    current_weight.clear();
    current_bias.clear();

    weight_count = 0;
    bias_count = 0;

    weight_request_sent = false;
    bias_request_sent = false;
    layer_sent = false;
    rom_seen_valid = false;

    if (layer_index >= (int)layer_plan.size()) {
        ctrl_state = CTRL_SOFTMAX_PRINT;
        return;
    }

    LayerSpec spec = layer_plan[layer_index];

    // cout << "[Controller] Moving to layer index "
    //      << layer_index
    //      << ", op_type = "
    //      << spec.op_type
    //      << endl;

    if (layer_needs_weight(spec)) {
        ctrl_state = CTRL_REQUEST_WEIGHT;
    } else {
        ctrl_state = CTRL_SEND_LAYER;
    }
}

// ============================================================
// ROM request process
// ============================================================
void Controller::ROM_request()
{
    if (rst.read()) {
        layer_id.write(0);
        layer_id_type.write(false);
        layer_id_valid.write(false);

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

        image_data.clear();
        padded_image.clear();

        current_input.clear();
        current_weight.clear();
        current_bias.clear();
        current_result.clear();

        reset_rx_packet();

        return;
    }

    if (ctrl_state == CTRL_RESET_STATE) {
        ctrl_state = CTRL_REQUEST_IMAGE;
    }

    // Default low.
    // ROM reports an error if layer_id_valid stays high while it is reading. :contentReference[oaicite:1]{index=1}
    layer_id_valid.write(false);

    // ------------------------------------------------------------
    // Request image
    // ------------------------------------------------------------
    if (ctrl_state == CTRL_REQUEST_IMAGE && !image_request_sent) {
        // cout << "[Controller] Requesting input image from ROM" << endl;

        layer_id.write(0);
        layer_id_type.write(false);
        layer_id_valid.write(true);

        image_request_sent = true;
        rom_seen_valid = false;

        ctrl_state = CTRL_READ_IMAGE;
        return;
    }

    // ------------------------------------------------------------
    // Request current layer weight
    // ------------------------------------------------------------
    if (ctrl_state == CTRL_REQUEST_WEIGHT && !weight_request_sent) {
        LayerSpec spec = layer_plan[layer_index];

        // cout << "[Controller] Requesting weight for ROM layer "
        //      << spec.rom_id
        //      << endl;

        layer_id.write(spec.rom_id);
        layer_id_type.write(false);
        layer_id_valid.write(true);

        weight_request_sent = true;
        rom_seen_valid = false;

        ctrl_state = CTRL_READ_WEIGHT;
        return;
    }

    // ------------------------------------------------------------
    // Request current layer bias
    // ------------------------------------------------------------
    if (ctrl_state == CTRL_REQUEST_BIAS && !bias_request_sent) {
        LayerSpec spec = layer_plan[layer_index];

        // cout << "[Controller] Requesting bias for ROM layer "
        //      << spec.rom_id
        //      << endl;

        layer_id.write(spec.rom_id);
        layer_id_type.write(true);
        layer_id_valid.write(true);

        bias_request_sent = true;
        rom_seen_valid = false;

        ctrl_state = CTRL_READ_BIAS;
        return;
    }

    // ------------------------------------------------------------
    // Send current layer to PE
    // ------------------------------------------------------------
    if (ctrl_state == CTRL_SEND_LAYER && !layer_sent) {
        LayerSpec spec = layer_plan[layer_index];
        LayerConfig config = make_config_from_spec(spec);

        // cout << "[Controller] Sending layer index "
        //      << layer_index
        //      << " to PE "
        //      << spec.dest_pe
        //      << ", op_type = "
        //      << spec.op_type
        //      << endl;

        enqueue_config_packet(spec.dest_pe, config);
        enqueue_data_packet(spec.dest_pe, PKT_DATA, current_input);

        if (layer_needs_weight(spec)) {
            enqueue_data_packet(spec.dest_pe, PKT_WEIGHT, current_weight);
            enqueue_data_packet(spec.dest_pe, PKT_BIAS, current_bias);
        }

        // cout << "[Controller] Layer index "
        //      << layer_index
        //      << " queued. Input size = "
        //      << current_input.size();

        // if (layer_needs_weight(spec)) {
        //     cout << ", weight size = "
        //          << current_weight.size()
        //          << ", bias size = "
        //          << current_bias.size();
        // }

        // cout << endl;

        layer_sent = true;
        ctrl_state = CTRL_WAIT_RESULT;
        return;
    }

    // ------------------------------------------------------------
    // Print final output
    // ------------------------------------------------------------
    if (ctrl_state == CTRL_SOFTMAX_PRINT) {
        print_softmax_top100(current_input);
        ctrl_state = CTRL_DONE;
        sc_stop();
        return;
    }
}

// ============================================================
// ROM receive process
//
// Uses data_valid falling edge to detect EOF.
// ============================================================
void Controller::ROM_receive()
{
    if (rst.read()) {
        return;
    }

    bool valid = data_valid.read();

    // ------------------------------------------------------------
    // Read image
    // ------------------------------------------------------------
    if (ctrl_state == CTRL_READ_IMAGE) {
        if (valid) {
            rom_seen_valid = true;
            image_data.push_back(data.read());
            image_count++;
            return;
        }

        if (!valid && rom_seen_valid) {
            // cout << "[Controller] Finished receiving image, size = "
            //      << image_data.size()
            //      << endl;

            padded_image = pad_input_224_to_227(image_data);
            current_input = padded_image;

            // cout << "[Controller] Padded image size = "
            //      << current_input.size()
            //      << endl;

            rom_seen_valid = false;

            LayerSpec spec = layer_plan[layer_index];

            if (layer_needs_weight(spec)) {
                ctrl_state = CTRL_REQUEST_WEIGHT;
            } else {
                ctrl_state = CTRL_SEND_LAYER;
            }
        }

        return;
    }

    // ------------------------------------------------------------
    // Read weight
    // ------------------------------------------------------------
    if (ctrl_state == CTRL_READ_WEIGHT) {
        if (valid) {
            rom_seen_valid = true;
            current_weight.push_back(data.read());
            weight_count++;
            return;
        }

        if (!valid && rom_seen_valid) {
            // cout << "[Controller] Finished receiving weight for layer index "
            //      << layer_index
            //      << ", size = "
            //      << current_weight.size()
            //      << endl;

            rom_seen_valid = false;
            ctrl_state = CTRL_REQUEST_BIAS;
        }

        return;
    }

    // ------------------------------------------------------------
    // Read bias
    // ------------------------------------------------------------
    if (ctrl_state == CTRL_READ_BIAS) {
        if (valid) {
            rom_seen_valid = true;
            current_bias.push_back(data.read());
            bias_count++;
            return;
        }

        if (!valid && rom_seen_valid) {
            // cout << "[Controller] Finished receiving bias for layer index "
            //      << layer_index
            //      << ", size = "
            //      << current_bias.size()
            //      << endl;

            rom_seen_valid = false;
            ctrl_state = CTRL_SEND_LAYER;
        }

        return;
    }
}

// ============================================================
// Softmax + top 100 print
// ============================================================
void Controller::print_softmax_top100(const vector<float>& logits)
{
    // cout << "[Controller] FC8 result received. Running softmax." << endl;

    if (logits.empty()) {
        cout << "[Controller] ERROR: logits is empty." << endl;
        return;
    }

    // ============================================================
    // Load class names
    // ============================================================
    vector<string> class_names;
    ifstream class_file("./data/imagenet_classes.txt");

    if (!class_file.is_open()) {
        class_file.open("./data/labels.txt");
    }

    if (!class_file.is_open()) {
        class_file.open("./data/class_name.txt");
    }

    string line;
    while (getline(class_file, line)) {
        class_names.push_back(line);
    }

    // ============================================================
    // Softmax
    // ============================================================
    float max_logit = logits[0];

    for (int i = 1; i < (int)logits.size(); i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    vector<float> probs(logits.size(), 0.0f);
    float sum = 0.0f;

    for (int i = 0; i < (int)logits.size(); i++) {
        probs[i] = exp(logits[i] - max_logit);
        sum += probs[i];
    }

    for (int i = 0; i < (int)probs.size(); i++) {
        probs[i] = probs[i] / sum;
    }

    // pair structure:
    // first  = probability
    // second = class index
    vector<pair<float, int> > ranking;

    for (int i = 0; i < (int)probs.size(); i++) {
        ranking.push_back(make_pair(probs[i], i));
    }

    sort(ranking.begin(), ranking.end(), greater<pair<float, int> >());

    // ============================================================
    // Print output
    // ============================================================
    cout << fixed << setprecision(2);
    cout << "Top 100 classes:" << endl;
    cout << "=================================================" << endl;
    cout << right
         << setw(6) << "idx"
         << " | "
         << setw(9) << "val"
         << " | "
         << setw(11) << "possibility"
         << " | "
         << "class name"
         << endl;
    cout << "-------------------------------------------------" << endl;

    int top_n = 100;

    if ((int)ranking.size() < top_n) {
        top_n = ranking.size();
    }

    for (int i = 0; i < top_n; i++) {
        int idx = ranking[i].second;
        float val = logits[idx];
        float possibility = probs[idx] * 100.0f;

        string class_name = "unknown";

        if (idx >= 0 && idx < (int)class_names.size()) {
            class_name = class_names[idx];
        }

        cout << right
             << setw(6) << idx
             << " | "
             << setw(9) << val
             << " | "
             << setw(11) << possibility
             << " | "
             << class_name
             << endl;
    }

    cout << "=================================================" << endl;
}