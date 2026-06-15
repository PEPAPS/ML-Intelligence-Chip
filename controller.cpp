#include "controller.h"

void Controller::init_controller_plan_from_pe_profiles()
{
    preload_pe_ids = PE::get_all_compute_pe_ids();
    input_pe_ids = PE::get_first_stage_pe_ids();
    expected_final_chunks = PE::get_final_output_chunk_count();

    cout << "[CTRL] PE roles are owned by PE::get_profile() in pe.cpp" << endl;
    cout << "[CTRL] No role/setup packet is sent by the controller." << endl;
    for (int i = 0; i < (int)preload_pe_ids.size(); i++) {
        cout << "[CTRL] " << PE::role_for_pe_static(preload_pe_ids[i]) << endl;
    }
}

unsigned int Controller::get_weight_base(int layer_id)
{
    if (layer_id == 1) return DRAM_W1_BASE;
    if (layer_id == 2) return DRAM_W2_BASE;
    if (layer_id == 3) return DRAM_W3_BASE;
    if (layer_id == 4) return DRAM_W4_BASE;
    if (layer_id == 5) return DRAM_W5_BASE;
    if (layer_id == 6) return DRAM_W6_BASE;
    if (layer_id == 7) return DRAM_W7_BASE;
    if (layer_id == 8) return DRAM_W8_BASE;
    return 0;
}

unsigned int Controller::get_bias_base(int layer_id)
{
    return DRAM_BIAS_BASE + layer_id * DRAM_BIAS_STRIDE;
}

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

sc_lv<34> Controller::make_flit(int dest_id, int flit_kind, int packet_type, int payload)
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

void Controller::enqueue_data_packet(int dest_id, int packet_type, const vector<float>& values)
{
    tx_flit_queue.push(make_flit(dest_id, FLIT_HEAD, packet_type, values.size()));

    for (int i = 0; i < (int)values.size(); i++) {
        int raw_float_bits = float_to_fixed24(values[i]);
        tx_flit_queue.push(make_flit(dest_id, FLIT_BODY, packet_type, raw_float_bits));
    }

    tx_flit_queue.push(make_flit(dest_id, FLIT_TAIL, packet_type, 0));
}

void Controller::enqueue_preload_for_current_pe()
{
    int pe_id = preload_pe_ids[preload_pe_index];
    PEProfile profile = PE::get_profile(pe_id);

    if (PE::profile_needs_weight(profile)) {
        unsigned int weight_addr =
            get_weight_base(profile.layer_id) + PE::profile_weight_offset_words(profile);
        unsigned int weight_count = PE::profile_weight_count_words(profile);

        unsigned int bias_addr = get_bias_base(profile.layer_id) + profile.output_start;
        unsigned int bias_count = profile.output_count;

        vector<float> weight = dma->burst_read(weight_addr, weight_count);
        vector<float> bias = dma->burst_read(bias_addr, bias_count);

        // No role/setup packet is sent. These are the only preload packets.
        enqueue_data_packet(pe_id, PKT_WEIGHT, weight);
        enqueue_data_packet(pe_id, PKT_BIAS, bias);
    }

    cout << "[CTRL] Sent weight/bias only to " << PE::role_for_pe_static(pe_id)
         << " | layer=" << profile.layer_id
         << " output_start=" << profile.output_start
         << " output_count=" << profile.output_count << endl;
}

void Controller::send_input_to_first_stage_group()
{
    global_sram_read_words += global_input_buffer.size() * input_pe_ids.size();

    for (int i = 0; i < (int)input_pe_ids.size(); i++) {
        int pe_id = input_pe_ids[i];
        enqueue_data_packet(pe_id, PKT_DATA, global_input_buffer);
        cout << "[CTRL] Sent original 224x224x3 input image to "
             << PE::role_for_pe_static(pe_id) << endl;
    }
}

void Controller::merge_final_output_chunk(int output_start, int output_count, const vector<float>& chunk)
{
    if (global_output_buffer.empty()) {
        global_output_buffer.assign(1000, 0.0f);
        global_sram_write_words += 1000;
    }

    for (int i = 0; i < output_count && i < (int)chunk.size(); i++) {
        int out_idx = output_start + i;
        if (out_idx >= 0 && out_idx < (int)global_output_buffer.size()) {
            global_output_buffer[out_idx] = chunk[i];
        }
    }
}

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
        if (rx_float_payload.size() >= 2) {
            int output_start = (int)rx_float_payload[0];
            int output_count = (int)rx_float_payload[1];

            vector<float> chunk;
            chunk.insert(chunk.end(), rx_float_payload.begin() + 2, rx_float_payload.end());

            merge_final_output_chunk(output_start, output_count, chunk);
            received_final_chunks++;

            cout << "[CTRL] Received final FC8 chunk " << received_final_chunks
                 << "/" << expected_final_chunks
                 << " start=" << output_start
                 << " count=" << output_count << endl;

            if (received_final_chunks >= expected_final_chunks) {
                ctrl_state = CTRL_SOFTMAX_PRINT;
            }
        }
    }

    reset_rx_packet();
}

void Controller::tx_process()
{
    if (rst.read()) {
        flit_tx.write(0);
        req_tx.write(false);
        tx_cnt = 0;
        while (!tx_flit_queue.empty()) {
            tx_flit_queue.pop();
        }
        return;
    }

    if (!tx_flit_queue.empty()) {
        sc_lv<34> flit = tx_flit_queue.front();
        tx_flit_queue.pop();
        flit_tx.write(flit);
        req_tx.write(true);
        tx_cnt++;
    } else {
        req_tx.write(false);
    }
}

void Controller::rx_process()
{
    if (rst.read()) {
        ack_rx.write(false);
        reset_rx_packet();
        return;
    }

    if (req_rx.read()) {
        sc_lv<34> flit = flit_rx.read();
        int kind = get_flit_kind(flit);
        int payload = get_flit_payload(flit);

        ack_rx.write(true);

        if (kind == FLIT_HEAD) {
            reset_rx_packet();
            rx_packet_type = get_flit_type(flit);
            rx_expected_payload_count = payload;
            rx_received_payload_count = 0;
            rx_rebuilding_packet = true;
        }
        else if (kind == FLIT_BODY) {
            if (rx_rebuilding_packet) {
                rx_float_payload.push_back(fixed24_to_float(payload));
                rx_received_payload_count++;
            }
        }
        else if (kind == FLIT_TAIL) {
            if (rx_rebuilding_packet) {
                finish_rx_packet();
            }
        }
    } else {
        ack_rx.write(false);
    }
}

void Controller::DMA_request()
{
    if (rst.read()) {
        ctrl_state = CTRL_RESET_STATE;
        preload_pe_index = 0;
        image_request_sent = false;
        input_sent = false;
        expected_final_chunks = PE::get_final_output_chunk_count();
        received_final_chunks = 0;
        global_input_buffer.clear();
        global_output_buffer.clear();
        global_sram_read_words = 0;
        global_sram_write_words = 0;
        cycle_count = 0;
        return;
    }

    // Count one accelerator cycle for each positive clock edge after reset.
    cycle_count++;

    if (ctrl_state == CTRL_RESET_STATE) {
        init_controller_plan_from_pe_profiles();
        ctrl_state = CTRL_PRELOAD_PE;
        return;
    }

    if (ctrl_state == CTRL_PRELOAD_PE) {
        if (!tx_flit_queue.empty()) {
            return;
        }

        if (preload_pe_index < (int)preload_pe_ids.size()) {
            enqueue_preload_for_current_pe();
            preload_pe_index++;
            return;
        }

        cout << "[CTRL] Weight/bias preload complete. No role/setup packets were sent." << endl;
        ctrl_state = CTRL_REQUEST_IMAGE;
        return;
    }

    if (ctrl_state == CTRL_REQUEST_IMAGE) {
        if (!image_request_sent) {
            global_input_buffer = dma->burst_read(DRAM_INPUT_BASE, 224 * 224 * 3);
            global_sram_write_words += global_input_buffer.size();
            image_request_sent = true;
            input_sent = false;
            received_final_chunks = 0;
            global_output_buffer.clear();
            ctrl_state = CTRL_SEND_INPUT;
            return;
        }
    }

    if (ctrl_state == CTRL_SEND_INPUT) {
        if (!tx_flit_queue.empty()) {
            return;
        }

        if (!input_sent) {
            send_input_to_first_stage_group();
            input_sent = true;
            ctrl_state = CTRL_WAIT_FINAL;
            return;
        }
    }

    if (ctrl_state == CTRL_WAIT_FINAL) {
        return;
    }

    if (ctrl_state == CTRL_SOFTMAX_PRINT) {
        if (!tx_flit_queue.empty()) {
            return;
        }

        dma->burst_write(DRAM_OUTPUT_BASE, global_output_buffer);
        vector<float> final_output = dma->burst_read(DRAM_OUTPUT_BASE, global_output_buffer.size());

        cout << "[CTRL] Final output written back to DRAM at DRAM_OUTPUT_BASE." << endl;
        cout << "[CTRL] Global SRAM read words  = " << global_sram_read_words << endl;
        cout << "[CTRL] Global SRAM write words = " << global_sram_write_words << endl;
        cout << "[CTRL] Intermediate activations used PE-to-PE forwarding, not global SRAM merge." << endl;
        cout << "[CTRL] PE mapping is hardcoded in pe.cpp. Controller sent only weight, bias, input, and received output." << endl;

        print_softmax_top100(final_output);

        if (dma != NULL) {
            dma->print_stats();
        }

        cout << "[METRIC] Optimized execution cycles = " << cycle_count << endl;
        cout << "[METRIC] Optimized simulated time = " << sc_time_stamp() << endl;

        ctrl_state = CTRL_DONE;
        sc_stop();
        return;
    }
}

void Controller::print_softmax_top100(const vector<float>& logits)
{
    ifstream label_file("./data/imagenet_classes.txt");
    vector<string> labels;
    string line;

    while (getline(label_file, line)) {
        labels.push_back(line);
    }

    vector<double> prob(logits.size(), 0.0);

    double max_logit = -1.0e300;
    for (int i = 0; i < (int)logits.size(); i++) {
        max_logit = max(max_logit, (double)logits[i]);
    }

    double exp_sum = 0.0;
    for (int i = 0; i < (int)logits.size(); i++) {
        prob[i] = exp((double)logits[i] - max_logit);
        exp_sum += prob[i];
    }

    vector<pair<double, int> > indexed_values;
    indexed_values.reserve(prob.size());

    for (int i = 0; i < (int)prob.size(); i++) {
        double possibility = (exp_sum > 0.0) ? (prob[i] / exp_sum) : 0.0;
        indexed_values.push_back(make_pair(possibility, i));
    }

    sort(indexed_values.begin(), indexed_values.end(),
         [](const pair<double, int>& a, const pair<double, int>& b) {
             return a.first > b.first;
         });

    cout << "Top 100 classes:" << endl;
    cout << "=================================================" << endl;
    cout << "   idx |      val | possibility | class name" << endl;
    cout << "-------------------------------------------------" << endl;
    cout << fixed << setprecision(2);

    int topn = min(100, (int)indexed_values.size());
    for (int i = 0; i < topn; ++i) {
        int class_id = indexed_values[i].second;
        string label = class_id < (int)labels.size() ? labels[class_id] : "unknown";

        cout << setw(6) << class_id
             << " | " << setw(8) << logits[class_id]
             << " | " << setw(11) << indexed_values[i].first * 100.0
             << " | " << label << endl;
    }

    cout << "=================================================" << endl;
}

