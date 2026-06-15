#include "core.h"

int Core::float_to_fixed24(float value)
{
    union {
        float f;
        unsigned int u;
    } converter;

    converter.f = value;
    return (int)converter.u;
}

float Core::fixed24_to_float(int payload)
{
    union {
        float f;
        unsigned int u;
    } converter;

    converter.u = (unsigned int)payload;
    return converter.f;
}

sc_lv<34> Core::make_flit(int dest_id, int flit_kind, int packet_type, int payload)
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

int Core::get_flit_kind(sc_lv<34> flit)
{
    sc_lv<2> bits = flit.range(33, 32);
    return bits.to_uint();
}

int Core::get_flit_type(sc_lv<34> flit)
{
    sc_lv<4> bits = flit.range(27, 24);
    return bits.to_uint();
}

int Core::get_flit_payload(sc_lv<34> flit)
{
    int flit_kind = get_flit_kind(flit);

    if (flit_kind == FLIT_HEAD) {
        sc_lv<24> bits = flit.range(23, 0);
        return bits.to_uint();
    }

    sc_lv<32> bits = flit.range(31, 0);
    return (int)bits.to_uint();
}

int Core::input_tensor_size()
{
    return profile.in_w * profile.in_h * profile.in_c;
}

void Core::reset_packet_rebuild()
{
    rebuilding_packet = false;
    current_packet_type = -1;
    expected_payload_count = 0;
    received_payload_count = 0;
    current_float_payload.clear();
}

void Core::reset_local_input_buffer()
{
    if (profile.expected_input_chunks > 1) {
        input_data.assign(input_tensor_size(), 0.0f);
    }
    else {
        input_data.clear();
    }
    received_input_chunks = 0;
    has_data = false;
}

void Core::merge_forwarded_input_chunk(int output_start, int output_count, const vector<float>& chunk)
{
    // FC6 receives Conv5 pooled feature-map chunks. The previous-stage chunk
    // metadata is channel-based, so each channel has 6x6 values.
    if (profile.layer_id == 6) {
        int spatial = 6 * 6;
        if ((int)input_data.size() != input_tensor_size()) {
            input_data.assign(input_tensor_size(), 0.0f);
        }

        for (int local_c = 0; local_c < output_count; local_c++) {
            int global_c = output_start + local_c;
            for (int idx = 0; idx < spatial; idx++) {
                int src_idx = local_c * spatial + idx;
                int dst_idx = global_c * spatial + idx;
                if (src_idx >= 0 && src_idx < (int)chunk.size() &&
                    dst_idx >= 0 && dst_idx < (int)input_data.size()) {
                    input_data[dst_idx] = chunk[src_idx];
                }
            }
        }
    }
    // FC7 and FC8 receive flat vector chunks.
    else if (profile.is_fc) {
        if ((int)input_data.size() != input_tensor_size()) {
            input_data.assign(input_tensor_size(), 0.0f);
        }

        for (int i = 0; i < output_count && i < (int)chunk.size(); i++) {
            int dst_idx = output_start + i;
            if (dst_idx >= 0 && dst_idx < (int)input_data.size()) {
                input_data[dst_idx] = chunk[i];
            }
        }
    }
    else {
        // Convolutional stages receive channel-major feature-map chunks.
        int spatial = profile.in_w * profile.in_h;
        if ((int)input_data.size() != input_tensor_size()) {
            input_data.assign(input_tensor_size(), 0.0f);
        }

        for (int local_c = 0; local_c < output_count; local_c++) {
            int global_c = output_start + local_c;
            for (int idx = 0; idx < spatial; idx++) {
                int src_idx = local_c * spatial + idx;
                int dst_idx = global_c * spatial + idx;
                if (src_idx >= 0 && src_idx < (int)chunk.size() &&
                    dst_idx >= 0 && dst_idx < (int)input_data.size()) {
                    input_data[dst_idx] = chunk[src_idx];
                }
            }
        }
    }

    received_input_chunks++;

    cout << "[Core " << id << "] Received forwarded activation chunk "
         << received_input_chunks << "/" << profile.expected_input_chunks
         << " start=" << output_start
         << " count=" << output_count << endl;

    if (received_input_chunks >= profile.expected_input_chunks) {
        has_data = true;
        compute_started = false;
        compute_done = false;
        cout << "[Core " << id << "] Full input activation assembled locally." << endl;
    }
}

void Core::finish_packet()
{
    if (current_packet_type == PKT_DATA) {
        // PKT_DATA is only the original input image sent by controller/global SRAM
        // to first-stage Conv1 PEs.
        input_data = current_float_payload;
        received_input_chunks = 1;
        has_data = true;
        compute_started = false;
        compute_done = false;

        cout << "[Core " << id << "] Received input image from controller, size="
             << input_data.size() << endl;
    }
    else if (current_packet_type == PKT_WEIGHT) {
        pe.preload_weight(current_float_payload);
        has_weight = true;
        compute_started = false;
        compute_done = false;
        cout << "[Core " << id << "] Weight chunk loaded into PE-local SRAM, size="
             << current_float_payload.size() << endl;
    }
    else if (current_packet_type == PKT_BIAS) {
        pe.preload_bias(current_float_payload);
        has_bias = true;
        compute_started = false;
        compute_done = false;
        cout << "[Core " << id << "] Bias chunk loaded into PE-local SRAM, size="
             << current_float_payload.size() << endl;
    }
    else if (current_packet_type == PKT_RESULT) {
        // PE-to-PE forwarded activation packet payload:
        // [0] previous-stage output_start
        // [1] previous-stage output_count
        // [2...] activation chunk
        if (current_float_payload.size() >= 2) {
            int output_start = (int)current_float_payload[0];
            int output_count = (int)current_float_payload[1];

            vector<float> chunk;
            chunk.insert(chunk.end(), current_float_payload.begin() + 2, current_float_payload.end());

            merge_forwarded_input_chunk(output_start, output_count, chunk);
        }
    }
    else {
        cout << "[Core " << id << "] Unknown packet type " << current_packet_type << endl;
    }

    reset_packet_rebuild();
}

bool Core::ready_to_compute()
{
    if (!has_data) {
        return false;
    }

    if (PE::profile_needs_weight(profile)) {
        return has_weight && has_bias && pe.has_weight() && pe.has_bias();
    }

    return true;
}

void Core::enqueue_data_packet(int dest_id, int packet_type, const vector<float>& values)
{
    tx_flit_queue.push(make_flit(dest_id, FLIT_HEAD, packet_type, values.size()));

    for (int i = 0; i < (int)values.size(); i++) {
        int raw_float_bits = float_to_fixed24(values[i]);
        tx_flit_queue.push(make_flit(dest_id, FLIT_BODY, packet_type, raw_float_bits));
    }

    tx_flit_queue.push(make_flit(dest_id, FLIT_TAIL, packet_type, 0));
}

void Core::enqueue_result_to_next_stage(const vector<float>& result)
{
    vector<float> result_packet;
    result_packet.reserve(result.size() + 2);
    result_packet.push_back((float)profile.output_start);
    result_packet.push_back((float)profile.output_count);
    result_packet.insert(result_packet.end(), result.begin(), result.end());

    if (profile.next_pe_count > 0) {
        for (int i = 0; i < profile.next_pe_count && i < MAX_NEXT_PE; i++) {
            int dest = profile.next_pe_ids[i];
            if (dest >= 0) {
                enqueue_data_packet(dest, PKT_RESULT, result_packet);
                cout << "[Core " << id << "] Forwarded layer " << profile.layer_id
                     << " chunk start=" << profile.output_start
                     << " count=" << profile.output_count
                     << " to PE" << dest << endl;
            }
        }
    }
    else {
        // Final FC8 result goes to external controller/output buffer.
        enqueue_data_packet(CTRL_ID, PKT_RESULT, result_packet);
        cout << "[Core " << id << "] Sent final output chunk to controller." << endl;
    }
}

void Core::rx_process()
{
    if (rst.read()) {
        ack_rx.write(false);
        rx_cnt = 0;

        reset_packet_rebuild();
        input_data.clear();
        weight_data.clear();
        bias_data.clear();
        pe.clear_preload();

        has_data = false;
        has_weight = false;
        has_bias = false;

        compute_started = false;
        compute_done = false;
        received_input_chunks = 0;

        reset_local_input_buffer();
        return;
    }

    if (req_rx.read()) {
        sc_lv<34> flit = flit_rx.read();
        int flit_kind = get_flit_kind(flit);
        int payload = get_flit_payload(flit);

        ack_rx.write(true);
        rx_cnt++;

        if (flit_kind == FLIT_HEAD) {
            reset_packet_rebuild();
            current_packet_type = get_flit_type(flit);
            expected_payload_count = (unsigned int)payload;
            received_payload_count = 0;
            rebuilding_packet = true;
        }
        else if (flit_kind == FLIT_BODY) {
            if (!rebuilding_packet) {
                cout << "[Core " << id << "] ERROR: BODY before HEAD" << endl;
                return;
            }

            current_float_payload.push_back(fixed24_to_float(payload));
            received_payload_count++;
        }
        else if (flit_kind == FLIT_TAIL) {
            if (!rebuilding_packet) {
                cout << "[Core " << id << "] ERROR: TAIL before HEAD" << endl;
                return;
            }
            finish_packet();
        }
    } else {
        ack_rx.write(false);
    }
}

void Core::tx_process()
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

void Core::CAL_process()
{
    if (rst.read()) {
        return;
    }

    if (compute_done || compute_started) {
        return;
    }

    if (!ready_to_compute()) {
        return;
    }

    compute_started = true;

    vector<float> result = pe.compute(input_data, weight_data, bias_data, profile, id);
    enqueue_result_to_next_stage(result);

    compute_done = true;

    // Prepare for a later image. Weight/bias stay in PE-local SRAM.
    reset_local_input_buffer();
}
