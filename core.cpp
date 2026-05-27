#include "core.h"

// ============================================================
// Exact float-bit conversion
// keep the function names fixed24_to_float / float_to_fixed24
// But internally, this now sends exact 32-bit float bits.
// ============================================================
float Core::fixed24_to_float(int payload)
{
    union {
        float f;
        unsigned int u;
    } converter;

    converter.u = (unsigned int)payload;
    return converter.f;
}

int Core::float_to_fixed24(float value)
{
    union {
        float f;
        unsigned int u;
    } converter;

    converter.f = value;
    return (int)converter.u;
}

// ============================================================
// New flit format:
//
// HEAD:
// [33:32] flit kind
// [31:28] destination ID
// [27:24] packet type
// [23:0]  payload count
//
// BODY:
// [33:32] flit kind
// [31:0]  exact float bits or config integer
//
// TAIL:
// [33:32] flit kind
// ============================================================
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

int Core::get_flit_dest(sc_lv<34> flit)
{
    sc_lv<4> bits = flit.range(31, 28);
    return bits.to_uint();
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

// ============================================================
// Reset current packet rebuild state
// ============================================================
void Core::reset_packet_rebuild()
{
    rebuilding_packet = false;

    current_packet_type = -1;
    expected_payload_count = 0;
    received_payload_count = 0;

    current_config_fields.clear();
    current_float_payload.clear();
}

// ============================================================
// Convert config packet fields into LayerConfig
// ============================================================
void Core::apply_config_fields(const vector<int>& fields)
{
    if ((int)fields.size() < 12) {
        cout << "[Core " << id << "] ERROR: config packet has only "
             << fields.size() << " fields" << endl;
        return;
    }

    current_config.layer_id = fields[0];
    current_config.op_type = fields[1];

    current_config.in_w = fields[2];
    current_config.in_h = fields[3];
    current_config.in_c = fields[4];

    current_config.out_w = fields[5];
    current_config.out_h = fields[6];
    current_config.out_c = fields[7];

    current_config.kernel = fields[8];
    current_config.stride = fields[9];
    current_config.padding = fields[10];

    current_config.next_dest_id = fields[11];

    has_config = true;

    input_data.clear();
    weight_data.clear();
    bias_data.clear();

    has_data = false;
    has_weight = false;
    has_bias = false;

    compute_started = false;
    compute_done = false;
    waiting_printed = false;

    // cout << "[Core " << id << "] Stored CONFIG"
    //      << " layer_id=" << current_config.layer_id
    //      << " op_type=" << current_config.op_type
    //      << " input=" << current_config.in_w << "x"
    //      << current_config.in_h << "x"
    //      << current_config.in_c
    //      << " output=" << current_config.out_w << "x"
    //      << current_config.out_h << "x"
    //      << current_config.out_c
    //      << " next_dest=" << current_config.next_dest_id
    //      << endl;
}

// ============================================================
// Finish one complete packet
// ============================================================
void Core::finish_packet()
{
    if (current_packet_type == PKT_CONFIG) {
        apply_config_fields(current_config_fields);
    }
    else if (current_packet_type == PKT_DATA) {
        input_data = current_float_payload;
        has_data = true;

        compute_started = false;
        compute_done = false;
        waiting_printed = false;

        // cout << "[Core " << id << "] Stored DATA packet, size = "
        //      << input_data.size() << endl;
    }
    else if (current_packet_type == PKT_WEIGHT) {
        weight_data = current_float_payload;
        has_weight = true;

        compute_started = false;
        compute_done = false;
        waiting_printed = false;

        // cout << "[Core " << id << "] Stored WEIGHT packet, size = "
        //      << weight_data.size() << endl;
    }
    else if (current_packet_type == PKT_BIAS) {
        bias_data = current_float_payload;
        has_bias = true;

        compute_started = false;
        compute_done = false;
        waiting_printed = false;

        // cout << "[Core " << id << "] Stored BIAS packet, size = "
        //      << bias_data.size() << endl;
    }
    else if (current_packet_type == PKT_RESULT) {
        input_data = current_float_payload;
        has_data = true;

        compute_started = false;
        compute_done = false;
        waiting_printed = false;

        // cout << "[Core " << id << "] Stored RESULT as next input, size = "
        //      << input_data.size() << endl;
    }
    else {
        cout << "[Core " << id << "] Unknown packet type "
             << current_packet_type << endl;
    }

    reset_packet_rebuild();
}

// ============================================================
// Check if PE has all needed packets
// ============================================================
bool Core::ready_to_compute()
{
    if (!has_config) {
        return false;
    }

    if (!has_data) {
        return false;
    }

    if (current_config.op_type == OP_MAXPOOL) {
        return true;
    }

    if (current_config.op_type == OP_CONV_RELU ||
        current_config.op_type == OP_FC_RELU ||
        current_config.op_type == OP_FC_ONLY) {
        return has_weight && has_bias;
    }

    return false;
}

// ============================================================
// Enqueue result packet
// ============================================================
void Core::enqueue_data_packet(int dest_id, int packet_type, const vector<float>& values)
{
    tx_flit_queue.push(make_flit(dest_id, FLIT_HEAD, packet_type, values.size()));

    for (int i = 0; i < (int)values.size(); i++) {
        int raw_float_bits = float_to_fixed24(values[i]);
        tx_flit_queue.push(make_flit(dest_id, FLIT_BODY, packet_type, raw_float_bits));
    }

    tx_flit_queue.push(make_flit(dest_id, FLIT_TAIL, packet_type, 0));

    // cout << "[Core " << id << "] Queued RESULT packet to "
    //      << dest_id
    //      << ", size = "
    //      << values.size()
    //      << endl;
}

// ============================================================
// RX process: receive flits from router
// ============================================================
void Core::rx_process()
{
    if (rst.read()) {
        ack_rx.write(false);
        rx_cnt = 0;

        reset_packet_rebuild();

        input_data.clear();
        weight_data.clear();
        bias_data.clear();

        has_config = false;
        has_data = false;
        has_weight = false;
        has_bias = false;

        compute_started = false;
        compute_done = false;
        waiting_printed = false;

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

            int dest_id = get_flit_dest(flit);
            int packet_type = get_flit_type(flit);

            rebuilding_packet = true;
            current_packet_type = packet_type;
            expected_payload_count = payload;
            received_payload_count = 0;

            // if (dest_id != id) {
            //     cout << "[Core " << id << "] WARNING: received HEAD for dest "
            //          << dest_id << endl;
            // }
        }
        else if (flit_kind == FLIT_BODY) {
            if (!rebuilding_packet) {
                cout << "[Core " << id << "] ERROR: BODY received before HEAD"
                     << endl;
                return;
            }

            int raw_bits = payload;

            if (current_packet_type == PKT_CONFIG) {
                current_config_fields.push_back(raw_bits);
            } else {
                current_float_payload.push_back(fixed24_to_float(raw_bits));
            }

            received_payload_count++;
        }
        else if (flit_kind == FLIT_TAIL) {
            if (!rebuilding_packet) {
                cout << "[Core " << id << "] ERROR: TAIL received before HEAD"
                     << endl;
                return;
            }

            // Do not check expected_payload_count here.
            // FC6 weight has more than 24-bit count, so HEAD count can wrap.
            // TAIL is the real end-of-packet marker.

            finish_packet();
        }
    } else {
        ack_rx.write(false);
    }
}

// ============================================================
// TX process: send flits to router
// ============================================================
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

// ============================================================
// CAL process: call local PE when packets are ready
// ============================================================
void Core::CAL_process()
{
    if (rst.read()) {
        return;
    }

    if (compute_done || compute_started) {
        return;
    }

    if (!has_config || !has_data) {
        return;
    }

    if (!ready_to_compute()) {
        if (!waiting_printed) {
            // cout << "[Core " << id << "] Waiting for required packets";

            // if (!has_weight &&
            //     (current_config.op_type == OP_CONV_RELU ||
            //      current_config.op_type == OP_FC_RELU ||
            //      current_config.op_type == OP_FC_ONLY)) {
            //     cout << " + WEIGHT";
            // }

            // if (!has_bias &&
            //     (current_config.op_type == OP_CONV_RELU ||
            //      current_config.op_type == OP_FC_RELU ||
            //      current_config.op_type == OP_FC_ONLY)) {
            //     cout << " + BIAS";
            // }

            // cout << endl;

            waiting_printed = true;
        }

        return;
    }

    compute_started = true;

    // cout << "[Core " << id << "] Sending layer "
    //      << current_config.layer_id
    //      << " to PE, op_type "
    //      << current_config.op_type
    //      << endl;

    vector<float> result = pe.compute(
        input_data,
        weight_data,
        bias_data,
        current_config,
        id
    );

    enqueue_data_packet(current_config.next_dest_id, PKT_RESULT, result);

    compute_done = true;

    // cout << "[Core " << id << "] PE compute finished for layer "
    //      << current_config.layer_id
    //      << endl;
}