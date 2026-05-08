#ifndef CORE_H
#define CORE_H

#include "systemc.h"
#include "pe.h"

SC_MODULE(Core) {
    sc_in<bool> rst;
    sc_in<bool> clk;

    sc_in<sc_lv<34> > flit_rx;
    sc_in<bool> req_rx;
    sc_out<bool> ack_rx;

    sc_out<sc_lv<34> > flit_tx;
    sc_out<bool> req_tx;
    sc_in<bool> ack_tx;

    PE pe;
    int core_id;

    enum FlitType {
        FLIT_BODY = 0,
        FLIT_TAIL = 1,
        FLIT_HEAD = 2
    };

    void init(int id) {
        core_id = id;
        pe.init(id);
    }

    sc_lv<34> make_head_flit(int source_id, int dest_id) {
        sc_lv<34> flit;
        flit.range(33, 32) = FLIT_HEAD;
        flit.range(31, 16) = dest_id;
        flit.range(15, 0) = source_id;
        return flit;
    }

    sc_lv<34> make_data_flit(float value, bool is_tail) {
        union {
            float fval;
            unsigned int ival;
        } converter;

        converter.fval = value;

        sc_lv<34> flit;
        flit.range(33, 32) = is_tail ? FLIT_TAIL : FLIT_BODY;
        flit.range(31, 0) = converter.ival;
        return flit;
    }

    float get_float_from_flit(const sc_lv<34>& flit) {
        union {
            float fval;
            unsigned int ival;
        } converter;

        converter.ival = flit.range(31, 0).to_uint();
        return converter.fval;
    }

    void tx_thread() {
        req_tx.write(false);
        flit_tx.write(0);
        wait();

        while (true) {
            Packet* packet = pe.get_packet();

            if (packet != NULL) {
                flit_tx.write(make_head_flit(packet->source_id, packet->dest_id));
                req_tx.write(true);
                wait();

                for (size_t i = 0; i < packet->datas.size(); ++i) {
                    bool is_tail = (i + 1 == packet->datas.size());
                    flit_tx.write(make_data_flit(packet->datas[i], is_tail));
                    req_tx.write(true);
                    wait();
                }

                req_tx.write(false);
                delete packet;
                wait();
            } else {
                req_tx.write(false);
                wait();
            }
        }
    }

    void rx_thread() {
        ack_rx.write(true);
        Packet* packet = NULL;
        wait();

        while (true) {
            if (req_rx.read()) {
                sc_lv<34> flit = flit_rx.read();
                unsigned int type = flit.range(33, 32).to_uint();

                if (type == FLIT_HEAD) {
                    if (packet != NULL) {
                        delete packet;
                    }
                    packet = new Packet();
                    packet->dest_id = flit.range(31, 16).to_uint();
                    packet->source_id = flit.range(15, 0).to_uint();
                } else if (packet != NULL) {
                    packet->datas.push_back(get_float_from_flit(flit));

                    if (type == FLIT_TAIL) {
                        pe.check_packet(packet);
                        delete packet;
                        packet = NULL;
                    }
                }
            }
            wait();
        }
    }

    SC_HAS_PROCESS(Core);

    Core(sc_module_name name) : sc_module(name), core_id(0) {
        SC_THREAD(tx_thread);
        sensitive << clk.pos();

        SC_THREAD(rx_thread);
        sensitive << clk.pos();
    }
};

#endif
