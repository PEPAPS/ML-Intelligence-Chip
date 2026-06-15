#ifndef AXI4DMA_H
#define AXI4DMA_H

#include <vector>
#include <iostream>
#include "systemc.h"
#include "DRAM.h"

using namespace std;

// Simplified AXI4 DMA model.
// It models burst transfer length/size and VALID/READY-style accounting.
// Data is moved between DRAM and controller buffers.
SC_MODULE(AXI4DMA) {
    DRAM* dram;

    unsigned int data_bytes;   // AWSIZE/ARSIZE equivalent. float = 4 bytes.
    unsigned int max_burst_len;

    unsigned long long read_bursts;
    unsigned long long write_bursts;
    unsigned long long read_words;
    unsigned long long write_words;

    unsigned long long arvalid_count;
    unsigned long long arready_count;
    unsigned long long rvalid_count;
    unsigned long long rready_count;
    unsigned long long awvalid_count;
    unsigned long long awready_count;
    unsigned long long wvalid_count;
    unsigned long long wready_count;
    unsigned long long bvalid_count;
    unsigned long long bready_count;

    SC_HAS_PROCESS(AXI4DMA);

    AXI4DMA(sc_module_name name)
        : sc_module(name)
    {
        dram = NULL;
        data_bytes = 4;
        max_burst_len = 256;
        reset_stats();
    }

    void bind_dram(DRAM* d);
    void reset_stats();

    vector<float> burst_read(unsigned int addr, unsigned int len);
    void burst_write(unsigned int addr, const vector<float>& data);

    void print_stats();
};

#endif