#ifndef DRAM_H
#define DRAM_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "systemc.h"

using namespace std;

// Behavioral external DRAM model.
// This is word-addressed: one address stores one float.
SC_MODULE(DRAM) {
    vector<float> mem;

    unsigned long long read_count;
    unsigned long long write_count;

    SC_HAS_PROCESS(DRAM);

    DRAM(sc_module_name name, unsigned int num_words = 100000000)
        : sc_module(name)
    {
        mem.resize(num_words, 0.0f);
        read_count = 0;
        write_count = 0;
    }

    float read(unsigned int addr);
    void write(unsigned int addr, float value);
    void load_file(const string& filename, unsigned int base_addr);
};

#endif