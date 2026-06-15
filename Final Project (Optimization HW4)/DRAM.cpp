#include "DRAM.h"

float DRAM::read(unsigned int addr)
{
    if (addr >= mem.size()) {
        cout << "[DRAM] ERROR: read address out of range: " << addr << endl;
        sc_stop();
        return 0.0f;
    }

    read_count++;
    return mem[addr];
}

void DRAM::write(unsigned int addr, float value)
{
    if (addr >= mem.size()) {
        cout << "[DRAM] ERROR: write address out of range: " << addr << endl;
        sc_stop();
        return;
    }

    write_count++;
    mem[addr] = value;
}

void DRAM::load_file(const string& filename, unsigned int base_addr)
{
    ifstream file(filename.c_str());
    if (!file.is_open()) {
        cout << "[DRAM] ERROR: cannot open file: " << filename << endl;
        sc_stop();
        return;
    }

    float value;
    unsigned int offset = 0;

    while (file >> value) {
        write(base_addr + offset, value);
        offset++;
    }

    file.close();

    cout << "[DRAM] Loaded " << offset
         << " words from " << filename
         << " to base address " << base_addr
         << endl;
}