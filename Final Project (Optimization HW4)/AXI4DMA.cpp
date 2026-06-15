#include "AXI4DMA.h"

void AXI4DMA::bind_dram(DRAM* d)
{
    dram = d;
}

void AXI4DMA::reset_stats()
{
    read_bursts = 0;
    write_bursts = 0;
    read_words = 0;
    write_words = 0;

    arvalid_count = 0;
    arready_count = 0;
    rvalid_count = 0;
    rready_count = 0;
    awvalid_count = 0;
    awready_count = 0;
    wvalid_count = 0;
    wready_count = 0;
    bvalid_count = 0;
    bready_count = 0;
}

vector<float> AXI4DMA::burst_read(unsigned int addr, unsigned int len)
{
    vector<float> out;
    out.reserve(len);

    if (dram == NULL) {
        cout << "[AXI4DMA] ERROR: DRAM is not bound." << endl;
        sc_stop();
        return out;
    }

    unsigned int remaining = len;
    unsigned int current_addr = addr;

    while (remaining > 0) {
        unsigned int burst_len =
            remaining > max_burst_len ? max_burst_len : remaining;

        // AXI read address channel: ARVALID/ARREADY.
        arvalid_count++;
        arready_count++;
        read_bursts++;

        for (unsigned int i = 0; i < burst_len; i++) {
            // AXI read data channel: RVALID/RREADY.
            rvalid_count++;
            rready_count++;

            out.push_back(dram->read(current_addr + i));
            read_words++;
        }

        current_addr += burst_len;
        remaining -= burst_len;
    }

    return out;
}

void AXI4DMA::burst_write(unsigned int addr, const vector<float>& data)
{
    if (dram == NULL) {
        cout << "[AXI4DMA] ERROR: DRAM is not bound." << endl;
        sc_stop();
        return;
    }

    unsigned int remaining = data.size();
    unsigned int current_addr = addr;
    unsigned int data_index = 0;

    while (remaining > 0) {
        unsigned int burst_len =
            remaining > max_burst_len ? max_burst_len : remaining;

        // AXI write address channel: AWVALID/AWREADY.
        awvalid_count++;
        awready_count++;
        write_bursts++;

        for (unsigned int i = 0; i < burst_len; i++) {
            // AXI write data channel: WVALID/WREADY.
            wvalid_count++;
            wready_count++;

            dram->write(current_addr + i, data[data_index + i]);
            write_words++;
        }

        // AXI write response channel: BVALID/BREADY.
        bvalid_count++;
        bready_count++;

        current_addr += burst_len;
        data_index += burst_len;
        remaining -= burst_len;
    }
}

void AXI4DMA::print_stats()
{
    cout << "===== AXI4 DMA / DRAM Statistics =====" << endl;
    cout << "Read bursts: " << read_bursts << endl;
    cout << "Write bursts: " << write_bursts << endl;
    cout << "Read words: " << read_words << endl;
    cout << "Write words: " << write_words << endl;
    cout << "Read volume MB: "
         << (read_words * data_bytes) / (1024.0 * 1024.0)
         << endl;
    cout << "Write volume MB: "
         << (write_words * data_bytes) / (1024.0 * 1024.0)
         << endl;
    cout << "ARVALID/ARREADY: "
         << arvalid_count << "/" << arready_count << endl;
    cout << "RVALID/RREADY: "
         << rvalid_count << "/" << rready_count << endl;
    cout << "AWVALID/AWREADY: "
         << awvalid_count << "/" << awready_count << endl;
    cout << "WVALID/WREADY: "
         << wvalid_count << "/" << wready_count << endl;
    cout << "BVALID/BREADY: "
         << bvalid_count << "/" << bready_count << endl;
    cout << "=====================================" << endl;
}