#include "clockreset.h"
#include "core.h"
#include "router.h"
#include "systemc.h"
#include "controller.h"
#include "pe.h"
#include "DRAM.h"
#include "AXI4DMA.h"
#include "MemoryMap.h"

#include <sstream>
#include <string>

int sc_main(int argc, char* argv[])
{
    sc_signal<bool> clk;
    sc_signal<bool> rst;

    sc_signal<sc_lv<34> > router_in_flit[16][ROUTER_PORTS];
    sc_signal<bool>       router_in_req[16][ROUTER_PORTS];
    sc_signal<bool>       router_out_ack[16][ROUTER_PORTS];

    sc_signal<sc_lv<34> > dummy_flit[16][ROUTER_PORTS];
    sc_signal<bool>       dummy_req[16][ROUTER_PORTS];
    sc_signal<bool>       dummy_ack[16][ROUTER_PORTS];

    sc_signal<sc_lv<34> > router_to_core_flit[16];
    sc_signal<bool>       router_to_core_req[16];
    sc_signal<bool>       core_to_router_ack[16];

    sc_signal<sc_lv<34> > router0_to_controller_flit;
    sc_signal<bool>       router0_to_controller_req;
    sc_signal<bool>       controller_to_router0_ack;

    Clock      m_clock("m_clock", 10);
    Reset      m_reset("m_reset", 15);

    DRAM       m_dram("m_dram", DRAM_TOTAL_WORDS);
    AXI4DMA    m_dma("m_dma");
    Controller m_controller("m_controller");

    m_dma.bind_dram(&m_dram);
    m_controller.bind_dma(&m_dma);

    Router* routers[16];
    Core*   cores[16];

    for (int i = 0; i < 16; i++) {
        std::stringstream router_name;
        router_name << "router_" << i;
        routers[i] = new Router(router_name.str().c_str(), i);

        std::stringstream core_name;
        core_name << "core_" << i;
        cores[i] = new Core(core_name.str().c_str(), i);
    }

    m_clock.clk(clk);
    m_reset.rst(rst);

    m_controller.clk(clk);
    m_controller.rst(rst);

    // External controller/global-SRAM adapter connects to router 0 CTRL_PORT.
    m_controller.flit_tx(router_in_flit[0][CTRL_PORT]);
    m_controller.req_tx(router_in_req[0][CTRL_PORT]);
    m_controller.ack_tx(router_out_ack[0][CTRL_PORT]);

    m_controller.flit_rx(router0_to_controller_flit);
    m_controller.req_rx(router0_to_controller_req);
    m_controller.ack_rx(controller_to_router0_ack);

    for (int r = 0; r < 16; r++) {
        routers[r]->clk(clk);
        routers[r]->rst(rst);

        for (int p = 0; p < ROUTER_PORTS; p++) {
            routers[r]->in_flit[p](router_in_flit[r][p]);
            routers[r]->in_req[p](router_in_req[r][p]);
            routers[r]->out_ack[p](router_out_ack[r][p]);
        }
    }

    for (int r = 0; r < 16; r++) {
        int x = r % 4;
        int y = r / 4;

        if (x < 3) {
            int nbr = r + 1;
            routers[r]->out_flit[EAST](router_in_flit[nbr][WEST]);
            routers[r]->out_req[EAST](router_in_req[nbr][WEST]);
            routers[r]->in_ack[EAST](router_out_ack[nbr][WEST]);
        } else {
            routers[r]->out_flit[EAST](dummy_flit[r][EAST]);
            routers[r]->out_req[EAST](dummy_req[r][EAST]);
            routers[r]->in_ack[EAST](dummy_ack[r][EAST]);
        }

        if (x > 0) {
            int nbr = r - 1;
            routers[r]->out_flit[WEST](router_in_flit[nbr][EAST]);
            routers[r]->out_req[WEST](router_in_req[nbr][EAST]);
            routers[r]->in_ack[WEST](router_out_ack[nbr][EAST]);
        } else {
            routers[r]->out_flit[WEST](dummy_flit[r][WEST]);
            routers[r]->out_req[WEST](dummy_req[r][WEST]);
            routers[r]->in_ack[WEST](dummy_ack[r][WEST]);
        }

        if (y < 3) {
            int nbr = r + 4;
            routers[r]->out_flit[SOUTH](router_in_flit[nbr][NORTH]);
            routers[r]->out_req[SOUTH](router_in_req[nbr][NORTH]);
            routers[r]->in_ack[SOUTH](router_out_ack[nbr][NORTH]);
        } else {
            routers[r]->out_flit[SOUTH](dummy_flit[r][SOUTH]);
            routers[r]->out_req[SOUTH](dummy_req[r][SOUTH]);
            routers[r]->in_ack[SOUTH](dummy_ack[r][SOUTH]);
        }

        if (y > 0) {
            int nbr = r - 4;
            routers[r]->out_flit[NORTH](router_in_flit[nbr][SOUTH]);
            routers[r]->out_req[NORTH](router_in_req[nbr][SOUTH]);
            routers[r]->in_ack[NORTH](router_out_ack[nbr][SOUTH]);
        } else {
            routers[r]->out_flit[NORTH](dummy_flit[r][NORTH]);
            routers[r]->out_req[NORTH](dummy_req[r][NORTH]);
            routers[r]->in_ack[NORTH](dummy_ack[r][NORTH]);
        }

        // Every router LOCAL port connects to a compute PE core.
        routers[r]->out_flit[LOCAL](router_to_core_flit[r]);
        routers[r]->out_req[LOCAL](router_to_core_req[r]);
        routers[r]->in_ack[LOCAL](core_to_router_ack[r]);

        // Only router 0 has an active external controller/global-SRAM port.
        if (r == 0) {
            routers[r]->out_flit[CTRL_PORT](router0_to_controller_flit);
            routers[r]->out_req[CTRL_PORT](router0_to_controller_req);
            routers[r]->in_ack[CTRL_PORT](controller_to_router0_ack);
        } else {
            routers[r]->out_flit[CTRL_PORT](dummy_flit[r][CTRL_PORT]);
            routers[r]->out_req[CTRL_PORT](dummy_req[r][CTRL_PORT]);
            routers[r]->in_ack[CTRL_PORT](dummy_ack[r][CTRL_PORT]);
        }
    }

    for (int i = 0; i < 16; i++) {
        cores[i]->clk(clk);
        cores[i]->rst(rst);

        cores[i]->flit_rx(router_to_core_flit[i]);
        cores[i]->req_rx(router_to_core_req[i]);
        cores[i]->ack_rx(core_to_router_ack[i]);

        cores[i]->flit_tx(router_in_flit[i][LOCAL]);
        cores[i]->req_tx(router_in_req[i][LOCAL]);
        cores[i]->ack_tx(router_out_ack[i][LOCAL]);
    }

    for (int r = 0; r < 16; r++) {
        for (int p = 0; p < ROUTER_PORTS; p++) {
            dummy_ack[r][p].write(false);
            dummy_req[r][p].write(false);
            dummy_flit[r][p].write(0);
        }
    }

    string data_path = "./data/";

    const char* env_file = getenv("IMAGE_FILE_NAME");
    string image_file = env_file ? env_file : "cat.txt";

    m_dram.load_file(data_path + image_file, DRAM_INPUT_BASE);

    m_dram.load_file(data_path + "conv1_weight.txt", DRAM_W1_BASE);
    m_dram.load_file(data_path + "conv2_weight.txt", DRAM_W2_BASE);
    m_dram.load_file(data_path + "conv3_weight.txt", DRAM_W3_BASE);
    m_dram.load_file(data_path + "conv4_weight.txt", DRAM_W4_BASE);
    m_dram.load_file(data_path + "conv5_weight.txt", DRAM_W5_BASE);

    m_dram.load_file(data_path + "fc6_weight.txt", DRAM_W6_BASE);
    m_dram.load_file(data_path + "fc7_weight.txt", DRAM_W7_BASE);
    m_dram.load_file(data_path + "fc8_weight.txt", DRAM_W8_BASE);

    m_dram.load_file(data_path + "conv1_bias.txt", DRAM_BIAS_BASE + 1 * DRAM_BIAS_STRIDE);
    m_dram.load_file(data_path + "conv2_bias.txt", DRAM_BIAS_BASE + 2 * DRAM_BIAS_STRIDE);
    m_dram.load_file(data_path + "conv3_bias.txt", DRAM_BIAS_BASE + 3 * DRAM_BIAS_STRIDE);
    m_dram.load_file(data_path + "conv4_bias.txt", DRAM_BIAS_BASE + 4 * DRAM_BIAS_STRIDE);
    m_dram.load_file(data_path + "conv5_bias.txt", DRAM_BIAS_BASE + 5 * DRAM_BIAS_STRIDE);

    m_dram.load_file(data_path + "fc6_bias.txt", DRAM_BIAS_BASE + 6 * DRAM_BIAS_STRIDE);
    m_dram.load_file(data_path + "fc7_bias.txt", DRAM_BIAS_BASE + 7 * DRAM_BIAS_STRIDE);
    m_dram.load_file(data_path + "fc8_bias.txt", DRAM_BIAS_BASE + 8 * DRAM_BIAS_STRIDE);

    // Exclude DRAM initialization from DMA/DRAM runtime statistics if supported.
    m_dma.reset_stats();
    m_dram.read_count = 0;
    m_dram.write_count = 0;

    sc_start();
    return 0;
}
