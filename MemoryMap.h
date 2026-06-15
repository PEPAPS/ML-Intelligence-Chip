#ifndef MEMORY_MAP_H
#define MEMORY_MAP_H

// Word-addressed DRAM map.
// Each address stores one float value.
// The data folder is only used at initialization time to preload these regions.

#define DRAM_INPUT_BASE      0

#define DRAM_W1_BASE         1000000
#define DRAM_W2_BASE         2000000
#define DRAM_W3_BASE         3000000
#define DRAM_W4_BASE         4000000
#define DRAM_W5_BASE         5500000
#define DRAM_W6_BASE         7000000
#define DRAM_W7_BASE         45000000
#define DRAM_W8_BASE         65000000

#define DRAM_BIAS_BASE       75000000
#define DRAM_BIAS_STRIDE     10000

#define DRAM_INTER_BASE      76000000
#define DRAM_INTER_STRIDE    1000000

#define DRAM_OUTPUT_BASE     89000000

#define DRAM_TOTAL_WORDS     90000000

#endif