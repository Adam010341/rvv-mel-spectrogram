CC      = riscv64-unknown-linux-gnu-gcc
SPIKE   = spike --isa=RV64GCV_Zicntr
PK      = $(RISCV)/riscv64-unknown-linux-gnu/bin/pk

INCLUDES      = -Iinclude
CFLAGS_BASE   = -static -march=rv64gcv -Wall $(INCLUDES)
CFLAGS        = -O3 -fno-tree-vectorize $(CFLAGS_BASE)
BUILD  = build
OUTPUT = output

BENCH  = src/bench.c
UTILS  = src/utils.c
MAIN   = src/main.c
HEADER = include/mel_spectrogram.h

.PHONY: compile run judge clean

compile: $(BUILD)/bench

$(BUILD)/bench: $(BENCH) $(MAIN) $(UTILS) $(HEADER)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(BENCH) $(MAIN) $(UTILS) -lm

run: compile
	@mkdir -p $(OUTPUT)
	$(SPIKE) $(PK) $(BUILD)/bench > $(OUTPUT)/results.csv
	@cat $(OUTPUT)/results.csv

judge: run
	uv run python -m scripts.judge $(OUTPUT)/results.csv

clean:
	rm -rf $(BUILD) $(OUTPUT)
