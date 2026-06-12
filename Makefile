CC      ?= cc
SRC      = vecdb.c turboquant.c
OBJ      = $(SRC:.c=.o)

# Per-platform tuning: -march=native is rejected by clang on Apple Silicon;
# the AArch64 equivalent is -mcpu=native.
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S)-$(UNAME_M),Darwin-arm64)
    ARCHFLAGS = -mcpu=native
    LDSHARED  = -shared -Wl,-install_name,@rpath/libvecdb.so
else
    ARCHFLAGS = -march=native
    LDSHARED  = -shared
endif

CFLAGS  ?= -O3 $(ARCHFLAGS) -fPIC -Wall -Wextra

all: libvecdb.so

libvecdb.so: $(OBJ)
	$(CC) $(LDSHARED) -o $@ $^ -lm

%.o: %.c vecdb.h turboquant.h
	$(CC) $(CFLAGS) -c $< -o $@

# bench is compiled statically from sources: no rpath/LD_LIBRARY_PATH needed
bench: $(SRC) bench.c vecdb.h turboquant.h
	$(CC) $(CFLAGS) -o $@ $(SRC) bench.c -lm

test: libvecdb.so
	python3 tests.py

clean:
	rm -f $(OBJ) libvecdb.so bench

.PHONY: all test clean
