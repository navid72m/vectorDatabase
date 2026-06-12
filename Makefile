CC = clang
CFLAGS = -O3 -march=native -fPIC -Wall -Wextra

# Conditionally set LDFLAGS based on OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS = -shared -Wl,-install_name,libvecdb.so
else
    LDFLAGS = -shared
endif

SRC = vecdb.c turboquant.c
OBJ = $(SRC:.c=.o)

all: libvecdb.so

libvecdb.so: $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

bench: bench.c libvecdb.so
	$(CC) $(CFLAGS) -o $@ $< -L. -lvecdb -lm

clean:
	rm -f $(OBJ) libvecdb.so bench
