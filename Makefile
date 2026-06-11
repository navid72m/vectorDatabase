CC = clang
CFLAGS = -O3 -march=native -fPIC -Wall -Wextra
LDFLAGS = -shared -Wl,-install_name,libvecdb.so

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
