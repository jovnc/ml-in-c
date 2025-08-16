#!/bin/sh

# Compile the C programs
clang -Wall -Wextra -o mlp mlp.c
clang -Wall -Wextra -o gates gates.c
clang -Wall -Wextra -o twice twice.c

# Run the compiled program (choose any)
./mlp