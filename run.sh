#!/bin/sh

# Compile the C programs
clang -Wall -Wextra -o build/mlp mlp.c
clang -Wall -Wextra -o build/gates gates.c
clang -Wall -Wextra -o build/twice twice.c

# Run the compiled program (choose any)
./build/mlp