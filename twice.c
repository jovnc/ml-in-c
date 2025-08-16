#include <stdio.h>
#include <stdlib.h>

#define SEED 120
#define LEARNING_RATE 0.1f
#define EPSILON 1e-5

// Sample training data: y = 2 * x where w = 2
float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8}
};

#define TRAIN_COUNT (sizeof(train) / sizeof(train[0]))

float rand_float(void);
float mean_squared_error(float, float);

int main() {
    srand(SEED);
    float w = rand_float() * 10.0f;
    float b = rand_float() * 5.0f;

    // start training loop
    for (int i = 0; i < 100; i++) {
        float c = mean_squared_error(w, b);

        // Calculate gradients
        float dw = (mean_squared_error(w + EPSILON, b) - c) / EPSILON;
        float db = (mean_squared_error(w, b + EPSILON) - c) / EPSILON;

        w -= LEARNING_RATE * dw;
        b -= LEARNING_RATE * db;
        printf("Loss: %f\n", mean_squared_error(w, b));
    }
    printf("Final weight: %f\n", w);
    printf("Final bias: %f\n", b);
    return 0;
}

/*
 * Generate a random float between 0 and 1
 */
float rand_float(void) {
    return (float) rand() / (float) RAND_MAX;
}

/*
 * Calculate the mean squared error for a given weight
 * Cost function
 */
float mean_squared_error(float w, float b) {
    float total_error = 0.0f;
    for (size_t i = 0; i < TRAIN_COUNT; i++) {
        float x = train[i][0];
        float y = x * w + b;
        float error = train[i][1] - y;
        total_error += error * error;
    }
    return total_error / (float) TRAIN_COUNT;
}
