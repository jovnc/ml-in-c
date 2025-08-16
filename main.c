#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
float mean_squared_error(float);
float calculate_gradient(float, float (*)(float));

int main() {
    srand(SEED);
    float w = rand_float() * 10.0f;

    // start training loop
    for (int i = 0; i < 100; i++) {
        float grad = calculate_gradient(w, mean_squared_error);
        w -= LEARNING_RATE * grad; // update weight in opposite direction of gradient
        printf("Loss: %f\n", mean_squared_error(w));
    }
    printf("Final weight: %f\n", w);
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
float mean_squared_error(float w) {
    float total_error = 0.0f;
    for (size_t i = 0; i < TRAIN_COUNT; i++) {
        float x = train[i][0];
        float y = x * w;
        float error = train[i][1] - y;
        total_error += error * error;
    }
    return total_error / (float) TRAIN_COUNT;
}

/*
* Calculate derivative (gradient) of function
* Simplification using finite difference
* Formula: f'(x) = (f(x + h) - f(x)) / h
*/
float calculate_gradient(float w, float (*function)(float)) {
    return (function(w + EPSILON) - function(w)) / EPSILON;
}