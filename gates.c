#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SEED 1000
#define LEARNING_RATE 1e-1
#define EPSILON 1e-1

// sample training data: OR gate
float train[][3] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1}
};

#define TRAIN_COUNT (sizeof(train) / sizeof(train[0]))

float rand_float(void);
float mean_squared_error(float, float, float);
float sigmoid(float);

int main() {
    srand(SEED);
    float w1 = rand_float();
    float w2 = rand_float();
    float b  = rand_float();

    // start training loop
    for (int i = 0; i < 100*1000; i++) {
        float c = mean_squared_error(w1, w2, b);

        // calculate gradients
        float dw1 = (mean_squared_error(w1 + EPSILON, w2, b) - c) / EPSILON;
        float dw2 = (mean_squared_error(w1, w2 + EPSILON, b) - c) / EPSILON;
        float db = (mean_squared_error(w1, w2, b + EPSILON) - c) / EPSILON;

        w1 -= LEARNING_RATE * dw1;
        w2 -= LEARNING_RATE * dw2;
        b -= LEARNING_RATE * db;
        printf("Loss: %f\n", mean_squared_error(w1, w2, b));
    }
    printf("Function: y = %fx1 + %fx2 + %f\n", w1, w2, b);

    // print predictions to verify function
    for (int i = 0; i < 4; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoid(x1 * w1 + x2 * w2 + b);
        printf("Input: (%f, %f) - Predicted: %f\n", x1, x2, y);
    }
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
float mean_squared_error(float w1, float w2, float b) {
    float total_error = 0.0f;
    for (size_t i = 0; i < TRAIN_COUNT; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoid(x1 * w1 + x2 * w2 + b);
        float error = train[i][2] - y;
        total_error += error * error;
    }
    return total_error / (float) TRAIN_COUNT;
}

/*
* Sigmoid activation function
* Maps any real-valued number into the range (0, 1)
*/
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
