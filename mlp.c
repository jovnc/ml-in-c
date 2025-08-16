#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SEED 1000
#define LEARNING_RATE 1e-1
#define EPSILON 1e-1

typedef struct {
    float or_w1;
    float or_w2;
    float or_b;
    
    float and_w1;
    float and_w2;
    float and_b;
    
    float nand_w1;
    float nand_w2;
    float nand_b;
} Xor;

typedef float sample[3];

float forward(Xor, float, float);
float sigmoid(float);
float mean_squared_error(Xor);
Xor rand_xor();
void update_weights(Xor *, Xor);
Xor finite_diff(Xor);

sample xor_train[] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}
};

#define TRAIN_COUNT (sizeof(xor_train) / sizeof(xor_train[0]))

sample *train = xor_train;

int main() {
    srand(SEED);
    Xor m = rand_xor();

    for (int i = 0; i < 1000 * 1000; i++) {
        Xor grad = finite_diff(m);
        update_weights(&m, grad);
        if (i % 1000 == 0) {
            float loss = mean_squared_error(m);
            printf("Iteration %d: Loss = %f\n", i, loss);
        }
    }

    printf("Final Loss = %f\n", mean_squared_error(m));
    
    // Verify results with training data
    for (size_t i = 0; i < TRAIN_COUNT; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = train[i][2];
        float output = forward(m, x1, x2);
        printf("Input: (%.1f, %.1f) - Expected: %.1f, Got: %.1f\n", x1, x2, y, output);
    }
    return 0;
}


/*
* Update weights based on gradient
*/
void update_weights(Xor *xor, Xor grad) {
    xor->or_w1 -= LEARNING_RATE * grad.or_w1;
    xor->or_w2 -= LEARNING_RATE * grad.or_w2;
    xor->or_b -= LEARNING_RATE * grad.or_b;

    xor->and_w1 -= LEARNING_RATE * grad.and_w1;
    xor->and_w2 -= LEARNING_RATE * grad.and_w2;
    xor->and_b -= LEARNING_RATE * grad.and_b;

    xor->nand_w1 -= LEARNING_RATE * grad.nand_w1;
    xor->nand_w2 -= LEARNING_RATE * grad.nand_w2;
    xor->nand_b -= LEARNING_RATE * grad.nand_b;
}

/*
* Find gradient using finite diff over multiple perceptrons
*/
Xor finite_diff(Xor xor) {
    Xor grad;
    
    float c = mean_squared_error(xor);
    float saved;

    saved = xor.or_w1;
    xor.or_w1 += EPSILON;
    grad.or_w1 = (mean_squared_error(xor) - c) / EPSILON;
    xor.or_w1 = saved;

    saved = xor.or_w2;
    xor.or_w2 += EPSILON;
    grad.or_w2 = (mean_squared_error(xor) - c) / EPSILON;
    xor.or_w2 = saved;

    saved = xor.or_b;
    xor.or_b += EPSILON;
    grad.or_b = (mean_squared_error(xor) - c) / EPSILON;
    xor.or_b = saved;

    saved = xor.and_w1;
    xor.and_w1 += EPSILON;
    grad.and_w1 = (mean_squared_error(xor) - c) / EPSILON;
    xor.and_w1 = saved;

    saved = xor.and_w2;
    xor.and_w2 += EPSILON;
    grad.and_w2 = (mean_squared_error(xor) - c) / EPSILON;
    xor.and_w2 = saved;

    saved = xor.and_b;
    xor.and_b += EPSILON;
    grad.and_b = (mean_squared_error(xor) - c) / EPSILON;
    xor.and_b = saved;

    saved = xor.nand_w1;
    xor.nand_w1 += EPSILON;
    grad.nand_w1 = (mean_squared_error(xor) - c) / EPSILON;
    xor.nand_w1 = saved;

    saved = xor.nand_w2;
    xor.nand_w2 += EPSILON;
    grad.nand_w2 = (mean_squared_error(xor) - c) / EPSILON;
    xor.nand_w2 = saved;

    saved = xor.nand_b;
    xor.nand_b += EPSILON;
    grad.nand_b = (mean_squared_error(xor) - c) / EPSILON;
    xor.nand_b = saved;

    return grad;
}

/*
 * Generate a random float between 0 and 1
 */
float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

/*
 * Generate random weights and biases for the XOR gate
 */
Xor rand_xor() {
    Xor xor;
    xor.or_w1 = rand_float();
    xor.or_w2 = rand_float();
    xor.or_b = rand_float();

    xor.and_w1 = rand_float();
    xor.and_w2 = rand_float();
    xor.and_b = rand_float();

    xor.nand_w1 = rand_float();
    xor.nand_w2 = rand_float();
    xor.nand_b = rand_float();

    return xor;
}

/*
 * Forward pass through the network
 */
float forward(Xor xor, float x1, float x2) {
    float a = sigmoid(x1 * xor.or_w1 + x2 * xor.or_w2 + xor.or_b);
    float b = sigmoid(x1 * xor.and_w1 + x2 * xor.and_w2 + xor.and_b);
    return sigmoid(a * xor.nand_w1 + b * xor.nand_w2 + xor.nand_b);
}

/*
 * Calculate the mean squared error for a given weight
 * Cost function
 */
float mean_squared_error(Xor xor) {
    float total_error = 0.0f;
    for (size_t i = 0; i < TRAIN_COUNT; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(xor, x1, x2);
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
