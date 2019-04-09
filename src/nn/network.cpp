#include "network.h"

NeuralNetwork::NeuralNetwork() {
    this->builder = createInferBuilder(this->logger);
}
