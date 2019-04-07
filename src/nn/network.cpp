#include "network.h"

NeuralNetwork::NeuralNetwork() {
    google::InitGoogleLogging("lilypad");
    this->builder = createInferBuilder(this->logger);
}
