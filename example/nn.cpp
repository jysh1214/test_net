#include "FuckNet.h"

static Layer* layer_0 = new ConnectedLayer(2, "INPUT_DATA");
static Layer* layer_1 = new ConnectedLayer(3, "tanh");
static Layer* layer_2 = new ConnectedLayer(3, "tanh");
static Layer* layer_3 = new ConnectedLayer(1, "tanh");

static tensor* input = new tensor(1, 2, 1); // input: [a_1, a_2, ...]
static tensor* target = new tensor(1, 1, 1);

int main()
{
    Net network;
    network.learningRate = 0.001;
    network.lossFunction = "MSE";
    network.layers.push_back(layer_0);
    network.layers.push_back(layer_1);
    network.layers.push_back(layer_2);
    network.layers.push_back(layer_3);

    input->data[0] = 3.5;
    input->data[1] = 0.6;

    target->data[0] = 1;
    network.target = target;

    size_t epoch = 10;
    network.train(input, epoch);
    network.predict(input);

    return 0;
}
