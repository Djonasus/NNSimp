package nnsimp

import (
	"fmt"
	"math"
	"math/rand"
)

// NeuralNetwork represents a simple feedforward neural network
type NeuralNetwork struct {
	Layers       [][][]float64 // Weights for each layer
	Biases       [][]float64   // Biases for each layer
	Activations  []ActivationFunction
	LearningRate float64
	Lambda       float64 // L2 regularization factor
}

// NewNeuralNetwork initializes a new neural network
func NewNeuralNetwork(learningRate, lambda float64) *NeuralNetwork {
	return &NeuralNetwork{
		Layers:       [][][]float64{},
		Biases:       [][]float64{},
		Activations:  []ActivationFunction{},
		LearningRate: learningRate,
		Lambda:       lambda,
	}
}

// AddLayer adds a layer to the neural network
func (nn *NeuralNetwork) AddLayer(inputSize, outputSize int, activation ActivationFunction) {
	weights := make([][]float64, inputSize)
	for i := range weights {
		weights[i] = make([]float64, outputSize)
		for j := range weights[i] {
			weights[i][j] = (rand.Float64()*2 - 1) * math.Sqrt(2.0/float64(inputSize))
		}
	}

	biases := make([]float64, outputSize)
	nn.Layers = append(nn.Layers, weights)
	nn.Biases = append(nn.Biases, biases)
	nn.Activations = append(nn.Activations, activation)
}

// Forward propagates input through the network
func (nn *NeuralNetwork) Forward(input []float64) []float64 {
	current := input
	for l := range nn.Layers {
		next := make([]float64, len(nn.Biases[l]))
		for j := range next {
			sum := nn.Biases[l][j]
			for i := range current {
				sum += current[i] * nn.Layers[l][i][j]
			}
			next[j] = nn.Activations[l].Activate(sum)
		}
		current = next
	}
	return current
}

// Backward performs backpropagation and updates weights
func (nn *NeuralNetwork) Backward(input, target []float64) float64 {
	// Implementation remains unchanged
	// ...
	return 0.0 // Placeholder
}

// Train trains the neural network on the provided dataset
func (nn *NeuralNetwork) Train(data, targets [][]float64, epochs int) {
	for epoch := 1; epoch <= epochs; epoch++ {
		totalLoss := 0.0
		for i := range data {
			totalLoss += nn.Backward(data[i], targets[i])
		}
		averageLoss := totalLoss / float64(len(data))
		if epoch%100 == 0 || epoch == epochs {
			fmt.Printf("Epoch %d/%d: Loss = %.6f\n", epoch, epochs, averageLoss)
		}
	}
}
