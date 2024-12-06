package nnsimp

import (
	"math"
	"math/rand"
)

// ActivationFunction - интерфейс стандартизации функции активации (содержит саму функцию и ее диференцированный вид)
type ActivationFunction interface {
	Activate(x float64) float64
	Derive(x float64) float64
}

// Сигмоида - имплементация интерфейса ActivationFunction
type Sigmoid struct{}

func (Sigmoid) Activate(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (Sigmoid) Derive(x float64) float64 {
	fx := 1.0 / (1.0 + math.Exp(-x))
	return fx * (1 - fx)
}

// NeuralNetwork - полносвязная нейронная сеть
type NeuralNetwork struct {
	Layers       [][][]float64 // Весы каждого слоя
	Biases       [][]float64   // Смещение слоев
	Activations  []ActivationFunction
	LearningRate float64
	Lambda       float64 // L2-регуляризация
}

// NewNeuralNetwork - функция инициализации НС
func NewNeuralNetwork(learningRate, lambda float64) *NeuralNetwork {
	return &NeuralNetwork{
		Layers:       [][][]float64{},
		Biases:       [][]float64{},
		Activations:  []ActivationFunction{},
		LearningRate: learningRate,
		Lambda:       lambda,
	}
}

// AddLayer позволяет добавить слой
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

// Forward - прогон НС
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

// Backward - обратное распространение
func (nn *NeuralNetwork) Backward(input, target []float64) float64 {
	// Forward pass
	outputs := [][]float64{input}
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
		outputs = append(outputs, next)
		current = next
	}

	// Compute loss and gradients
	deltas := make([][]float64, len(nn.Layers))
	loss := 0.0
	for j := range outputs[len(outputs)-1] {
		d := outputs[len(outputs)-1][j] - target[j]
		loss += d * d
	}
	loss /= float64(len(target))

	for l := len(nn.Layers) - 1; l >= 0; l-- {
		deltas[l] = make([]float64, len(nn.Biases[l]))
		for j := range deltas[l] {
			err := 0.0
			if l == len(nn.Layers)-1 {
				err = outputs[len(outputs)-1][j] - target[j]
			} else {
				for k := range deltas[l+1] {
					err += deltas[l+1][k] * nn.Layers[l+1][j][k]
				}
			}
			deltas[l][j] = err * nn.Activations[l].Derive(outputs[l+1][j])
			for i := range outputs[l] {
				nn.Layers[l][i][j] -= nn.LearningRate * (deltas[l][j]*outputs[l][i] + nn.Lambda*nn.Layers[l][i][j])
			}
			nn.Biases[l][j] -= nn.LearningRate * deltas[l][j]
		}
	}
	return loss
}
