package nnsimp

import (
	"encoding/json"
	"os"
)

// SaveWeights saves the network's weights and biases to a file
func (nn *NeuralNetwork) SaveWeights(filename string) error {
	data := map[string]interface{}{
		"layers": nn.Layers,
		"biases": nn.Biases,
	}
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	return encoder.Encode(data)
}

// LoadWeights loads the network's weights and biases from a file
func (nn *NeuralNetwork) LoadWeights(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	data := map[string]interface{}{}
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&data)
	if err != nil {
		return err
	}
	nn.Layers = data["layers"].([][][]float64)
	nn.Biases = data["biases"].([][]float64)
	return nil
}
