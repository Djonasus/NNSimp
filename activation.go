package nnsimp

import "math"

// ActivationFunction defines the interface for activation functions
type ActivationFunction interface {
	Activate(x float64) float64
	Derive(x float64) float64
}

// Sigmoid activation function
type Sigmoid struct{}

func (Sigmoid) Activate(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (Sigmoid) Derive(x float64) float64 {
	fx := 1.0 / (1.0 + math.Exp(-x))
	return fx * (1 - fx)
}
