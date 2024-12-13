package nnsimp

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	ActivationFunction ActivationFunction
	Weights            *mat.Dense
	Biases             *mat.VecDense
}

type LayerIniter interface {
	Init(*Layer)
}

// Случайная инициализация
type Random struct{}

func (Random) Init(l *Layer) {

	r, c := l.Weights.Dims()

	for i := range r {
		for j := range c {
			l.Weights.Set(i, j, rand.NormFloat64()*0.01)
		}
	}
}

// Инициализация Ксавье
type Xavier struct{}

func (Xavier) Init(l *Layer) {

	r, c := l.Weights.Dims()

	limit := math.Sqrt(6.0 / float64(r+c))
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			l.Weights.Set(i, j, rand.Float64()*2*limit-limit) // Случайные значения в диапазоне [-limit, limit]
		}
	}

}

type Zero struct{}

func (Zero) Init(l *Layer) {
	r, c := l.Weights.Dims()
	l.Weights = mat.NewDense(r, c, nil)
}
