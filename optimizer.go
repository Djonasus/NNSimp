package nnsimp

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Optimizer interface {
	Update(weights, gradients *mat.Dense, learningRate float64)
}

// SGD (Стохастический градиентный спуск)
type SGD struct{}

func (s *SGD) Update(weights, gradients *mat.Dense, learningRate float64) {
	r, c := weights.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			weights.Set(i, j, weights.At(i, j)-learningRate*gradients.At(i, j))
		}
	}
}

// Оптимизатор Adam
type Adam struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64
	M            []*mat.Dense // Первая моментная переменная
	V            []*mat.Dense // Вторая моментная переменная
	T            int          // Счётчик шагов
}

// Конструктор для Adam.
// Default - NewAdam(0.001, 0.9, 0.999, 1e-8, nn.Layers)
func NewAdam(learningRate, beta1, beta2, epsilon float64, layers []Layer) *Adam {
	adam := &Adam{
		LearningRate: learningRate,
		Beta1:        beta1,
		Beta2:        beta2,
		Epsilon:      epsilon,
		T:            0,
	}

	// Инициализация моментов M и V для каждого слоя
	for _, layer := range layers {
		r, c := layer.Weights.Dims()
		adam.M = append(adam.M, mat.NewDense(r, c, nil))
		adam.V = append(adam.V, mat.NewDense(r, c, nil))
	}

	return adam
}

// Метод обновления весов для Adam
func (a *Adam) Update(weights, gradients *mat.Dense, layerIdx int) {
	r, c := weights.Dims()

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			grad := gradients.At(i, j)

			// Обновление моментов
			mOld := a.M[layerIdx].At(i, j)
			vOld := a.V[layerIdx].At(i, j)

			mNew := a.Beta1*mOld + (1-a.Beta1)*grad
			vNew := a.Beta2*vOld + (1-a.Beta2)*grad*grad

			a.M[layerIdx].Set(i, j, mNew)
			a.V[layerIdx].Set(i, j, vNew)

			// Коррекция смещения
			mHat := mNew / (1 - math.Pow(a.Beta1, float64(a.T+1)))
			vHat := vNew / (1 - math.Pow(a.Beta2, float64(a.T+1)))

			// Обновление весов
			weights.Set(i, j, weights.At(i, j)-a.LearningRate*mHat/(math.Sqrt(vHat)+a.Epsilon))
		}
	}

	a.T++ // Увеличиваем шаг
}
