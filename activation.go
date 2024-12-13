package nnsimp

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// ActivationFunction интерфейс для активационных функций
type ActivationFunction interface {
	Activate(input *mat.VecDense) *mat.VecDense
	Derivative(input *mat.VecDense) *mat.VecDense
}

// Функция активации ReLU
type ReLU struct{}

func (r *ReLU) Activate(input *mat.VecDense) *mat.VecDense {
	result := mat.VecDenseCopyOf(input)

	for i := range result.Len() {
		val := input.AtVec(i)
		if val > 0 {
			result.SetVec(i, val)
		} else {
			result.SetVec(i, 0)
		}
	}

	return result
}

func (r *ReLU) Derivative(input *mat.VecDense) *mat.VecDense {
	// Производная ReLU: 1 для положительных значений, 0 для отрицательных
	result := mat.VecDenseCopyOf(input)

	for i := range result.Len() {
		if input.AtVec(i) > 0 {
			result.SetVec(i, 1)
		} else {
			result.SetVec(i, 0)
		}
	}

	return result
}

// Функция активации Sigmoid
type Sigmoid struct{}

func (s *Sigmoid) Activate(input *mat.VecDense) *mat.VecDense {
	result := mat.VecDenseCopyOf(input)
	for i := range result.Len() {
		result.SetVec(i, 1.0/(1.0+math.Exp(-result.AtVec(i))))
	}
	return result
}

func (s *Sigmoid) Derivative(input *mat.VecDense) *mat.VecDense {
	// Производная сигмоиды: sigmoid(x) * (1 - sigmoid(x))
	result := mat.VecDenseCopyOf(input)

	for i := range result.Len() {
		val := input.AtVec(i)
		sigmoidVal := 1.0 / (1.0 + math.Exp(-val))
		result.SetVec(i, sigmoidVal*(1.0-sigmoidVal))
	}

	return result
}

// Функция активации гиперболического тангенса
type Tanh struct{}

func (t *Tanh) Activate(input *mat.VecDense) *mat.VecDense {
	// Применяем гиперболический тангенс к каждому элементу матрицы
	result := mat.VecDenseCopyOf(input)

	for i := range result.Len() {
		val := input.AtVec(i)
		result.SetVec(i, math.Tanh(val))
	}

	return result
}

func (t *Tanh) Derivative(input *mat.VecDense) *mat.VecDense {
	// Производная гиперболического тангенса: 1 - tanh(x)^2
	result := mat.VecDenseCopyOf(input)

	for i := range result.Len() {
		val := input.AtVec(i)
		tanhVal := math.Tanh(val)
		result.SetVec(i, 1.0-tanhVal*tanhVal)
	}

	return result
}

// Функция активации Softmax
type Softmax struct{}

func (s *Softmax) Activate(input *mat.VecDense) *mat.VecDense {
	result := mat.VecDenseCopyOf(input)

	// Находим максимальное значение для численной стабильности
	maxVal := input.AtVec(0)
	for i := 1; i < input.Len(); i++ {
		if input.AtVec(i) > maxVal {
			maxVal = input.AtVec(i)
		}
	}

	// Вычисляем экспоненты и их сумму
	sum := 0.0
	for i := 0; i < result.Len(); i++ {
		exp := math.Exp(input.AtVec(i) - maxVal) // Вычитаем максимум для стабильности
		result.SetVec(i, exp)
		sum += exp
	}

	// Нормализуем значения
	for i := 0; i < result.Len(); i++ {
		result.SetVec(i, result.AtVec(i)/sum)
	}

	return result
}

func (s *Softmax) Derivative(input *mat.VecDense) *mat.VecDense {
	// Для Softmax производная сложнее, так как каждый выход зависит от всех входов
	// Для упрощения можно использовать приближение
	result := mat.VecDenseCopyOf(input)

	// Сначала получаем значения Softmax
	softmaxOutput := s.Activate(input)

	// Приближенная производная: yi * (1 - yi)
	for i := 0; i < result.Len(); i++ {
		yi := softmaxOutput.AtVec(i)
		result.SetVec(i, yi*(1-yi))
	}

	return result
}
