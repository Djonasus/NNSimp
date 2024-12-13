package nnsimp

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type LossFunction interface {
	Compute(predicted, target *mat.VecDense) float64        // Вычисление значения функции потерь
	Gradient(predicted, target *mat.VecDense) *mat.VecDense // Градиент по предсказанию
}

// MSE (Среднеквадратичная ошибка)
type MSE struct{}

func (m *MSE) Compute(predicted, target *mat.VecDense) float64 {
	diff := mat.NewVecDense(predicted.Len(), nil)
	diff.SubVec(predicted, target)
	return mat.Dot(diff, diff) / float64(diff.Len())
}

func (m *MSE) Gradient(predicted, target *mat.VecDense) *mat.VecDense {
	diff := mat.NewVecDense(predicted.Len(), nil)
	diff.SubVec(predicted, target)
	diff.ScaleVec(2.0/float64(diff.Len()), diff) // Градиент MSE: 2 * (predicted - target) / N
	return diff
}

// CrossEntropy (Перекрестная энтропия)
type CrossEntropy struct{}

// func (ce *CrossEntropy) Compute(predicted, target *mat.VecDense) float64 {
// 	if predicted.Len() != target.Len() {
// 		panic("CrossEntropy: размерности предсказанного и целевого векторов не совпадают")
// 	}

// 	var loss float64
// 	epsilon := 1e-15 // Маленькое число для предотвращения log(0)

// 	for i := 0; i < predicted.Len(); i++ {
// 		// Ограничиваем предсказанные значения, чтобы избежать log(0)
// 		p := math.Max(math.Min(predicted.AtVec(i), 1-epsilon), epsilon)
// 		t := target.AtVec(i)
// 		loss += -t * math.Log(p)
// 	}

// 	return loss / float64(predicted.Len())
// }

// func (ce *CrossEntropy) Gradient(predicted, target *mat.VecDense) *mat.VecDense {
// 	if predicted.Len() != target.Len() {
// 		panic("CrossEntropy gradient: размерности предсказанного и целевого векторов не совпадают")
// 	}

// 	gradient := mat.NewVecDense(predicted.Len(), nil)
// 	epsilon := 1e-15

// 	for i := 0; i < predicted.Len(); i++ {
// 		// Ограничиваем предсказанные значения
// 		p := math.Max(math.Min(predicted.AtVec(i), 1-epsilon), epsilon)
// 		t := target.AtVec(i)
// 		// Градиент: -target/predicted
// 		gradient.SetVec(i, -t/p)
// 	}

// 	// Нормализуем градиент
// 	gradient.ScaleVec(1/float64(predicted.Len()), gradient)
// 	return gradient
// }

// Compute вычисляет значение функции потерь
func (c *CrossEntropy) Compute(output, target *mat.VecDense) float64 {
	var sum float64
	epsilon := 1e-15 // Малое значение для предотвращения log(0)

	for i := 0; i < output.Len(); i++ {
		y := output.AtVec(i)
		t := target.AtVec(i)

		// Ограничиваем y, чтобы избежать log(0)
		y = math.Max(math.Min(y, 1-epsilon), epsilon)

		// Отладочная информация
		if math.IsNaN(y) || math.IsNaN(t) {
			fmt.Printf("NaN detected in Compute: y=%v, t=%v\n", y, t)
		}

		sum += -t * math.Log(y)
	}

	return sum
}

// Gradient вычисляет градиент функции потерь
func (c *CrossEntropy) Gradient(output, target *mat.VecDense) *mat.VecDense {
	grad := mat.NewVecDense(output.Len(), nil)
	epsilon := 1e-15 // Малое значение для предотвращения деления на 0

	for i := 0; i < output.Len(); i++ {
		y := output.AtVec(i)
		t := target.AtVec(i)

		// Ограничиваем y, чтобы избежать деления на 0
		y = math.Max(math.Min(y, 1-epsilon), epsilon)

		// Отладочная информация
		if math.IsNaN(y) || math.IsNaN(t) {
			fmt.Printf("NaN detected in Gradient: y=%v, t=%v\n", y, t)
		}

		grad.SetVec(i, -t/y)
	}

	return grad
}

// BinaryCrossEntropy для бинарной классификации
type BinaryCrossEntropy struct{}

func (bce *BinaryCrossEntropy) Compute(predicted, target *mat.VecDense) float64 {
	if predicted.Len() != target.Len() {
		panic("BinaryCrossEntropy: размерности предсказанного и целевого векторов не совпадают")
	}

	var loss float64
	epsilon := 1e-15

	for i := 0; i < predicted.Len(); i++ {
		p := math.Max(math.Min(predicted.AtVec(i), 1-epsilon), epsilon)
		t := target.AtVec(i)
		loss += -t*math.Log(p) - (1-t)*math.Log(1-p)
	}

	return loss / float64(predicted.Len())
}

func (bce *BinaryCrossEntropy) Gradient(predicted, target *mat.VecDense) *mat.VecDense {
	if predicted.Len() != target.Len() {
		panic("BinaryCrossEntropy gradient: размерности предсказанного и целевого векторов не совпадают")
	}

	gradient := mat.NewVecDense(predicted.Len(), nil)
	epsilon := 1e-15

	for i := 0; i < predicted.Len(); i++ {
		p := math.Max(math.Min(predicted.AtVec(i), 1-epsilon), epsilon)
		t := target.AtVec(i)
		gradient.SetVec(i, -t/p+(1-t)/(1-p))
	}

	gradient.ScaleVec(1/float64(predicted.Len()), gradient)
	return gradient
}
