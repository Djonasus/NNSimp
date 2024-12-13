package nnsimp

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type FullyConnected struct {
	Layers       []Layer
	LearningRate float64
	L2           float64
	Loss         LossFunction
	Optimizer    Optimizer
}

func (f *FullyConnected) AddLayer(inputSize, outputSize int, activationFunc ActivationFunction, InitFunc LayerIniter) {
	layer := Layer{ActivationFunction: activationFunc}
	layer.Biases = mat.NewVecDense(outputSize, nil)

	layer.Weights = mat.NewDense(inputSize, outputSize, nil)
	InitFunc.Init(&layer)

	f.Layers = append(f.Layers, layer)
}

func (f *FullyConnected) Forward(input *mat.VecDense) (result *mat.VecDense) {
	result = mat.VecDenseCopyOf(input)

	for _, l := range f.Layers {
		_, outN := l.Weights.Dims()
		temp := mat.NewVecDense(outN, nil)

		// Используем транспонированную матрицу весов
		temp.MulVec(l.Weights.T(), result)
		temp.AddVec(temp, l.Biases)
		temp = l.ActivationFunction.Activate(temp)

		result = temp
	}
	return result
}

// func (fc *FullyConnected) DetailedForward(input *mat.VecDense) (result []*mat.VecDense) {
// 	result = make([]*mat.VecDense, len(fc.Layers))

// 	for i, l := range fc.Layers {
// 		_, outN := l.Weights.Dims()
// 		result[i] = mat.NewVecDense(outN, nil)
// 		if i == 0 {
// 			result[i].MulVec(l.Weights, input)
// 		} else {
// 			result[i].MulVec(l.Weights, result[i-1])
// 		}
// 		result[i].AddVec(result[i], l.Biases)
// 		result[i] = l.ActivationFunction.Activate(result[i])
// 	}

// 	return
// }

func (fc *FullyConnected) DetailedForward(input *mat.VecDense) (result []*mat.VecDense) {
	result = make([]*mat.VecDense, len(fc.Layers))
	for i, l := range fc.Layers {
		_, outN := l.Weights.Dims()
		result[i] = mat.NewVecDense(outN, nil)

		if i == 0 {
			result[i].MulVec(l.Weights.T(), input)
		} else {
			result[i].MulVec(l.Weights.T(), result[i-1])
		}
		// fmt.Printf("Слой %d после умножения: %v\n", i, mat.Formatted(result[i]))

		result[i].AddVec(result[i], l.Biases)
		// fmt.Printf("Слой %d после добавления смещения: %v\n", i, mat.Formatted(result[i]))

		result[i] = l.ActivationFunction.Activate(result[i])
		// fmt.Printf("Сл��й %d после активации: %v\n", i, mat.Formatted(result[i]))
	}
	return
}

func (fc *FullyConnected) PrintWeights() {
	for _, l := range fc.Layers {
		fmt.Println(l.Weights)
	}
}

func (fc *FullyConnected) Backward(input, target *mat.VecDense) (loss float64) {
	// Вычисляем выходы и ошибки
	outputs := fc.DetailedForward(input)
	errors := make([]*mat.VecDense, len(fc.Layers))

	// Базовая функция потерь
	loss = fc.Loss.Compute(outputs[len(outputs)-1], target)

	// Добавляем L2 регуляризацию к функции потерь
	for _, l := range fc.Layers {
		rows, cols := l.Weights.Dims()
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				w := l.Weights.At(i, j)
				loss += 0.5 * fc.L2 * w * w
			}
		}
	}

	// Градиенты последнего слоя
	errors[len(fc.Layers)-1] = fc.Loss.Gradient(outputs[len(outputs)-1], target)

	// Распространение ошибки назад
	for i := len(fc.Layers) - 2; i >= 0; i-- {
		// Получаем производную активации
		dActivation := fc.Layers[i].ActivationFunction.Derivative(outputs[i])

		// Вычисляем ошибку для текущего слоя
		rows, _ := fc.Layers[i+1].Weights.Dims()
		errors[i] = mat.NewVecDense(rows, nil)
		errors[i].MulVec(fc.Layers[i+1].Weights, errors[i+1])

		// Применяем производную активации
		for j := 0; j < errors[i].Len(); j++ {
			errors[i].SetVec(j, errors[i].AtVec(j)*dActivation.AtVec(j))
		}
	}

	// Обновление весов
	for i, l := range fc.Layers {
		var inputVec *mat.VecDense
		if i == 0 {
			inputVec = input
		} else {
			inputVec = outputs[i-1]
		}

		rows, cols := l.Weights.Dims()
		gradients := mat.NewDense(rows, cols, nil)

		for j := 0; j < rows; j++ {
			for k := 0; k < cols; k++ {
				gradVal := errors[i].AtVec(k)*inputVec.AtVec(j) +
					fc.L2*l.Weights.At(j, k)
				gradients.Set(j, k, gradVal)
			}
		}

		fc.Optimizer.Update(l.Weights, gradients, fc.LearningRate)

		for j := 0; j < cols; j++ {
			biasGrad := errors[i].AtVec(j)
			l.Biases.SetVec(j, l.Biases.AtVec(j)-fc.LearningRate*biasGrad)
		}
	}

	return loss
}

// Train обучает нейронную сеть на заданном наборе данных
func (fc *FullyConnected) Train(epochs int, inputs []*mat.VecDense, targets []*mat.VecDense) []float64 {
	if len(inputs) != len(targets) {
		panic("Количество входных данных не соответствует количеству целевых значений")
	}

	// Сохраняем ошибки для каждой эпохи
	lossHistory := make([]float64, epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0

		// Проходим по всем обучающим примерам
		for i := range inputs {
			loss := fc.Backward(inputs[i], targets[i])
			totalLoss += loss
		}

		// Вычисляем среднюю ошибку за эпоху
		avgLoss := totalLoss / float64(len(inputs))
		lossHistory[epoch] = avgLoss

		// Выводим прогресс каждые n эпох
		if (epoch+1)%100 == 0 || epoch == 0 {
			fmt.Printf("Эпоха %d/%d, ошибка: %f\n", epoch+1, epochs, avgLoss)
		}
	}

	return lossHistory
}

// TrainBatch обучает нейронную сеть с использованием мини-батчей
func (fc *FullyConnected) TrainBatch(epochs int, batchSize int, inputs []*mat.VecDense, targets []*mat.VecDense) []float64 {
	if len(inputs) != len(targets) {
		panic("Количество входных данных не соответствует количеству целевых значений")
	}

	lossHistory := make([]float64, epochs)
	numBatches := len(inputs) / batchSize

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0

		// Перемешиваем индексы для случайного выбора примеров
		indices := make([]int, len(inputs))
		for i := range indices {
			indices[i] = i
		}
		rand.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})

		// Проходим по мини-батчам
		for batch := 0; batch < numBatches; batch++ {
			batchLoss := 0.0

			// Обрабатываем каждый пример в батче
			for i := 0; i < batchSize; i++ {
				idx := indices[batch*batchSize+i]
				loss := fc.Backward(inputs[idx], targets[idx])
				batchLoss += loss
			}

			totalLoss += batchLoss
		}

		// Вычисляем среднюю ошибку за эпоху
		avgLoss := totalLoss / float64(len(inputs))
		lossHistory[epoch] = avgLoss

		// Выводим прогресс каждые n эпох
		if (epoch+1)%10 == 0 || epoch == 0 {
			fmt.Printf("Эпоха %d/%d, ошибка: %f\n", epoch+1, epochs, avgLoss)
		}
	}

	return lossHistory
}

// func (fc *FullyConnected) TrainBatch(epochs int, batchSize int, inputs []*mat.VecDense, targets []*mat.VecDense) []float64 {
// 	if len(inputs) != len(targets) {
// 		panic("Количество входных данных не соответствует количеству целевых значений")
// 	}

// 	fmt.Printf("Starting batch training with %d samples, batch size: %d\n", len(inputs), batchSize)
// 	lossHistory := make([]float64, epochs)
// 	numBatches := len(inputs) / batchSize

// 	for epoch := 0; epoch < epochs; epoch++ {
// 		fmt.Printf("Starting epoch %d/%d\n", epoch+1, epochs)
// 		totalLoss := 0.0

// 		// Перемешиваем индексы для случайного выбора примеров
// 		indices := make([]int, len(inputs))
// 		for i := range indices {
// 			indices[i] = i
// 		}
// 		rand.Shuffle(len(indices), func(i, j int) {
// 			indices[i], indices[j] = indices[j], indices[i]
// 		})

// 		// Проходим по мини-батчам
// 		for batch := 0; batch < numBatches; batch++ {
// 			if batch%10 == 0 {
// 				fmt.Printf("Processing batch %d/%d\n", batch+1, numBatches)
// 			}

// 			batchLoss := 0.0

// 			// Обрабатываем каждый пример в батче
// 			for i := 0; i < batchSize; i++ {
// 				idx := indices[batch*batchSize+i]
// 				loss := fc.Backward(inputs[idx], targets[idx])

// 				if math.IsNaN(loss) {
// 					fmt.Printf("NaN loss detected at epoch %d, batch %d, sample %d\n",
// 						epoch+1, batch+1, i+1)
// 					continue
// 				}

// 				batchLoss += loss
// 			}

// 			totalLoss += batchLoss
// 		}

// 		// Вычисляем среднюю ошибку за эпоху
// 		avgLoss := totalLoss / float64(len(inputs))
// 		lossHistory[epoch] = avgLoss

// 		fmt.Printf("Epoch %d/%d completed, average loss: %f\n", epoch+1, epochs, avgLoss)
// 	}

// 	return lossHistory
// }
