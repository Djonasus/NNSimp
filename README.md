# NNSimp - простая и легковесная библиотека для создания полносвязных моделей ИИ.

NNSimp - простая библиотека для создания нейросетей. Я не математик, не программист-трайхардер, либа создается с исследовательской целью изучения работы нейросетей. Контрибьюции приветствуются (если объясните, что вы вообще сделали...)

Пример использования показан ниже (Простой алгоритм предсказания XOR):

```go
package main

import (
	"fmt"

	nnsimp "github.com/Djonasus/NNSimp"
)

func main() {
	nn := nnsimp.NewNeuralNetwork(0.01, 0.001)
	nn.AddLayer(2, 3, nnsimp.Sigmoid{})
	nn.AddLayer(3, 1, nnsimp.Sigmoid{})

	data := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	nn.Train(data, targets, 10000)

	for _, input := range data {
		output := nn.Forward(input)
		fmt.Printf("Input: %v, Output: %v\n", input, output)
	}
}
```
