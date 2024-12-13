package nnsimp

import (
	"encoding/json"
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)

// NetworkWeights структура для сериализации весов
type NetworkWeights struct {
	LayerWeights []LayerWeights `json:"layers"`
}

// LayerWeights структура для хранения весов одного слоя
type LayerWeights struct {
	Weights [][]float64 `json:"weights"`
	Biases  []float64   `json:"biases"`
}

// SaveWeights сохраняет веса сети в JSON файл
func (fc *FullyConnected) SaveWeights(filename string) error {
	networkWeights := NetworkWeights{
		LayerWeights: make([]LayerWeights, len(fc.Layers)),
	}

	// Преобразуем веса каждого слоя в формат для JSON
	for i, layer := range fc.Layers {
		rows, cols := layer.Weights.Dims()
		weights := make([][]float64, rows)

		// Копируем веса
		for r := 0; r < rows; r++ {
			weights[r] = make([]float64, cols)
			for c := 0; c < cols; c++ {
				weights[r][c] = layer.Weights.At(r, c)
			}
		}

		// Копируем смещения
		biases := make([]float64, layer.Biases.Len())
		for j := 0; j < layer.Biases.Len(); j++ {
			biases[j] = layer.Biases.AtVec(j)
		}

		networkWeights.LayerWeights[i] = LayerWeights{
			Weights: weights,
			Biases:  biases,
		}
	}

	// Создаем файл
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Записываем JSON
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "    ") // Красивое форматирование
	return encoder.Encode(networkWeights)
}

// LoadWeights загружает веса сети из JSON файла
func (fc *FullyConnected) LoadWeights(filename string) error {
	// Открываем файл
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Читаем JSON
	var networkWeights NetworkWeights
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&networkWeights); err != nil {
		return err
	}

	// Проверяем соответствие количества слоев
	if len(networkWeights.LayerWeights) != len(fc.Layers) {
		return fmt.Errorf("количество слоев в файле (%d) не соответствует структуре сети (%d)",
			len(networkWeights.LayerWeights), len(fc.Layers))
	}

	// Загружаем веса в каждый слой
	for i, layerWeights := range networkWeights.LayerWeights {
		rows := len(layerWeights.Weights)
		if rows == 0 {
			return fmt.Errorf("слой %d: отсутствуют веса", i)
		}
		cols := len(layerWeights.Weights[0])

		// Проверяем размерности
		currentRows, currentCols := fc.Layers[i].Weights.Dims()
		if rows != currentRows || cols != currentCols {
			return fmt.Errorf("слой %d: несоответствие размерностей весов", i)
		}

		// Загружаем веса
		weights := mat.NewDense(rows, cols, nil)
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				weights.Set(r, c, layerWeights.Weights[r][c])
			}
		}
		fc.Layers[i].Weights = weights

		// Проверяем размерность смещений
		if len(layerWeights.Biases) != fc.Layers[i].Biases.Len() {
			return fmt.Errorf("слой %d: несоответствие размерностей смещений", i)
		}

		// Загружаем смещения
		biases := mat.NewVecDense(len(layerWeights.Biases), layerWeights.Biases)
		fc.Layers[i].Biases = biases
	}

	return nil
}
