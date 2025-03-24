# Importación de clases
from classes.perceptron import Perceptron
from classes.dataset import Dataset
from classes.evaluator import NeuralEvaluator

# definicion de la funcion main
def main():
  # Declaración de la instancia de la clase Dataset
  dataset = Dataset()
  # Llamado al metodo get data para almacenar los valores de la base de conocimiento
  trainning_data, expected_outputs = dataset.get_data()

  # Declaración de la instancia de la clase Perceptron
  perceptron = Perceptron(input_size=4, learning_rate=0.1)

  # Entrenamiento de la neurona:
  print(f"\n Entrenando a la neurona")
  epoch = perceptron.train(trainning_data, expected_outputs)
  print(f"\n Entrenamiento de la neurona finalizado con {epoch} iteraciones")

  # Evaluación de la neurona
  print(f"\n evaluación de la neurona")
  evaluator = NeuralEvaluator(perceptron)
  evaluator.evaluate()

if __name__ == "__main__":
  main()