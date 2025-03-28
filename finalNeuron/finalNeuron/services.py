# Importación de librerias
import numpy as np

# Declaración de la clase para la base de conocimiento
class Dataset:
  # definicion del constructor de la clase Dataset
  def __init__(self):
    # Atributos de entrenamiento
    self.training_data = np.array([
      [-1, -1, -1, -1],
      [-1, -1, -1, 1],
      [-1, -1, 1, -1],
      [-1, -1, 1, 1],
      [-1, 1, -1, -1],
      [-1, 1, -1, 1],
      [-1, 1, 1, -1],
      [-1, 1, 1, 1],
      [1, -1, -1, -1],
      [1, -1, -1, 1],
      [1, -1, 1, -1],
      [1, -1, 1, 1],
      [1, 1, -1, -1],
      [1, 1, -1, 1],
      [1, 1, 1, -1],
      [1, 1, 1, 1]
      ])
    # Atributos de salida esperada
    self.expected_outputs = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1])
  
  # Metodo para obtener los datos de entrenamiento y sus respectivas salidas
  def get_data(self):
    return self.training_data, self.expected_outputs
  

# Declaración de la clase del perceptron
class Perceptron:
  # Declaración del constructor de la clase
  def __init__(self, input_size, learning_rate = 0.1, epochs = 100):
    # Atributos de la clase del Perceptron
    self.input_size = input_size
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.weights = np.random.uniform(-1, 1, input_size)
    self.bias = np.random.uniform(-1,1)
  
  # Declaración del metodo para la función de activación
  def activation_function(self, x):
    # Devuelve el valor aplicado de la TANGENTE HIPERBOLICA al resultado de la operación del perceptron
    return np.tanh(x)
  
  # Declaración del metodo para Redondear el resultado
  def  threshold(self, x):
    return 1 if x >= 0 else -1

  # Declaración del metodo para calcular la salida de la neurona
  def predict(self, inputs):
    # Operación del algoritmo del perceptron
    net_input = np.dot(inputs, self.weights) + self.bias
    raw_output = self.activation_function(net_input)
    return self.threshold(raw_output)
  
  # Declaración del metodo para el entrenamiento de la neurona
  def train(self, training_data, expected_output):
    for epoch in range(self.epochs):
      print(f"Epoca # {epoch}")
      total_error = 0
      for inputs, expected in zip(training_data, expected_output):
        output = self.predict(inputs)
        error = expected - output
        self.weights += self.learning_rate * error * np.array(inputs)
        self.bias += self.learning_rate * error
        total_error += abs(error)
      if total_error < 0.1:
        return epoch + 1
    return self.epochs


# Definicion de la clase para evaluar la neurona
class NeuralEvaluator:
  # Declaración del constructor de la clase 
  def __init__(self, perceptron):
    # Atributos de la clase
    self.perceptron = perceptron
  
  # Declaración del metodo para evaluar la neurona
  def evaluate(self, user_inputs):
    result = self.perceptron.predict(user_inputs)
    print(result)
    return round(result,4)
