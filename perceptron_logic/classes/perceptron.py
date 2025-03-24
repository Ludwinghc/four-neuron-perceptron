# Importaciónde librerias
import numpy as np

# Declaración de la clase del perceptron
class Perceptron:
  # Declaración del constructor de la clase
  def __init__(self, input_size, learning_rate = 0.1, epochs = 100):
    # Atributos de la clase del Perceptron
    self.input_size = input_size
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.weights = np.random.uniform(-1, 1, input_size)
    self.bias =  np.random.uniform(-1,1)
  
  # Declaración del metodo para la función de activación
  def activation_function(self, x):
    # Devuelve el valor aplicado de la TANGENTE HIPERBOLICA al resultado de la operación del perceptron
    return np.tanh(x)
  
  # Declaración del metodo para calcular la salida de la neurona
  def predict(self, inputs):
    # Operación del algoritmo del perceptron
    net_input = np.dot(inputs, self.weights) + self.bias
    return self.activation_function(net_input)
  
  # Declaración del metodo para el entrenamiento de la neurona
  def train(self, training_data, expected_output):
    for epoch in range(self.epochs):
      total_error = 0
      
      for inputs, expected in zip(training_data, expected_output):
        output = self.predict(inputs)
        error = expected - output
        self.weights += self.learning_rate * error * np.array(inputs)
        self.bias += self.learning_rate * error
        total_error = abs(error)
      if total_error < 1e-5:
        return epoch + 1
        break 