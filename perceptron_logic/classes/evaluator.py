# Importación de librerias
import numpy as np

# Definicion de la clase para evaluar la neurona
class NeuralEvaluator:
  # Declaración del constructor de la clase 
  def __init__(self, perceptron):
    # Atributos de la clase
    self.perceptron = perceptron
  
  # Declaración del metodo para solicitar informacion al usuario
  def get_user_inputs(self):
    user_input =[]
    print("\n Ingrese 4 valores entre -1 y 1")
    for i in range(4):
      while True:
        try:
          value = float(input(f"Valor {i+1}: "))
          if -1.0 <= value <= 1.0:
            user_input.append(value)
            break
          else:
            print("Error: Ingrese un valor entre -1 y 1")
        except ValueError:
          print("Error: Ingrese un valor valido")
    return np.array(user_input)
  
  # Declaración del metodo para evaluar la neurona
  def evaluate(self):
    user_input = self.get_user_inputs()
    result = self.perceptron.predict(user_input)
    print(f"\n Salida de la Neurona: {result:.4f}")

