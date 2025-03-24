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