from django.shortcuts import render

# Importación de librerias
import numpy as np

# Llamado a las clases
from .services import Dataset, Perceptron, NeuralEvaluator

def Home(request):
  mensaje = ""
  otro_mensaje = ""
  # Declaración de la instancia de la clase DataSet
  dataset = Dataset()
  training_data, expected_output = dataset.get_data()
  # combinación de las listas para visualizar la base de conocimientos en la plantilla
  data_combined = [list(row) + [expected_output[idx]] for idx, row in enumerate(training_data)]
  # Declaración de la instancia de la clase Perceptron
  # Intentar recuperar el perceptrón de la sesión
  perceptron = None
  if "perceptron_weights" in request.session:
      weights = np.array(request.session["perceptron_weights"])
      bias = request.session["perceptron_bias"]
      perceptron = Perceptron(input_size=4)
      perceptron.weights = weights
      perceptron.bias = bias
  else:
      perceptron = Perceptron(input_size=4)

  if request.method == 'POST':
      if "boton_mensaje" in request.POST:
          epoch = perceptron.train(training_data, expected_output)
          mensaje = f"Neuron's training finished with {epoch} iterations"
          
          # Guardar pesos en la sesión
          request.session["perceptron_weights"] = perceptron.weights.tolist()
          request.session["perceptron_bias"] = perceptron.bias

      elif "boton_otro" in request.POST:
          user_inputs = np.array([
              float(request.POST.get('input1')),
              float(request.POST.get('input2')),
              float(request.POST.get('input3')),
              float(request.POST.get('input4'))
          ])
          evaluator = NeuralEvaluator(perceptron)
          result = evaluator.evaluate(user_inputs)
          otro_mensaje = f"Neuron Output: {result}"

  return render(request, "pages/home.html",
                {
                    "training_data": data_combined,
                    "mensaje": mensaje,
                    "otro_mensaje": otro_mensaje
                })