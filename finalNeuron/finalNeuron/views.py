"""
View para el entrenamiento y evaluación de la neurona de cuatro entradas
"""
from django.shortcuts import render
import numpy as np
from .services import Dataset, Perceptron, NeuralEvaluator
# Variable global para mantener la percistencia del perceptron durante la ejecución
global_perceptron = None

def Home(request):
	
	global global_perceptron
	mensaje = ""
	otro_mensaje = ""

	# Carga y preparación de la data de entrenamiento
	dataset = Dataset()
	training_data, expected_output = dataset.get_data()
	data_combined = [list(row) + [expected_output[idx]] for idx, row in enumerate(training_data)]
	
	# Inicializar el perceptron si aun no existe
	if global_perceptron is None:
		global_perceptron = Perceptron(input_size=4)

	# Manejo de peticiones POST
	if request.method == 'POST':
			# Entrenamiento de la neurona
			if "boton_mensaje" in request.POST:
					# Entrenamiento de la neurona
					epoch = global_perceptron.train(training_data, expected_output)
					mensaje = f"Neuron's training finished with {epoch} iterations"

			# Evaluación de la neurona
			elif "boton_otro" in request.POST:
					if global_perceptron is None:
						otro_mensaje = "Error: Train the neuron first!"
					else:
						try:
							user_inputs = np.array([
								float(request.POST.get('input1')),
								float(request.POST.get('input2')),
								float(request.POST.get('input3')),
								float(request.POST.get('input4'))
							])
							evaluator = NeuralEvaluator(global_perceptron)
							result = evaluator.evaluate(user_inputs)
							otro_mensaje = f"Neuron Output: {result}"
						except (TypeError, ValueError):
							otro_mensaje = "Error: Invalid input values"

	return render(request, "pages/home.html",
								{
										"training_data": data_combined,
										"mensaje": mensaje,
										"otro_mensaje": otro_mensaje
								})