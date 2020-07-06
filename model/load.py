# Carrega o Modelo

# Imports
import os
import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize, imshow
import tensorflow as tf

# Nível de log
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# Init Function
def init(): 
	json_file = open('C:/WorkSpace/Reconhecimento-Digitos-usando-RedeNeural-Convolucionais/model/model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("C:/WorkSpace/Reconhecimento-Digitos-usando-RedeNeural-Convolucionais/model/model.h5")
	print("Modelo Carregado do Disco")

	# Compila e Avalia o Modelo Carregado
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	graph = tf.get_default_graph()

	return loaded_model, graph