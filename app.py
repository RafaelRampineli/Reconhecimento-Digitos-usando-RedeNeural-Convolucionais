# Web App Para Previsão de Dígitos

# Imports
# Usamos o Flask para renderização da página HTML e Scipy para salvar, ler e redimensionar as imagens. 
# Numpy e Keras são usados para a criação do modelo.
# Os pacotes re e base64 são usados para expressões regulares e para tratar dados no formato string.
# Os pacotes sys e os são usados patra manipulações do sistema operacional.
from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
from imageio import imread
# from keras.preprocessing.image import save_img
import numpy as np
import keras.models
import re
import base64
import sys 
import os

# Indicando onde o modelo está salvo
sys.path.append(os.path.abspath("model"))

# carregar todas as funções do arquivo load.py
from load import * 

# Inicializando a app flask
app = Flask(__name__)

# Variáveis globais para reutilização
global model, graph

# Inicializando as variáveis
model, graph = init()


# Decodificando uma imagem do formato base64 em uma representação raw 
def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	with open('output.png','wb') as output:
		output.write(base64.b64decode(imgstr))


# Renderização HTML
@app.route('/')
def index():
	# Renderização da página html
	return render_template("index.html")


@app.route('/predict/', methods=['GET','POST'])
def predict():
	# Sempre que o método de previsão é chamado, entraremos com o dígito desenhado pelo usuário como uma imagem no modelo 
	# executando a inferência e retornaremos a classificação obter o formato de dados brutos da imagem
	imgData = request.get_data()
	
	# Endode em formato que possa ser alimentado no modelo 
	convertImage(imgData)
	
	# Grava a imagem na memória
	x = imread('output.png', mode='L')
	
	# Calcula uma inversão bit-wise onde preto torna-se branco e vice-versa
	x = np.invert(x)
	
	# Redimensiona a imagem para o tamanho que será alimentado no modelo
	x = imresize(x,(28,28))

	# Converte para um tensor 4D e alimenta nosso modelo
	x = x.reshape(1,28,28,1)
	
	# Inicia o grafo computacional
	with graph.as_default():
		# Faz a previsão
		out = model.predict(x)
		print(out)
		print(np.argmax(out, axis=1))
		# Converte a resposta em uma string
		response = np.array_str(np.argmax(out,axis=1))
		return response	
	
# Função Main
if __name__ == "__main__":


	# Decide em que porta executa a app
	port = int(os.environ.get('PORT', 8000))

	# Executa a app localmente
	app.run(host='127.0.0.1', port=port)



