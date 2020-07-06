# Reconhecimento-Digitos-usando-RedeNeural-Convolucionais

Projeto Bônus da formação Cientista de Dados fornecido pela Data Science Academy (DSA).

Desenvolvimento de um modelo de Deep Learning para classificação de imagens usando Redes Neurais Convolucionais. 
O modelo será criado em Keras e ao final do treinamento vamos salvar o modelo em formato JSON.
Na sequência, criaremos uma aplicação web em Python. 
Nossa aplicação será processada via browser com o Microframework Web chamado Flask.
A aplicação vai reconhecer imagens em tempo real.


# EXECUÇÃO:

- Treinamento do Modelo:

Obs.: Utilizar GPU para o treinamento do modelo. Sem GPU o modelo pode levar horas para ser processado.

$ python train.py

- Execução da app:

$ python app.py

# VERSÕES:

- Keras: 2.09

$ pip install keras==2.09
- Tensorflow: 1.13.2

$ pip install tensorflow==1.13.2
- scipy: 1.2.1

$ pip install scipy==1.2.1
