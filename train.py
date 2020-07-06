# Treinamento de uma Rede Neural Convolucional

# Imports
import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Nível de log
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# Hiperparâmetros
batch_size = 128
num_classes = 10
epochs = 12

# Dimensões da imagem de input
img_rows, img_cols = 28, 28

# Carrega o dataset de Digitos escritos a mão fornecido pelo Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Formatando os dados
# "channels_last" assume: (conv_dim1, conv_dim2, conv_dim3, channels)
# enquanto
# "channels_first" assume: (channels, conv_dim1, conv_dim2, conv_dim3).
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Normalização
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'amostras de treino')
print(x_test.shape[0], 'amostras de teste')

# Converte vetores de classe para matrizes de classe binária
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Construindo o Modelo
model = Sequential()
model.add(Conv2D(name="convolution2d_1", filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(name="convolution2d_2", filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(name="maxpooling2d_1", pool_size=[2, 2]))
model.add(Dropout(name="dropout_1", rate=0.25))
model.add(Flatten())
model.add(Dense(name="dense_1", units=128, activation='relu'))
model.add(Dropout(name="dropout_2", rate=0.5))
model.add(Dense(name="dense_2", units=num_classes, activation='softmax'))

# Otimização
# Adaptive learning rate (adaDelta) é uma forma popular de descida do gradiente,
# rivalizada apenas com adam e adagrad desde que tenhamos várias classes (10)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# Treinamento
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Acurácia
score = model.evaluate(x_test, y_test, verbose=0)
print('Loss em Teste:', score[0])
print('Acurácia em Teste:', score[1])


# Salvando do Modelo e Serializando com JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Serializando os pesos com HDF5
model.save_weights("model.h5")
print("Modelo salvo em disco")
