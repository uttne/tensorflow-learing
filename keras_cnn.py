# %%
import keras
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt

# %%
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# %%
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=0.175)


# %%
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# %%
# 配列を要素1の配列の配列に変換している
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_valid /= 255
x_test /= 255

# %%
y_train = keras.utils.to_categorical(y_train, 10)
y_valid = keras.utils.to_categorical(y_valid, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# %%

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=128, epochs=10,
                    verbose=1, validation_data=(x_valid, y_valid))


# %%
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %%
plt.plot(history.history['accuracy'], marker='.', label='acc')
plt.plot(history.history['val_accuracy'], marker='.', label='val_acc')
plt.title('model accuracy')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()


# %%
print(history.history)

# %%
print(x_train.shape)


# %%
x_train

# %%
x_train.shape


# %%
a = x_train.shape[0]
a

# %%
b = x_train.reshape(x_train.shape[0], 28, 28, 1)
b.shape


# %%
b
