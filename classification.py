# %%
import keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model
import pydot

#%%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# %%
y_train

# %%
## データを 0 - 255 -> 0.0 - 1.0 に変換する
x_train_p = x_train.astype('float32') / 255.0
x_test_p = x_test.astype('float32') / 255.0

## 正解ラベルを one-hot 表現に変換
y_train_o = keras.utils.to_categorical(y_train,10)
y_test_o = keras.utils.to_categorical(y_test,10)

# %%
print(y_train_o.shape)

# %%
# 画像イメージを表示する
plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show()

# %%
# Model を構築する
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

# %%
# Model をコンパイルする
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
# model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

# %%
# 学習を開始する
history = model.fit(x_train_p, y_train_o,batch_size=128, epochs=5)

# %%
plot_model(model, to_file='model.png', show_shapes=True)

# %%
