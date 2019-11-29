# https://hajimerobot.co.jp/ai/cnn_heatmap/

# %%
import os

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras import optimizers
from keras.applications.vgg16 import VGG16


# %%
# 分類クラス
classes = ['daisy', 'dandelion','rose','sunflower','tulip']
nb_classes = len(classes)
batch_size_for_data_generator = 20

base_dir = "./heatmap_sample"

test_dir = os.path.join(base_dir, 'test_images')

test_daisy_dir = os.path.join(test_dir, 'daisy')
test_dandelion_dir = os.path.join(test_dir, 'dandelion')
test_rose_dir = os.path.join(test_dir, 'rose')
test_sunflower_dir = os.path.join(test_dir, 'sunflower')
test_tulip_dir = os.path.join(test_dir, 'tulip')

# 画像サイズ
img_rows, img_cols = 200, 200

# %%
input_tensor = Input(shape=(img_rows, img_cols, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
#vgg16.summary()

# %%
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
model.summary()


# %%
model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])


# %%
hdf5_file = os.path.join(base_dir, 'flower-model.hdf5')
model.load_weights(hdf5_file)

# %%
