# %%
import keras
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, Concatenate
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import plot_model
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.patheffects as PathEffects
import seaborn as sns
import my_util as mu
import random


# %%
(tr_img,tr_label),(te_img,te_label) = keras.datasets.mnist.load_data()

tr_img = tr_img.astype('float32')
te_img = te_img.astype('float32')

tr_img /= 255
te_img /= 255

reshape_tr_img = tr_img.reshape([tr_img.shape[0]] + list(tr_img.shape[1:]) + [1])
reshape_te_img = te_img.reshape([te_img.shape[0]] + list(te_img.shape[1:]) + [1])

# %%
plt.imshow(tr_img[0])
plt.title(tr_label[0])
plt.show()

# %%
# トレーニング画像の散布図を描画
tsne = TSNE()
tsne_embeds = tsne.fit_transform(te_img.reshape(-1,te_img.shape[1] * te_img.shape[2])[:512])
mu.scatter(tsne_embeds, te_label[:512], 10, "MNIST test image scatter")

# %%
def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu')(input)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input,x)

def euclidean_distance(vects):
    x,y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.square(K.maximum(sum_square,K.epsilon()))

def contrastive_loss(y_true, y_pred):
    margin = 1
    positive_d = y_pred[:,0]
    negative_d = y_pred[:,1]
    return K.maximum(positive_d - negative_d + margin,0)

def create_set(images, labels):
    digit_indices = [np.where(labels == i)[0] for i in range(10)]

    image_set = []
    anckor_label = []

    for class_index in range(10):
        indices = digit_indices[class_index] 
        for i in indices[:512]:
            anckor_img = images[i]
            positive_i = indices[(random.randrange(1,len(indices)) + i) % len(indices)]
            positive_img = images[positive_i]

            negative_indices = digit_indices[(random.randrange(1,10) + class_index) % 10]
            negative_i = negative_indices[random.randrange(0,len(negative_indices))]
            negative_img = images[negative_i]
            image_set += [[anckor_img, positive_img, negative_img]]
            anckor_label += [class_index]

    return np.array(image_set), np.array(anckor_label)

# %%
image_set, anckor_label = create_set(reshape_tr_img,tr_label)
te_image_set, te_anckor_label = create_set(reshape_te_img,te_label)

# %%
image_set.shape

# %%
# 画像セットの確認
img = image_set[127][0]
plt.imshow(img.reshape(img.shape[0],img.shape[1]))
plt.title(anckor_label[0])
plt.show()
img = image_set[127][1]
plt.imshow(img.reshape(img.shape[0],img.shape[1]))
plt.title(anckor_label[0])
plt.show()
img = image_set[127][2]
plt.imshow(img.reshape(img.shape[0],img.shape[1]))
plt.title(anckor_label[0])
plt.show()

# %%
input_shape = reshape_tr_img.shape[1:]

base_network = create_base_network(input_shape)

input_anchor = Input(shape=input_shape)
input_positive = Input(shape=input_shape)
input_negative = Input(shape=input_shape)

processed_anchor = base_network(input_anchor)
processed_positive = base_network(input_positive)
processed_negative = base_network(input_negative)

distance_positive = Lambda(euclidean_distance)([processed_anchor,processed_positive])
distance_negative = Lambda(euclidean_distance)([processed_anchor,processed_negative])

concat = Concatenate(axis=1)([distance_positive, distance_negative])

model = Model([input_anchor,input_positive,input_negative],concat)
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)

plot_model(model, to_file='model.png', show_shapes=True)

# %%
history = model.fit([image_set[:,0], image_set[:,1], image_set[:,2]], anckor_label, batch_size=128, epochs=10, validation_data=([te_image_set[:,0], te_image_set[:,1], te_image_set[:,2]], te_anckor_label))

# %%
pred = model.predict([image_set[:,0], image_set[:,1], image_set[:,2]])
pred.shape

# %%
pred[0]

# %%
output_layer = None
input_layer = None
layer = model.layers[3]
print(layer.name)
output_layer = layer.get_output_at(0)
print(output_layer)
input_layer = layer.get_input_at(0)
print(input_layer)

active_model = Model(inputs=input_layer,outputs=output_layer)
active_model.summary()

# %%
pred = active_model.predict([reshape_te_img[:512]])

# %%
tsne = TSNE()
train_tsne_embeds = tsne.fit_transform(pred)
mu.scatter(train_tsne_embeds, te_label[:512], 10, "triplet test scatter")



# %%
