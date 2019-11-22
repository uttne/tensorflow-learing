# %%
import keras
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, Concatenate
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import plot_model


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