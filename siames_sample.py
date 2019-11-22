# %%
import numpy as np
import random
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D
from keras.optimizers import RMSprop
from keras import backend as K
import my_util as mu

# %%
# 関数の定義
def euclidean_distance(vects):
    '''
    特徴量の距離の計算
    '''
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square,K.epsilon()))

def contrastive_loss(y_true, y_pred):
    '''
    損失関数
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def create_pairs(positive_images, negative_images):
    '''
    Positive と Negative のペアを作成する
    '''
    pairs = []
    labels = []
    negative_n = len(negative_images)
    n = len(positive_images) - 1 - negative_n
    
    for i in range(n):
        for j in range(negative_n):
            pairs += [[positive_images[i],positive_images[i+1+j]]]
            pairs += [[positive_images[i],negative_images[j]]]
            labels += [1,0]
    return np.array(pairs), np.array(labels)

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    return Model(input, x)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

# %%
tr_positive_image_paths = mu.get_all_image_paths("./datasets/bottle_datasets/train/positive")
tr_negative_image_paths = mu.get_all_image_paths("./datasets/bottle_datasets/train/negative")
te_positive_image_paths = mu.get_all_image_paths("./datasets/bottle_datasets/test/positive")
te_negative_image_paths = mu.get_all_image_paths("./datasets/bottle_datasets/test/negative")

tr_positive_images = mu.load_and_preprocess_images(tr_positive_image_paths)
tr_negative_images = mu.load_and_preprocess_images(tr_negative_image_paths)
te_positive_images = mu.load_and_preprocess_images(te_positive_image_paths)
te_negative_images = mu.load_and_preprocess_images(te_negative_image_paths)

# mu.show_image(tr_positive_images[0])
# mu.show_image(tr_negative_images[0])
# %%
tr_pairs, tr_y = create_pairs(tr_positive_images, tr_negative_images)
te_pairs, te_y = create_pairs(te_positive_images, te_negative_images)

# %%
input_shape = tr_pairs[0].shape[1:]
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

# distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a,processed_b])
distance = Lambda(euclidean_distance)([processed_a,processed_b])

model = Model([input_a, input_b], distance)

# %%
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

# %%
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
            batch_size=128,
            epochs=10,
            validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))