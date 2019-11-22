import glob
import tensorflow as tf
import matplotlib.pyplot as plt

def get_all_image_paths(dir_path):
    return glob.glob(dir_path.rstrip("/") + "/*")

def preprocess_image(image):
    image = tf.image.decode_png(image,channels=3)
    image = tf.compat.v2.image.rgb_to_grayscale(image)
    image = tf.image.resize(image,[64,64])
    # image = tf.reshape(image,[192,192])
    image /= 255.0
    image = tf.get_static_value(image)
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def load_and_preprocess_images(paths):
    images = []
    for path in paths:
        images.append(load_and_preprocess_image(path))
    return images

def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()











def main():

    print("test")

if __name__ == "__main__":
    main()

    exit(0)