import os

# for windows
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

print("tf version:", tf.__version__)

def save_image(image, filename):
  """
    Saves unscaled Tensor Images.
    Args:
      image: 3D image tensor. [height, width, channels]
      filename: Name of the file to save to.
  """
  if not isinstance(image, Image.Image):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  image.save("%s.jpg" % filename)
  print("Saved as %s.jpg" % filename)

def plot_image(image, title=""):
  """
    Plots images from image tensors.
    Args:
      image: 3D image tensor. [height, width, channels].
      title: Title to display in the plot.
  """
  image = np.asarray(image)
  image = tf.clip_by_value(image, 0, 255)
  image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  print('call imshow: ', title)
  plt.imshow(image)
  plt.axis("off")
  plt.title(title)
  plt.show() # to shart a window

IMAGE_PATH = "original.png"

# Load image and preprocess
hr_image = tf.image.decode_image(tf.io.read_file(IMAGE_PATH))
# If PNG, remove the alpha channel. The model only supports
# images with 3 color channels.
if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
hr_image = tf.cast(hr_image, tf.float32)
hr_image = tf.expand_dims(hr_image, 0)

# plot_image(tf.squeeze(hr_image), title="original")

t1 = time.time()
# download and uncompress the model on
# https://tfhub.dev/captain-pool/esrgan-tf2/1
SAVED_MODEL_PATH = "esrgan-tf2_1"
model = hub.load(SAVED_MODEL_PATH)
gen_image = model(hr_image)
t2 = time.time()
print('model process time:', t2 - t1)

# plot_image(tf.squeeze(gen_image), title="output")
save_image(tf.squeeze(gen_image), filename="output")
