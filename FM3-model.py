#i!/usr/bin/env python

#print ('test1')
import numpy as np
import matplotlib.cm as cm

import h5py 
import pickle
import matplotlib.pyplot as plt
from keras.layers import SpatialDropout2D
#print ('test2')
import tensorflow as tf 
from tensorflow import keras
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D, BatchNormalization, Conv2D,UpSampling2D, Reshape, ReLU, Layer 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate, Add, Softmax, Multiply
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Multiply, Concatenate, Reshape, Add, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model, load_model
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import gaussian_filter


class GaussianSmoothing(Layer):
	def __init__(self, kernel_size=3, **kwargs):
		super(GaussianSmoothing, self).__init__(**kwargs)
		self.kernel_size = kernel_size

	def build(self, input_shape):
        # Create Gaussian kernel
		self.sigma = self.kernel_size / 2.0
		x = tf.range(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1)
		x = tf.cast(x, dtype=tf.float32)
		gauss = tf.exp(-x**2 / (2.0 * self.sigma**2))
		gauss /= tf.reduce_sum(gauss)
		gauss_kernel = tf.tensordot(gauss, gauss, axes=0)
		gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
		self.gauss_kernel = tf.tile(gauss_kernel, [1, 1, input_shape[-1], 1])
		self.gauss_kernel = tf.Variable(self.gauss_kernel, trainable=False)

	def call(self, inputs):
		smoothed = tf.nn.depthwise_conv2d(inputs, self.gauss_kernel, strides=[1, 1, 1, 1], padding='SAME')
		return smoothed

	def compute_output_shape(self, input_shape):
		return input_shape

	def get_config(self):
		config = super(GaussianSmoothing, self).get_config()
		config.update({'kernel_size': self.kernel_size})
		return config

def multiscale_smoothing(input_layer, kernel_sizes):
	smoothed_images = [GaussianSmoothing(kernel_size=size)(input_layer) for size in kernel_sizes]
	combined_image = Concatenate(axis=-1)(smoothed_images)
	return combined_image


# Define Gaussian smoothing kernel sizes
#kernel_sizes = [4, 8, 12, 16]
kernel_sizes = [2, 4, 6, 8]
#kernel_sizes = [3, 5, 7]


fileinp = '../../../training.h5'
fild = h5py.File(fileinp, 'r')
s1 = np.array(fild['sen1'])
print (s1.shape)

s2 = np.array(fild['sen2'])
print (s2.shape)

lab = np.array(fild['label'])
print (lab.shape)


n = 352366
X1 = s1.reshape(n,32,32,8)
X2 = s2.reshape(n,32,32,10)
Y = lab.reshape(n,17)

n = s1.shape[0]
n_classes = 17


# Define input shapes
input_shape_s1 = (32, 32, 8)  # MS input (Ensure correct shape)
input_shape_s2 = (32, 32, 10)  # SAR input (Ensure correct shape)


# Define input layers
input_layer_s1 = Input(shape=input_shape_s1, name="input_s1")  # MS input
input_layer_s2 = Input(shape=input_shape_s2, name="input_s2")  # SAR input


smoothed_s1 = multiscale_smoothing(input_layer_s1, kernel_sizes)
print(smoothed_s1)
print(smoothed_s1.shape)

smoothed_s2 = multiscale_smoothing(input_layer_s2, kernel_sizes)
print (smoothed_s2)

concatenated_raw_data = Concatenate()([smoothed_s1, smoothed_s2])

pix = Conv2D(32, 3, padding='same', activation='relu', name="conv_pix")(concatenated_raw_data)
pix = BatchNormalization(name="bn_pix")(pix)
pix = SpatialDropout2D(0.2)(pix)
pix_flattened = GlobalAveragePooling2D(name="gap_pix")(pix)

x_s1 = Conv2D(32, 3, activation='relu', padding='same', name="conv_s1")(smoothed_s1)
x_s1 = BatchNormalization(name="bn_s1")(x_s1)
x_s1 = SpatialDropout2D(0.2)(x_s1)

x_s2 = Conv2D(32, 3, activation='relu', padding='same', name="conv_s2")(smoothed_s2)
x_s2 = BatchNormalization(name="bn_s2")(x_s2)
x_s2 = SpatialDropout2D(0.2)(x_s2)

fusion = Multiply()([x_s1, x_s2])
fusion = Conv2D(64, 3, padding='same')(fusion)
fusion = BatchNormalization()(fusion)
fusion = ReLU()(fusion)
fusion = GlobalAveragePooling2D()(fusion)

combined_features = Concatenate(name="concat_features")([pix_flattened, fusion])
combined_features = Dense(64, activation='relu')(combined_features)
output = Dense(17, activation='softmax')(combined_features)

model = Model(inputs=[input_layer_s1, input_layer_s2], outputs=output)

model.summary()

print(Y.shape)
history = model.fit([X1, X2], Y, epochs=100)

model.save("FM3final-100.h5")

with tf.keras.utils.custom_object_scope({'GaussianSmoothing': GaussianSmoothing}):
	model = load_model("FM3final-100.h5")

# === Load tes:t data ===
filetestinp = '../../../testing.h5'
filtd = h5py.File(filetestinp, 'r')
s1_test = np.array(filtd['sen1'])  # (n, 32, 32, 8)
s2_test = np.array(filtd['sen2'])  # (n, 32, 32, 10)
true_labels = np.array(filtd['label'])  # (n, 17)
print("Shapes:", s1_test.shape, s2_test.shape, true_labels.shape)

ntest = s1_test.shape[0]

y_pred = model.predict([s1_test, s2_test], batch_size=128)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(true_labels, axis=1)

accuracy = accuracy_score(y_true_labels, y_pred_labels)
precision = precision_score(y_true_labels, y_pred_labels, average='macro')
recall = recall_score(y_true_labels, y_pred_labels, average='macro')
f1 = f1_score(y_true_labels, y_pred_labels, average='macro')
kappa = cohen_kappa_score(y_true_labels, y_pred_labels)

print("\n=== Overall Metrics ===")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"Kappa     : {kappa:.4f}")
print("\n=== Classification Report (Per Class) ===")
print(classification_report(y_true_labels, y_pred_labels, digits=3))

conf_mat = confusion_matrix(y_true_labels, y_pred_labels)
print("\n=== Confusion Matrix ===")
print(conf_mat)

conf_mat_percent = conf_mat.astype(np.float32) / conf_mat.sum(axis=1, keepdims=True) * 100






