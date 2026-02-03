#i!/usr/bin/env python

#print ('test1')
import numpy as np
import matplotlib.cm as cm

import h5py 
import pickle
import matplotlib.pyplot as plt

#print ('test2')

from tensorflow import keras
from keras.layers import SpatialDropout2D
from tensorflow.keras.layers import MaxPool2D, MaxPooling2D,  GlobalAveragePooling2D, BatchNormalization, Conv2D,UpSampling2D,  Conv2DTranspose 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Lambda
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt


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


# Define the shape of individual image modalities
input_shape_s1 = (32, 32, 8)
input_shape_s2 = (32, 32, 10)

#kernel_size = 5
stride_size = 1
pool_size = 2

input_layer_s1 = Input(shape=input_shape_s1)
input_layer_s2 = Input(shape=input_shape_s2)


#Unet for S1
def unetModel(input_dt):
	c1 = Conv2D(32, 3, activation='relu', padding='same')(input_dt)
	c1 = BatchNormalization()(c1)
	c1 = Conv2D(32, 3, activation='relu', padding='same')(c1)
	p1 = MaxPooling2D(2)(c1)  # 16×16

	b = Conv2D(32, 3, activation='relu', padding='same')(p1)
	b = Dropout(0.3)(b)
	b = Conv2D(32, 3, activation='relu', padding='same')(b)

	u1 = Conv2DTranspose(32, 2, strides=2, padding='same')(b)  # 16→32
	u1 = Concatenate()([u1, c1])
	c2 = Conv2D(32, 3, activation='relu', padding='same')(u1)
	c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)

	return c2

def cnnModel(input_dt):

	x = Conv2D(32, 3, activation='relu', padding='same', name="conv_s1")(input_dt)
	x = BatchNormalization(name="bn_s1")(x)
	x = SpatialDropout2D(0.2)(x)
	return (x)
feat_s1 = unetModel(input_layer_s1)      # (32, 32, 32)
feat_s2 = cnnModel(input_layer_s2)       # (8, 8, 64)

x1 = GlobalAveragePooling2D()(feat_s1)
x1 = Dense(64, activation='relu')(x1)
softmax_s1 = Dense(n_classes, activation='softmax', name='softmax_s1')(x1)

x2 = GlobalAveragePooling2D()(feat_s2)
x2 = Dense(64, activation='relu')(x2)
softmax_s2 = Dense(n_classes, activation='softmax', name='softmax_s2')(x2)

alpha = .2
weighted_s1 = Lambda(lambda x: x * alpha)(softmax_s1)
weighted_s2 = Lambda(lambda x: x * (1 - alpha))(softmax_s2)
fused_output = Add(name='late_fusion')([weighted_s1, weighted_s2])  # shape: (None, n_classes)

model = Model(inputs=[input_layer_s1, input_layer_s2], outputs=fused_output)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit([X1,X2], Y, epochs=100)
model.save("FM4final-.2-100.h5")

model = load_model("FM4final-.2-100.h5")


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

   


