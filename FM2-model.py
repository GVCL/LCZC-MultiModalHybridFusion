#i!/usr/bin/env python

#print ('test1')
import numpy as np
import matplotlib.cm as cm

import h5py 
import pickle
import matplotlib.pyplot as plt

#print ('test2')
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D, BatchNormalization, Conv2D,UpSampling2D, Reshape, ReLU 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate, Add, Softmax, Multiply, SpatialDropout2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Multiply, Concatenate, Reshape, Add, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model, load_model
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

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
input_shape_s1 = (32, 32, 8)  #SAR 
input_shape_s2 = (32, 32, 10)  #MS 

def self_attention(inputs, num_heads=8, key_dim=32):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attn_output = LayerNormalization()(attn_output)
    return Add()([inputs, attn_output])

def cross_attention(query, key, value, name):
	attn_weights = tf.keras.layers.Attention(name=f"cross_attn_{name}")([query, key])
	attn_output = tf.keras.layers.Add(name=f"cross_attn_out_{name}")([attn_weights, value])
	return attn_output

# Define input layers
input_layer_s1 = Input(shape=input_shape_s1, name="input_s1") 
input_layer_s2 = Input(shape=input_shape_s2, name="input_s2")

concatenated_raw_data = Concatenate(name="concat_raw_data")([input_layer_s1, input_layer_s2])

pix = Conv2D(32, 3, padding='same', activation='relu', name="conv_pix")(concatenated_raw_data)
pix = BatchNormalization(name="bn_pix")(pix)
pix = SpatialDropout2D(0.2)(pix)
pix_flattened = GlobalAveragePooling2D(name="gap_pix")(pix)

x_s1 = Conv2D(32, 3, activation='relu', padding='same', name="conv_s1")(input_layer_s1)
x_s1 = BatchNormalization(name="bn_s1")(x_s1)
x_s1 = MaxPooling2D(pool_size=(2, 2), name="pool_s1")(x_s1)
x_s1 = SpatialDropout2D(0.2)(x_s1)

x_s2 = Conv2D(32, 3, activation='relu', padding='same', name="conv_s2")(input_layer_s2)
x_s2 = BatchNormalization(name="bn_s2")(x_s2)
x_s2 = MaxPooling2D(pool_size=(2, 2), name="pool_s2")(x_s2)
x_s2 = SpatialDropout2D(0.2)(x_s2)

x_s1_reshaped = Reshape((x_s1.shape[1] * x_s1.shape[2], x_s1.shape[3]), name="reshape_s1")(x_s1)
x_s2_reshaped = Reshape((x_s2.shape[1] * x_s2.shape[2], x_s2.shape[3]), name="reshape_s2")(x_s2)

attn_s1 = self_attention(x_s1_reshaped)
attn_s2 = self_attention(x_s2_reshaped)

query_s1 = Dense(32, name="query_s1")(attn_s1)
key_s1 = Dense(32, name="key_s1")(attn_s1)
value_s1 = Dense(32, name="value_s1")(attn_s1)

query_s2 = Dense(32, name="query_s2")(attn_s2)
key_s2 = Dense(32, name="key_s2")(attn_s2)
value_s2 = Dense(32, name="value_s2")(attn_s2)

cross_s1 = cross_attention(query_s1, key_s2, value_s2, "s1")
cross_s2 = cross_attention(query_s2, key_s1, value_s1, "s2")

cross_s1_reshaped = Reshape((x_s1.shape[1], x_s1.shape[2], x_s1.shape[3]), name="reshape_back_cross_s1")(cross_s1)
cross_s2_reshaped = Reshape((x_s2.shape[1], x_s2.shape[2], x_s2.shape[3]), name="reshape_back_cross_s2")(cross_s2)

fusion_features = Multiply(name="multiply_features")([cross_s1_reshaped, cross_s2_reshaped])
fusion = Conv2D(64, 3, padding='same')(fusion_features)
fusion = BatchNormalization()(fusion)
fusion = ReLU()(fusion)
fusion = GlobalAveragePooling2D()(fusion)

combined_features = Concatenate(name="concat_features")([pix_flattened, fusion])
combined_features = Dense(64, activation='relu')(combined_features)
output = Dense(n_classes, activation='softmax', name="output")(combined_features)

model = Model(inputs=[input_layer_s1, input_layer_s2], outputs=output, name="MS_SAR_Fusion_Model")
model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print(Y.shape)
history = model.fit([X1, X2], Y, epochs=100)
model.save("FM2updated-100.h5")
model = load_model("FM2updated-100.h5")

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






