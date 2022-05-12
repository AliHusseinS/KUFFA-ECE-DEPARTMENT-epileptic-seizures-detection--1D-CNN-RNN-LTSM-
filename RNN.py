import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, SimpleRNN,LSTM,Activation,Dropout
import os
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import layers
from keras import regularizers

df = pd.read_csv('data.csv')
df['y'] = (df['y'] ==1).astype('int')
df['y']
# The spread of labels in the dataframe
df['y'].value_counts()
# Defining a list with class names corresponding to list indices
class_names = ['Non-epileptic', 'Epileptic']
# separating into 2 dataframes, one for each class

df1 = df[df['y'] == 1]
df0 = df[df['y'] == 0]
print("Number of samples in:")
print("Class label 0 - ", len(df0))
print("Class label 1 - ", len(df1))

# Upsampling

df1 = df1.sample(len(df0), replace = True)    # replace = True enables resampling

print('\nAfter resampling - ')

print("Number of samples in:")
print("Class label 0 - ", len(df0))
print("Class label 1 - ", len(df1))
# concatente to form a single dataframe

df = df0.append(df1)

print('Total number of samples - ', len(df))
# defining the input and output columns to separate the dataset in the later cells.

input_columns = list(df.columns[1:-1])    # exculding the first 'Unnamed' column
output_columns = list(df.columns[-1])

print("Number of input columns: ", len(input_columns))
#print("Input columns: ", ', '.join(input_columns))

print("Number of output columns: ", len(output_columns))
#print("Output columns: ", ', '.join(output_columns))
# Splitting into train, val and test set -- 80-10-10 split

# First, an 80-20 split
train_df, val_test_df = train_test_split(df, test_size = 0.2)

# Then split the 20% into half
val_df, test_df = train_test_split(val_test_df, test_size = 0.5)

print("Number of samples in...")
print("Training set: ", len(train_df))
print("Validation set: ", len(val_df))
print("Testing set: ", len(test_df))
# Splitting into X (input) and y (output)

Xtrain, ytrain = np.array(train_df[input_columns]), np.array(train_df[output_columns])

Xval, yval = np.array(val_df[input_columns]), np.array(val_df[output_columns])

Xtest, ytest = np.array(test_df[input_columns]), np.array(test_df[output_columns])
df.describe()
# Using standard scaler to standardize them to values with mean = 0 and variance = 1.

standard_scaler = StandardScaler()

# Fit on training set alone
Xtrain = standard_scaler.fit_transform(Xtrain)

# Use it to transform val and test input
Xval = standard_scaler.transform(Xval)
Xtest = standard_scaler.transform(Xtest)
Xtrain = np.expand_dims(Xtrain, -1)



epochs = 32
def build_RNN():
    model = Sequential()
    model.add(LSTM(128,input_shape=Xtrain[0].shape,return_sequences=True))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(1))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model
model = build_RNN()

model.summary()


plot_model(model,to_file='model_plot.png',show_shapes=True,show_layer_names=True)

history1 = model.fit(Xtrain, ytrain, validation_data = (Xval, yval),batch_size=128, epochs=epochs)

model.evaluate(Xtest, ytest)


#
# cm = confusion_matrix(ytest, (model.predict(Xtest)>0.5).astype('int'))
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
# for i in range(cm.shape[1]):
#     for j in range(cm.shape[0]):
#         plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="black")
#
#
# plt.imshow(cm, cmap=plt.cm.Blues)
# #

# pick random test data sample from one batch
x = random.randint(0, len(Xtest) - 1)

output_true = np.array(ytest)[x][0]
print("True: ", class_names[output_true])

output = model.predict(Xtest[x].reshape(1, -1))[0][0]
pred = int(output>0.5)    # finding max
print("Predicted: ", class_names[pred], "(",output, "-->", pred, ")")    # Picking the label from class_names base don the model output
###
plt.figure(1)

# summarize history for accuracy

plt.subplot(211)
plt.plot(history1.history['acc'])
plt.plot(history1.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

# summarize history for loss

plt.subplot(212)
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.tight_layout()
plt.show()

pred = model.predict(Xtest)
cnn_pred = (pred > 0.5)
acc_cnn = accuracy_score(cnn_pred,ytest)
model.save('epileptic_seizure_cnn.h5')
print("The RNN ACC =__ ",acc_cnn)
