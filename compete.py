import sys
import numpy as np
import pandas as pd
import keras
import tensorflow
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout, Input, concatenate
from keras.optimizers import Adam, SGD, Adagrad
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from scipy.ndimage.filters import gaussian_filter

def unsharp_mask1(img):
    image = img/255.0
    better = blurred = gaussian_filter(image,
                              sigma=1.0,
                              mode='reflect')
    better = image + (image - better) * 2.0
    negative = np.any(better < 0)
    if negative:
        vrange = [0., 1.]
    else:
        vrange = [0., 1.]
    result = np.clip(better, vrange[0], vrange[1])
    return result

# %% [code]
df = pd.read_csv(sys.argv[1], header=None).values
df2 = pd.read_csv(sys.argv[2], header=None).values
num_arr = np.zeros((df.shape[0], 32, 32, 3))
test_arr = np.zeros((df2.shape[0], 32, 32, 3))
Y_arr = np.zeros((df.shape[0], 1))
for i in range(0, df.shape[0]):
    a = df[i, 0].split(' ')
    a = np.array(a).astype(np.int32).reshape(1, 3074)
    num_arr[i, :, :, :] = a[0, :-2].reshape((3, 32, 32)).transpose([1, 2, 0]).astype(np.int32)
    Y_arr[i, 0] = int(a[0, -1].astype(np.uint8))
    # print(i)
    if i % 1000 == 0:
        print(i)

for i in range(0, df2.shape[0]):
    a = df2[i, 0].split(' ')
    a = np.array(a).astype(np.int32).reshape(1, 3074)
    test_arr[i, :, :, :] = a[0, :-2].reshape((3, 32, 32)).transpose([1, 2, 0]).astype(np.int32)
    if i % 1000 == 0:
        print(i)
Y = np.zeros((Y_arr.shape[0], 100))
for i in range(0, Y_arr.shape[0]):
    Y[i][int(Y_arr[i])] = 1

# %% [code]
better = np.zeros(num_arr.shape)
#blur =  np.zeros(num_arr.shape)
for i in range(num_arr.shape[0]):
    better[i,:,:,:] = unsharp_mask1(num_arr[i,:,:,:].astype(np.uint8))
 #   blur[i,:,:,:] = gaussian_filter(num_arr[i,:,:,:]/255.0,sigma=0.2, mode='reflect')
    if i%1000==0:
        print(i)
X_train = num_arr[0:50000,:,:,:]
X_train = ((X_train.astype(np.float32))-127.5)/127.5
X_train = np.concatenate((X_train,(better-0.5)/0.5),axis=0)
#X_train = np.concatenate((X_train,(blur-0.5)/0.5),axis=0)
Y_train = np.concatenate((Y[0:50000,:],Y[0:50000,:]),axis=0)
#Y_train = np.concatenate((Y_train,Y[0:50000,:]),axis=0)
test_arr = ((test_arr.astype(np.float32))-127.5)/127.5
validn = X_train[95000:,:,:,:]
Y_validn = Y_train[95000:,:]
#X_train = np.concatenate((X_train[:45000,:,:,:],X_train[50000:95000,:,:,:]),axis=0)
#Y_train = np.concatenate((Y_train[:45000,:],Y_train[50000:95000,:]),axis=0)

# %% [code]

class MixupImageDataGenerator():
    def __init__(self, generator, X,Y, batch_size, img_height, img_width, alpha=0.2, subset=None):
        """Constructor for mixup image data generator.
        Arguments:
            generator {object} -- An instance of Keras ImageDataGenerator.
            directory {str} -- Image directory.
            batch_size {int} -- Batch size.
            img_height {int} -- Image height in pixels.
            img_width {int} -- Image width in pixels.
        Keyword Arguments:
            alpha {float} -- Mixup beta distribution alpha parameter. (default: {0.2})
            subset {str} -- 'training' or 'validation' if validation_split is specified in
            `generator` (ImageDataGenerator).(default: {None})
        """

        self.batch_index = 0
        self.batch_size = batch_size
        self.alpha = alpha

        # First iterator yielding tuples of (x, y)
        self.generator1 = generator.flow(X,Y,batch_size)

        # Second iterator yielding tuples of (x, y)
        self.generator2 = generator.flow(X,Y,batch_size)

        # Number of images across all classes in image directory.
        self.n = X.shape[0]

    def reset_index(self):
        """Reset the generator indexes array.
        """

        self.generator1._set_index_array()
        self.generator2._set_index_array()

    def on_epoch_end(self):
        self.reset_index()

    def reset(self):
        self.batch_index = 0

    def __len__(self):
        # round up
        return (self.n + self.batch_size - 1) // self.batch_size

    def get_steps_per_epoch(self):
        """Get number of steps per epoch based on batch size and
        number of images.
        Returns:
            int -- steps per epoch.
        """

        return self.n // self.batch_size

    def __next__(self):
        """Get next batch input/output pair.
        Returns:
            tuple -- batch of input/output pair, (inputs, outputs).
        """

        if self.batch_index == 0:
            self.reset_index()

        current_index = (self.batch_index * self.batch_size) % self.n
        if self.n > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        # random sample the lambda value from beta distribution.
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)

        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        # Get a pair of inputs and outputs from two iterators.
        X1, y1 = self.generator1.next()
        X2, y2 = self.generator2.next()

        # Perform the mixup.
        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)
        return X, y

    def __iter__(self):
        while True:
            yield next(self)

import random
import numpy as np


class CutMixImageDataGenerator():
    def __init__(self, generator, img_size, batch_size,X,Y):
        self.batch_index = 0
        self.samples = X.shape[0]
#         self.class_indices = generator1.class_indices
        self.generator1 = generator.flow(X,Y,batch_size)
        self.generator2 = generator.flow(X,Y,batch_size)
        self.img_size = img_size
        self.batch_size = batch_size

    def reset_index(self):  # Ordering Reset (If Shuffle is True, Shuffle Again)
        self.generator1._set_index_array()
        self.generator2._set_index_array()

    def reset(self):
        self.batch_index = 0
        self.generator1.reset()
        self.generator2.reset()
        self.reset_index()

    def get_steps_per_epoch(self):
        quotient, remainder = divmod(self.samples, self.batch_size)
        return (quotient + 1) if remainder else quotient
    
    def __len__(self):
        self.get_steps_per_epoch()

    def __next__(self):
        if self.batch_index == 0: self.reset()

        crt_idx = self.batch_index * self.batch_size
        if self.samples > crt_idx + self.batch_size:
            self.batch_index += 1
        else:  # If current index over number of samples
            self.batch_index = 0

        reshape_size = self.batch_size
        last_step_start_idx = (self.get_steps_per_epoch()-1) * self.batch_size
        if crt_idx == last_step_start_idx:
            reshape_size = self.samples - last_step_start_idx
            
        X_1, y_1 = self.generator1.next()
        X_2, y_2 = self.generator2.next()
        
        cut_ratio = np.random.beta(a=1, b=1, size=reshape_size)
        cut_ratio = np.clip(cut_ratio, 0.2, 0.8)
        label_ratio = cut_ratio.reshape(reshape_size, 1)
        cut_img = X_2

        X = X_1
        for i in range(reshape_size):
            cut_size = int((self.img_size-1) * cut_ratio[i])
            y1 = random.randint(0, (self.img_size-1) - cut_size)
            x1 = random.randint(0, (self.img_size-1) - cut_size)
            y2 = y1 + cut_size
            x2 = x1 + cut_size
            cut_arr = cut_img[i][y1:y2, x1:x2]
            cutmix_img = X_1[i]
            cutmix_img[y1:y2, x1:x2] = cut_arr
            X[i] = cutmix_img
            
        y = y_1 * (1 - (label_ratio ** 2)) + y_2 * (label_ratio ** 2)
        return X, y

    def __iter__(self):
        while True:
            yield next(self)


def LDCNet():
    b = Input(shape=(32,32,3))
    a = Conv2D(64, kernel_size=3, strides=(1, 1), padding = 'same', activation='elu')(b)
    a = BatchNormalization()(a)
    a = Conv2D(128, kernel_size=3, strides=(1, 1), padding = 'same', activation='elu')(a)
    a = BatchNormalization()(a)
    a = MaxPooling2D(pool_size=(2,2), padding='same')(a)
    a = Dropout(0.1)(a)
    a = Conv2D(128, kernel_size=3, strides=(1, 1), padding = 'same', activation='elu')(a)
    a = BatchNormalization()(a)
    a = Conv2D(256, kernel_size=3, strides=(1, 1), padding = 'same', activation='elu')(a)
    a = BatchNormalization()(a)
    a = MaxPooling2D(pool_size=(2,2),strides=(1, 1), padding='same')(a)
    a = Dropout(0.1)(a)
    a = Conv2D(256, kernel_size=3, strides=(1, 1), padding = 'same', activation='elu')(a)
    a = BatchNormalization()(a)
    a = Conv2D(512, kernel_size=3, strides=(1, 1), padding = 'same', activation='elu')(a)
    a = BatchNormalization()(a)
    a = MaxPooling2D(pool_size=(2,2),strides=(1, 1), padding='same')(a)
    a = Dropout(0.2)(a)
    ####################
    a1 = Conv2D(64, kernel_size=1, strides=(1, 1), padding = 'same', activation='elu')(a)
    a2 = Conv2D(128, kernel_size=3, strides=(1, 1), padding = 'same', activation='elu')(a)
    a3 = Conv2D(64, kernel_size=5, strides=(1, 1), padding = 'same', activation='elu')(a)
#     a4 = MaxPooling2D(pool_size=(2,2),strides=(1, 1), padding='same')(a)
    a = concatenate([a1,a2,a3],axis=3)
    a = BatchNormalization()(a)
    ####################
    a = MaxPooling2D(pool_size=(2,2),strides=(2, 2), padding='same')(a)
    a = Dropout(0.2)(a)
    ###################
    a1 = Conv2D(64, kernel_size=1, strides=(1, 1), padding = 'same', activation='elu')(a)
    a2 = Conv2D(128, kernel_size=3, strides=(1, 1), padding = 'same', activation='elu')(a)
    a3 = Conv2D(64, kernel_size=5, strides=(1, 1), padding = 'same', activation='elu')(a)
#     a4 = MaxPooling2D(pool_size=(2,2),strides=(1, 1), padding='same')(a)
    a = concatenate([a1,a2,a3],axis=3)
    a = BatchNormalization()(a)
    ###############3###
    a = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding='same')(a)
    a = Dropout(0.2)(a)
    ###################
    a1 = Conv2D(128, kernel_size=1, strides=(1, 1), padding = 'same', activation='elu')(a)
    a2 = Conv2D(256, kernel_size=3, strides=(1, 1), padding = 'same', activation='elu')(a)
    a3 = Conv2D(64, kernel_size=5, strides=(1, 1), padding = 'same', activation='elu')(a)
#     a4 = MaxPooling2D(pool_size=(2,2), strides=(1, 1),padding='same')(a)
    a = concatenate([a1,a2,a3],axis=3)
    a = BatchNormalization()(a)
    ##################

     
    ######################

    a = MaxPooling2D(pool_size=(2,2),strides=(1, 1), padding='same')(a)
    a = Dropout(0.3)(a)
    a = Flatten()(a)
    a = Dense(512,activation='elu')(a)
    a = BatchNormalization()(a)
    a = Dropout(0.3)(a)
    a = Dense(512,activation='elu')(a)
    a = BatchNormalization()(a)
    a = Dropout(0.3)(a)
    a = Dense(100,activation='softmax')(a)
    model = Model(inputs = b,outputs=a,name = 'LDCNet' )
    return model
model = LDCNet()
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
#datagen = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,shear_range=0.1,zoom_range=0.1,channel_shift_range=0.1)
datagen1 = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,shear_range=0.1,zoom_range=0.1)
train_generator = CutMixImageDataGenerator(generator=datagen1,img_size=32,batch_size=500,X=X_train,Y=Y_train)
#datagen.fit(X_train)
model.fit_generator(train_generator,steps_per_epoch=len(X_train)/500, epochs=17,validation_data = (validn,Y_validn))
model.fit(X_train,Y_train,batch_size=500,epochs=3,validation_data = (validn,Y_validn))
#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=500),steps_per_epoch=len(X_train)/500, epochs=7)
model.fit_generator(train_generator,steps_per_epoch=len(X_train)/500, epochs=7,validation_data = (validn,Y_validn))
model.fit(X_train,Y_train,batch_size=500,epochs=3,validation_data = (validn,Y_validn))
#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=500),steps_per_epoch=len(X_train)/500, epochs=7)
model.fit_generator(train_generator,steps_per_epoch=len(X_train)/500, epochs=7,validation_data = (validn,Y_validn))
model.fit(X_train,Y_train,batch_size=500,epochs=3,validation_data = (validn,Y_validn))
datagen1 = ImageDataGenerator(rotation_range=10,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,shear_range=0.2,zoom_range=0.2,channel_shift_range=0.2)
train_generator = CutMixImageDataGenerator(generator=datagen1,img_size=32,batch_size=500,X=X_train,Y=Y_train)
#datagen.fit(X_train)
#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=500),steps_per_epoch=len(X_train)/500, epochs=10,validation_data = (validn,Y_validn))
model.fit_generator(train_generator,steps_per_epoch=len(X_train)/500, epochs=17,validation_data = (validn,Y_validn))
model.fit(X_train,Y_train,batch_size=500,epochs=3,validation_data = (validn,Y_validn))
#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=500),steps_per_epoch=len(X_train)/500, epochs=7)
model.fit_generator(train_generator,steps_per_epoch=len(X_train)/500, epochs=7,validation_data = (validn,Y_validn))
model.fit(X_train,Y_train,batch_size=500,epochs=3,validation_data = (validn,Y_validn))
#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=500),steps_per_epoch=len(X_train)/500, epochs=5)
model.fit_generator(train_generator,steps_per_epoch=len(X_train)/500, epochs=5,validation_data = (validn,Y_validn))
model.fit(X_train,Y_train,batch_size=500,epochs=3,validation_data = (validn,Y_validn))
#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=500),steps_per_epoch=len(X_train)/500, epochs=5)
model.fit_generator(train_generator,steps_per_epoch=len(X_train)/500, epochs=5,validation_data = (validn,Y_validn))
model.fit(X_train,Y_train,batch_size=500,epochs=3,validation_data = (validn,Y_validn))
#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=500),steps_per_epoch=len(X_train)/500, epochs=5)
model.fit_generator(train_generator,steps_per_epoch=len(X_train)/500, epochs=5,validation_data = (validn,Y_validn))
model.fit(X_train,Y_train,batch_size=500,epochs=3,validation_data = (validn,Y_validn))
#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=500),steps_per_epoch=len(X_train)/500, epochs=5)
#model.fit(X_train,Y_train,batch_size=500,epochs=3)
a = np.argmax(model.predict(test_arr), axis=1)
for row in a:
    print(np.asscalar(row), file=open(sys.argv[3], "a"))