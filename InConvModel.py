""" In-Conv Model is an attempt to combine Involution and Convolutional Neural Network. 

Involutional Kerenl,(as described in https://arxiv.org/abs/2103.06255) is both location specific and channel-agnostic. Convolution, on the other
hand is location-agnostic and channel specific.

In-Conv model attempts to combine these orthagonal properties to experiment with the CIFAR10 dataset.   """

import os
from dataclasses import dataclass
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

@dataclass
class Configuration:
    INPUT_SHAPE = (32,32,3)
    NUM_CLASSES = 10
    EPOCHS = 5

config = Configuration() 

#Constants for TensorBoard
LOGS_DIR = "/tmp/tb/tf_logs/CIFAR_10_INVOLUTION/" # Add path as required
HIST_FREQ = 1
PROFILE_BATCH = (500,520)

# Credits : https://github.com/keras-team/keras-io/blob/master/examples/vision/involution.py

class InvolutionLayer(tf.keras.layers.Layer):
    def __init__(self,Channel,Group_Number,Kernel_Size,Stride,Reduction_Ratio,**kwargs):
        super().__init__(**kwargs)
        self.channel = Channel
        self.group_number = Group_Number
        self.kernel_size = Kernel_Size
        self.stride = Stride
        self.red_ratio = Reduction_Ratio

    def build(self, input_shape):
        (_,height,width,num_channels) = input_shape
        height = height // self.stride
        width = width // self.stride

        self.stride_layer = (tf.keras.layers.AveragePooling2D(
            pool_size=self.stride,strides=self.stride,padding="same"
            )
            if self.stride > 1
            else tf.identity
        )

        self.kernel_gen = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.channel // self.red_ratio,kernel_size=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=self.kernel_size*self.kernel_size*self.group_number,kernel_size=1)
        ])


        self.kerenl_reshape = tf.keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size*self.kernel_size,
                1,
                self.group_number,
            )
        )

        self.input_pathches_reshape = tf.keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size*self.kernel_size,
                num_channels//self.group_number,
                self.group_number,
            )
        )
        
        self.output_reshape = tf.keras.layers.Reshape(
            target_shape=(height,width,num_channels)
        )

    def call(self,x):
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)
        kernel = self.kerenl_reshape(kernel)
        
        input_patches = tf.image.extract_patches(
            images=x,
            sizes=[1,self.kernel_size,self.kernel_size,1],
            strides=[1,self.stride,self.stride,1],
            rates=[1,1,1,1],
            padding='SAME',
        ) 

        input_patches = self.input_pathches_reshape(input_patches)

        output = tf.multiply(kernel,input_patches)

        output = tf.reduce_sum(output,axis=3)

        output = self.output_reshape(output)

        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "channel":self.channel,
            "group_number":self.group_number,
            "kernel_size":self.kernel_size,
            "stride":self.stride,
            "reduction_ratio":self.red_ratio,
        })
        return config
        

# Data Loading and Preprocessing
(xTrain,yTrain),(xTest,yTest) = tf.keras.datasets.cifar10.load_data()
(xTrain,xTest) = (xTrain/255.0 , xTest/255.0)


TrainDataset = tf.data.Dataset.from_tensor_slices((xTrain,yTrain)).shuffle(256).batch(256)
TestDataset = tf.data.Dataset.from_tensor_slices((xTest,yTest)).batch(256)


in_conv_model = tf.keras.Sequential([

    tf.keras.layers.InputLayer(input_shape = config.INPUT_SHAPE),
    tf.keras.layers.Conv2D(32,(3,3),padding="same",name="CONV_1"),
    tf.keras.layers.ReLU(name="ReLU_1"),
    tf.keras.layers.MaxPooling2D(),
    InvolutionLayer(Channel=3,Group_Number=1,Kernel_Size=3,Stride=1,Reduction_Ratio=2,name = "INV_1"),
    tf.keras.layers.ReLU(name="ReLU_2"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,(3,3),padding="same",name="CONV_2"),
    tf.keras.layers.ReLU(name="ReLU_3"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(10),
])        

# Debug: in_conv_model.summary()

tBoardCallback = tf.keras.callbacks.TensorBoard(LOGS_DIR,histogram_freq = HIST_FREQ,profile_batch = PROFILE_BATCH)

in_conv_model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

in_conv_hist = in_conv_model.fit(TrainDataset,epochs= config.EPOCHS,validation_data=TestDataset,callbacks = [tBoardCallback])

icscore = in_conv_model.evaluate(TestDataset)

print("\n In-Convolution--> Loss:{} Accuracy:{} for {} epochs \n".format(icscore[0],icscore[1],config.EPOCHS))

in_conv_model.save("Invol_Convol_CIFAR10.h5")
in_conv_model.save_weights("Invol_Convol_CIFAR10_Weights.h5")

import matplotlib.pyplot as plt

plt.title("In-Convolution Loss")
plt.plot(in_conv_hist.history['loss'],label="Loss")
plt.plot(in_conv_hist.history['val_loss'],label="Value Loss")
plt.legend()
plt.show()


plt.title("In-Convolution Accuracy")
plt.plot(in_conv_hist.history['accuracy'],label="Accuracy")
plt.plot(in_conv_hist.history['val_accuracy'],label="Value Accuracy")
plt.legend()

plt.show()
