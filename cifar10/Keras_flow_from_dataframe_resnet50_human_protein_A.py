

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


traindf=pd.read_csv("C:/Users/macie/wujeccoai/git/wujeccoKeras/cifar10/train.csv")



#%%

datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.25)

#%%

train_generator=datagen.flow_from_dataframe(

    dataframe=traindf,
    directory="C:/Users/macie/wujeccoai/git/wujeccoKeras/cifar10/sampleImage/train",

    x_col="Id",
    y_col="Target",
    
    has_ext=False,
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(224,224),
    color_mode= "grayscale"
                                            )
#%%

valid_generator=datagen.flow_from_dataframe(
    
    dataframe=traindf,
    directory="C:/Users/macie/wujeccoai/git/wujeccoKeras/cifar10/sampleImage/train",
    
    x_col="Id",
    y_col="Target",

    has_ext=False,
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(224,224),
    color_mode= "grayscale"
                                            )
#%%


                                            
#%%
model = ResNet50(weights=None,classes=28)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


#%%
STEP_SIZE_TRAIN=1
STEP_SIZE_VALID=1
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)

#%%
model.evaluate_generator(generator=valid_generator
)


print('dupa')



