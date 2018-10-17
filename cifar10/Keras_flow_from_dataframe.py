

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
traindf=pd.read_csv("E:/wujeccoai_data/cifar10/trainLabels.csv")
testdf=pd.read_csv("E:/wujeccoai_data/cifar10/sampleSubmission.csv")


#%%

datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.25)

#%%

train_generator=datagen.flow_from_dataframe(

    dataframe=traindf,
    directory="E:/wujeccoai_data/cifar10/train",

    x_col="id",
    y_col="label",
    
    has_ext=False,
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(32,32)
                                            )
#%%

valid_generator=datagen.flow_from_dataframe(
    
    dataframe=traindf,
    directory="E:/wujeccoai_data/cifar10/train",
    
    x_col="id",
    y_col="label",

    has_ext=False,
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(32,32)
                                            )
#%%

test_generator=datagen.flow_from_dataframe(

    dataframe=testdf,
    directory="E:/wujeccoai_data/cifar10/test",

    x_col="id",
    y_col=None,
    has_ext=False,
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(32,32)
                                            )
#%%
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])


#%%
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)

#%%
model.evaluate_generator(generator=valid_generator
)

test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)


model.save("C:/Users/m.wujec/wujeccoai/keras/cifar10/wujeccoaimod.h5")
print('Saved trained model at %s ')



