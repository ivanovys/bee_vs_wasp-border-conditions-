# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 02:11:07 2021

@author: ivanov
"""


import tensorflow.keras
import tensorflow as tf
import cv2

from tensorflow.keras.utils import plot_model


def dataset():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        zoom_range=0.2)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    )


#!!!!Flow from pd!!! TO DO
    train_generator = train_datagen.flow_from_directory(
        'C:\\data\\BW\\train',
        target_size=(224, 224),
        shuffle=True,
        batch_size=32,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        'C:\\data\\BW\\val',
        target_size=(224, 224),
        shuffle=True,
        batch_size=32,
        class_mode='categorical')

    return train_generator, validation_generator


def build_model():
    base_model = tf.keras.applications.MobileNetV2(include_top=False)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    predication = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predication)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train():
    model = build_model()
    model.summary()
    plot_model(model, to_file=('v2' +'.png'), show_shapes=True, show_layer_names=True,rankdir='TB', expand_nested=False, dpi=96)
    file_name = 'weights-{epoch:02d}.hdf5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(file_name, monitor='loss', verbose=1, period=1)
    callbacks_list = [checkpoint]
    train, val = dataset()
    model.fit_generator(train,  
                        epochs=10,
                        verbose=1,
                        validation_data=val,
                        callbacks=callbacks_list)

    model.save_weights('model2.h5')



#model.fit_generator(train_ds,epochs=10,verbose=1,validation_data=val_ds, callbacks=callbacks_list)

#model.fit_generator(genetator(train_ds),epochs=10,verbose=1,validation_data=genetator(val_ds), callbacks=callbacks_list)

class ClassifityInsects(object):

    def __init__(self):
        self.model = self.build_model()
        self.model.load_weights("model2.h5")
        self.fix_map_classes = ["0 bee", "1 wasp"]

    def transform_images(self, x, size):
        x = tf.image.resize(x, (size, size))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        return x

    def preprocessing(self, image):
        processing = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processing = tf.expand_dims(processing, 0)
        processing = self.transform_images(processing, 224)
        return processing

    def build_model(self):
        base_model = tf.keras.applications.MobileNetV2(include_top=False)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        predication = tf.keras.layers.Dense(2, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predication)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def run(self, image):
        image = self.preprocessing(image)
        output = self.model.predict(image)
        return output

    def get_top_k(self, values):
        return values[0].argsort()[-2:][::-1]

    def draw_on_images(self, image):
        values = self.run(image)
        top_k_classes = self.get_top_k(values)
        template = "MobileNet 2.0 Output \n" \
                   "1. {0} - {1}\n" \
                   "2. {2} - {3}\n".format(self.fix_map_classes[top_k_classes[0]], values[0][top_k_classes[0]],
                                         self.fix_map_classes[top_k_classes[1]], values[0][top_k_classes[1]])
        template = template.split('\n')
        return template
    
classifity = ClassifityInsects()
#classifity2 = ClassifityInsects()
filename = "C:\\data\\BW\\val\\wasp\\2747451923_3a0e3ff868_w.jpg"  # картинка

frame = cv2.imread(filename)
frame = cv2.resize(frame,(640,480))
text = classifity.draw_on_images(frame)
shift = 20
for item in text:
    print(item)
    cv2.putText(frame,item,(0,shift),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    shift += 20
cv2.imshow('output', frame)
cv2.waitKey()