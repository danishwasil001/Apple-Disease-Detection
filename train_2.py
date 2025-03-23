import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

dataset_path = '../APPLE DISEASES'  # dataset path
image_size = (224, 224)
batch_size = 32  # batch size

# ImageDataGenerator for preprocessing
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values
    validation_split=0.2 
)

# Load and preprocess the data using the generator
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical', 
    subset='training'  
)

# Load the pre-trained DenseNet121 model
base_model = DenseNet121(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)

# fully-connected layer
x = Dense(1024, activation='relu')(x)

# logistic layer with the number of classes
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model (DenseNet121) for initial training
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the generator
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=train_generator,
    validation_steps=train_generator.samples // batch_size * 0.2 
)

# Save the trained model
model.save('apple_latest.h5')
