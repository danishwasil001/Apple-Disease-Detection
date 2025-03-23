import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Define your dataset path, image size, and batch size
dataset_path = 'C:/Users/mohdd/Desktop/APPLE DISEASES'  # Update with your dataset path
image_size = (224, 224)
batch_size = 32  # Adjust batch size as needed

# Create an ImageDataGenerator for preprocessing
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values
    validation_split=0.2  # Use validation split if needed
)

# Load and preprocess the data using the generator
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Adjust class_mode as needed
    subset='training'  # Use 'validation' for validation data if applicable
)

# Load the pre-trained DenseNet121 model
base_model = DenseNet121(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# Add a logistic layer with the number of classes
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Construct the full model
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
    validation_steps=train_generator.samples // batch_size * 0.2  # Adjust validation steps if needed
)

# Save the trained model
model.save('apple_latest.h5')
