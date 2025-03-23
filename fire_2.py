import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121

INIT_LR = 0.1
BATCH_SIZE = 32  # Adjust the batch size
NUM_EPOCHS = 20  # Adjust the number of epochs
lr_find = True

classes = ['BLOTCH', 'NORMAL', 'ROT', 'SCAB']

images = []
labels = []
for c in classes:
    try:
        for img in os.listdir('C:/Users/mohdd/Desktop/APPLE DISEASES/' + c):
            img_path = 'C:/Users/mohdd/Desktop/APPLE DISEASES/' + c + '/' + img
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(classes.index(c))  # Assign label as index of class
    except Exception as e:
        print(f"Error processing class {c}: {e}")

if len(images) == 0:
    print("No images were loaded. Check your image directory path and contents.")
else:
    print(f"Loaded {len(images)} images.")

images = np.array(images, dtype='float32') / 255.0
if len(images) > 0:
    ind = np.random.randint(0, len(images))
    selected_image = images[ind]
    cv2.imshow(str(labels[ind]),images[ind])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

labels = np.array(labels)
labels = np_utils.to_categorical(labels,num_classes=2)


d = {}

classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

d[0] = classWeight[0]
d[1] = classWeight[1]

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, shuffle=True, random_state=42)

aug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Load the pre-trained DenseNet121 model
base_model = DenseNet121(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# Add a logistic layer with the number of classes
predictions = Dense(len(classes), activation='softmax')(x)

# Construct the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model (DenseNet121) for initial training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
opt = SGD(learning_rate=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

print("[INFO] training network...")

H = model.fit(
    aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    class_weight=d,
    verbose=1)

print("[INFO] serializing network to '{}'...".format('output/model'))
model.save('output/apple_latest.h5')


N = np.arange(0, NUM_EPOCHS)

plt.figure(figsize=(12,8))

plt.subplot(121)
plt.title("Losses")
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")

plt.subplot(122)
plt.title("Accuracies")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")


plt.legend()
plt.savefig("output/training_plot.png")

# load the trained model from disk
print("[INFO] loading model...")
model = load_model('output/apple_latest.h5')


for i in range(50):
    random_index = np.random.randint(0,len(X_test))
    org_img = X_test[random_index]*255
    img = org_img.copy()
    img = cv2.resize(img,(128,128))

    img = img.astype('float32')/256
    pred = model.predict(np.expand_dims(img,axis=0))[0]
    result = classes[np.argmax(pred)]
    org_img = cv2.resize(org_img,(500,500))
    cv2.putText(org_img, result, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 3)
    cv2.imwrite('output/testing/{}.png'.format(i),org_img)
    print(result)