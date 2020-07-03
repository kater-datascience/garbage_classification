import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

size = 299 # Xception input size
# # size = 224 # VGG16 input size
img_dim = (size, size)
batch_size = 32
epochs = 256

# create callbacks
checkpoint = ModelCheckpoint("best_model_garbage_xception_test.hdf5", monitor='val_acc', verbose=1,
                             save_best_only=True, mode='auto', period=1)
earlyStop = EarlyStopping(monitor='val_acc', min_delta=0, patience=16, verbose=0, mode='auto', baseline=None)
log_dir = "logs\\scalars\\" + datetime.now().strftime("%Y%m%d-%H%M%S") + "vgg16"
tensorboard_callback = TensorBoard(log_dir=log_dir)

# model adaptation
base_net = tf.keras.applications.xception.Xception(weights='imagenet')
# base_net=tf.keras.applications.vgg16.VGG16(weights='imagenet')
base_net_input = base_net.get_layer(index=0).input
base_net_output = base_net.get_layer(index=-2).output
base_net_model = models.Model(inputs=base_net_input, outputs=base_net_output)

for layer in base_net_model.layers:
    layer.trainable = False

new_xception_model = models.Sequential()
new_xception_model.add(base_net_model)
new_xception_model.add(layers.Dense(5, activation='softmax', input_dim=2048))

# hyperparameters
base_learning_rate = 0.0001
opt = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

new_xception_model.compile(optimizer=opt,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

# data preparation and augmentation
train_data_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

valid_data_gen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_data_gen.flow_from_directory(
    directory='training_data',
    target_size=img_dim,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=True,
    seed=42
)

valid_generator = valid_data_gen.flow_from_directory(
    directory='validation_data',
    target_size=img_dim,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=True,
    seed=42
)

# model training
garbage_recognition_model = new_xception_model.fit_generator(generator=train_generator,
                                                             validation_data=valid_generator,
                                                             epochs=epochs,
                                                             callbacks=[checkpoint, tensorboard_callback, earlyStop]
                                                             )
