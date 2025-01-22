import tensorflow as tf 
from keras.utils import load_img, img_to_array
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np 
import json

# STATIC
TRAIN_DIR = "family_detection/data"

class stop_training(tf.keras.callbacks.Callback):
    def __init__(self): 
        self.temp_loss = float('inf')

    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs.get('loss')
        if self.temp_loss <= current_loss: 
            print("\n Loss Not Decrease Anymore !\n")
            self.model.stop_training = True
            return
        self.temp_loss = current_loss
    
def model_solution():
    # datagen 
    train_datagen = ImageDataGenerator(rescale=1/255)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, 
        target_size=(150, 150), 
        batch_size=32,
        class_mode='categorical'
    )
    print(train_generator.class_indices)

    # load weight of inception v3 from local, without top layers
    local_weights_file = 'inception_v3_weights_tf_kernel_notop.h5'
    pre_trained_model = InceptionV3(input_shape=(150, 150, 3), 
                                    include_top=False, 
                                    weights=None)
    pre_trained_model.load_weights(local_weights_file)

    # make all layers non trainable 
    for layer in pre_trained_model.layers: 
        layer.trainable = False
    
    # select last layer to add custom on top
    last_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_layer.output

    ## perbandingan model
    # model 2 
    x_2 = tf.keras.layers.Conv2D(64, (2, 2), activation='relu')(last_output)
    x_2 = tf.keras.layers.MaxPooling2D(2, 2)(x_2)
    x_2 = tf.keras.layers.Conv2D(128, (2, 2), activation='relu')(x_2)
    x_2 = tf.keras.layers.MaxPooling2D(2, 2)(x_2)

    x_2 = tf.keras.layers.Flatten()(x_2)
    x_2 = tf.keras.layers.Dense(1024, activation='relu')(x_2)
    x_2 = tf.keras.layers.Dense(7, activation='softmax')(x_2)
    model_2 = tf.keras.Model(pre_trained_model.input, x_2)

    # compiler    
    model_2.compile(optimizer='adam', 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
    
    # fitting 
    print("Model 2\n")
    e_stop = stop_training()
    model_2.fit(train_generator, 
                epochs=50,
                callbacks=e_stop, 
                verbose=1)

    class_labels = {v: k for k, v in train_generator.class_indices.items()}

    # Fungsi untuk preprocessing dan prediksi
    def preprocess_and_predict(image_path, model, class_labels):
        test_image = load_img(image_path, target_size=(150, 150))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalisasi
        result = model.predict(test_image)
        predicted_class = np.argmax(result)
        print(f"Predicted Class: {class_labels[predicted_class]}")
        print(f"Raw Output: {result}")
    
    # Prediksi untuk setiap gambar
    print("Ludy")
    preprocess_and_predict('family_detection/data/Ludy Hasby/1.jpg', model_2, class_labels)

    print("Fawas")
    preprocess_and_predict('family_detection/data/Fawwas Mubarrak/1.jpg', model_2, class_labels)

    print("Fathan")
    preprocess_and_predict('family_detection/data/Fathan Tornado/1.png', model_2, class_labels)

    print("Nanda")
    preprocess_and_predict('family_detection/data/Nanda Sobrina/1.png', model_2, class_labels)

    return model_2, class_labels

if __name__ == "__main__":
    model, label_indeks = model_solution()
    model.save("family_detection/model.h5")
    with open("label_indeks.json", "w") as json_file:
        json.dump(label_indeks, json_file)