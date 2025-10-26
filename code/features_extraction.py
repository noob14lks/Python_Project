from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
import os

DATA_ROOT = "./data/extracteddata" 
OUTPUT_ROOT = "./features"           
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

os.makedirs(OUTPUT_ROOT, exist_ok=True)

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = GlobalAveragePooling2D()(base_model.output)
feature_model = Model(inputs=base_model.input, outputs=x)
feature_model.trainable = False

for dataset_name in os.listdir(DATA_ROOT):
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    if not os.path.isdir(dataset_path):
        continue 

    print(f"\nProcessing dataset: {dataset_name}")

    data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=False
    )

    steps = int(np.ceil(generator.samples / BATCH_SIZE))
    features = feature_model.predict(generator, steps=steps, verbose=1)
    labels = generator.classes
    class_indices = generator.class_indices

    print("Features Shape: ", features.shape)
    print("Labels Shape: ", labels.shape)

    output_file = os.path.join(OUTPUT_ROOT, f"{dataset_name}_features.npz")
    np.savez_compressed(
        output_file,
        features=features,
        labels=labels,
        class_indices=class_indices
    )
    print(f"Saved features for {dataset_name} to: {output_file}")
