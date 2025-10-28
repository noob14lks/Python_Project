import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

def find_images_by_class(folder_path, only_augmented=True):
    """Find all images - ONLY from augmented folders"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    class_images = {}
    
    for root, dirs, files in os.walk(folder_path):
        # SKIP original folders, ONLY process augmented
        if only_augmented:
            if 'original' in root.lower():
                continue  # Skip original folders
            if 'augmented' not in root.lower() and any('augmented' in d.lower() for d in os.listdir(os.path.dirname(root)) if os.path.isdir(os.path.join(os.path.dirname(root), d))):
                continue  # Skip if augmented folder exists but we're not in it
        
        for file in files:
            if file.lower().endswith(image_extensions):
                img_path = os.path.join(root, file)
                
                parent = os.path.basename(os.path.dirname(img_path))
                if parent.lower() in ['augmented', 'images']:
                    parent = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
                
                if parent not in class_images:
                    class_images[parent] = []
                class_images[parent].append(img_path)
    
    return class_images

def has_train_test_split(dataset_path):
    """Check if dataset has train/test folders"""
    subdirs = [d.lower() for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    has_train = any('train' in d for d in subdirs)
    has_test = any('test' in d for d in subdirs)
    return has_train and has_test

def extract_features_batch(image_paths, model, batch_size=32):
    """Extract features from images in batches"""
    features_list = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                batch_images.append(img_array)
            except:
                continue
        
        if len(batch_images) == 0:
            continue
        
        batch_array = np.array(batch_images, dtype=np.float32)
        batch_preprocessed = preprocess_input(batch_array)
        batch_features = model.predict(batch_preprocessed, verbose=0)
        features_list.append(batch_features)
        
        print(f"    Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)}", end='\r')
    
    if len(features_list) == 0:
        return None
    
    return np.vstack(features_list)

def process_dataset_direct(dataset_path, output_folder, dataset_name, only_augmented=True):
    """Process dataset with train/test detection - ONLY AUGMENTED DATA"""
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}")
    print(f"Mode: AUGMENTED DATA ONLY")
    print(f"{'='*70}")
    
    # Load DenseNet121
    print("Loading DenseNet121 model...")
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    model.trainable = False
    print("✓ Model loaded (1024 features)")
    
    # Check for train/test split
    if has_train_test_split(dataset_path):
        print("\n✓ Found existing train/test split")
        
        train_folder = None
        test_folder = None
        
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                if 'train' in item.lower():
                    train_folder = item_path
                if 'test' in item.lower():
                    test_folder = item_path
        
        if not train_folder or not test_folder:
            print("✗ Could not locate train/test folders")
            return None
        
        # Load train images (augmented only)
        print(f"\nLoading from train (AUGMENTED ONLY): {train_folder}")
        train_class_images = find_images_by_class(train_folder, only_augmented=True)
        
        if len(train_class_images) == 0:
            print("✗ No augmented training images found")
            return None
        
        train_paths = []
        train_labels = []
        class_names = sorted(train_class_images.keys())
        
        for class_idx, class_name in enumerate(class_names):
            paths = train_class_images[class_name]
            train_paths.extend(paths)
            train_labels.extend([class_idx] * len(paths))
            print(f"  {class_name}: {len(paths)} augmented train images")
        
        train_labels = np.array(train_labels)
        
        # Load test images (augmented only)
        print(f"\nLoading from test (AUGMENTED ONLY): {test_folder}")
        test_class_images = find_images_by_class(test_folder, only_augmented=True)
        
        test_paths = []
        test_labels = []
        
        for class_idx, class_name in enumerate(class_names):
            if class_name in test_class_images:
                paths = test_class_images[class_name]
                test_paths.extend(paths)
                test_labels.extend([class_idx] * len(paths))
                print(f"  {class_name}: {len(paths)} augmented test images")
        
        test_labels = np.array(test_labels)
        
        # Create validation split
        print(f"\nCreating validation split from training data...")
        val_size = min(0.14, max(0.1, 2.0 / len(train_paths)))
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=val_size, random_state=42,
            stratify=train_labels if len(np.unique(train_labels)) > 1 else None
        )
        
        print(f"  Train: {len(train_paths)}")
        print(f"  Val: {len(val_paths)}")
        print(f"  Test: {len(test_paths)}")
        
    else:
        print("\n✗ No train/test split - creating 70:10:20 split from AUGMENTED data")
        
        # Load augmented images only
        class_images = find_images_by_class(dataset_path, only_augmented=True)
        
        if len(class_images) == 0:
            print("✗ No augmented images found")
            return None
        
        print(f"Found {len(class_images)} classes (AUGMENTED ONLY):")
        for cls, imgs in class_images.items():
            print(f"  {cls}: {len(imgs)} augmented images")
        
        all_paths = []
        all_labels = []
        class_names = sorted(class_images.keys())
        
        for class_idx, class_name in enumerate(class_names):
            paths = class_images[class_name]
            all_paths.extend(paths)
            all_labels.extend([class_idx] * len(paths))
        
        all_labels = np.array(all_labels)
        
        # Split 70:30
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            all_paths, all_labels, test_size=0.30, random_state=42,
            stratify=all_labels if len(np.unique(all_labels)) > 1 else None
        )
        
        # Split 30 into 10:20
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.67, random_state=42,
            stratify=temp_labels if len(np.unique(temp_labels)) > 1 else None
        )
        
        print(f"\n  Train: {len(train_paths)}")
        print(f"  Val: {len(val_paths)}")
        print(f"  Test: {len(test_paths)}")

    print("\n[Extracting train features...]")
    train_features = extract_features_batch(train_paths, model)
    
    print("\n[Extracting validation features...]")
    val_features = extract_features_batch(val_paths, model)
    
    print("\n[Extracting test features...]")
    test_features = extract_features_batch(test_paths, model)
    
    if train_features is None:
        print("✗ Failed to extract features")
        return None

    output_path = os.path.join(output_folder, f"{dataset_name}_augmented_densenet121_features.npz")
    np.savez_compressed(output_path,
                       X_train=train_features, y_train=train_labels,
                       X_val=val_features, y_val=val_labels,
                       X_test=test_features, y_test=test_labels,
                       class_names=class_names)
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  Shape: {train_features.shape}")
    
    return output_path

def process_all_datasets(input_folder='./data/extracteddata',
                        output_folder='./features',
                        only_augmented=True):
    """Process all datasets - AUGMENTED DATA ONLY"""
    os.makedirs(output_folder, exist_ok=True)
    
    datasets = [d for d in os.listdir(input_folder) 
                if os.path.isdir(os.path.join(input_folder, d))]
    
    print(f"\n{'='*70}")
    print(f"DenseNet121 Feature Extraction (1024 features)")
    print(f"Datasets: {len(datasets)}")
    print(f"Mode: AUGMENTED DATA ONLY")
    print(f"{'='*70}")
    
    processed = []
    for dataset_name in datasets:
        dataset_path = os.path.join(input_folder, dataset_name)
        result = process_dataset_direct(dataset_path, output_folder, dataset_name, only_augmented=True)
        if result:
            processed.append(result)
    
    print(f"\n{'='*70}")
    print(f"✓ Processed: {len(processed)}/{len(datasets)}")
    print(f"{'='*70}\n")
    
    return processed

if __name__ == "__main__":
    process_all_datasets(only_augmented=True)
