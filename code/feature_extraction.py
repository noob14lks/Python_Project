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

def find_images_by_class(folder_path, only_augmented=False):
    """Find all images organized by class"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    class_images = {}
    
    for root, dirs, files in os.walk(folder_path):
        # Handle augmented/original filtering
        if only_augmented:
            if 'original' in root.lower():
                continue
            if 'augmented' not in root.lower() and any('augmented' in d.lower() for d in os.listdir(os.path.dirname(root)) if os.path.isdir(os.path.join(os.path.dirname(root), d))):
                continue
        
        for file in files:
            if file.lower().endswith(image_extensions):
                img_path = os.path.join(root, file)
                
                parent = os.path.basename(os.path.dirname(img_path))
                if parent.lower() in ['augmented', 'original', 'images']:
                    parent = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
                
                if parent not in class_images:
                    class_images[parent] = []
                class_images[parent].append(img_path)
    
    return class_images

def check_split_structure(dataset_path):
    """
    Check what split structure exists:
    Returns: 'train_val_test', 'train_test', or 'none'
    """
    subdirs = [d.lower() for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    has_train = any('train' in d for d in subdirs)
    has_val = any('val' in d or 'valid' in d for d in subdirs)
    has_test = any('test' in d for d in subdirs)
    
    if has_train and has_val and has_test:
        return 'train_val_test'
    elif has_train and has_test:
        return 'train_test'
    else:
        return 'none'

def find_split_folders(dataset_path):
    """Find train/validation/test folders"""
    train_folder = None
    val_folder = None
    test_folder = None
    
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            item_lower = item.lower()
            if 'train' in item_lower:
                train_folder = item_path
            elif 'val' in item_lower or 'valid' in item_lower:
                val_folder = item_path
            elif 'test' in item_lower:
                test_folder = item_path
    
    return train_folder, val_folder, test_folder

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

def process_dataset_direct(dataset_path, output_folder, dataset_name, only_augmented=False):
    """Process dataset with smart train/val/test detection"""
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}")
    print(f"Mode: {'AUGMENTED ONLY' if only_augmented else 'ALL IMAGES'}")
    print(f"{'='*70}")
    
    # Load DenseNet121
    print("Loading DenseNet121 model...")
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    model.trainable = False
    print("✓ Model loaded (1024 features)")
    
    # Check split structure
    split_type = check_split_structure(dataset_path)
    
    if split_type == 'train_val_test':
        print("\n✓ Found train/validation/test split (using all three)")
        
        train_folder, val_folder, test_folder = find_split_folders(dataset_path)
        
        if not all([train_folder, val_folder, test_folder]):
            print("✗ Could not locate all three folders")
            return None
        
        # Load train
        print(f"\nLoading train from: {train_folder}")
        train_class_images = find_images_by_class(train_folder, only_augmented)
        
        train_paths = []
        train_labels = []
        class_names = sorted(train_class_images.keys())
        
        for class_idx, class_name in enumerate(class_names):
            paths = train_class_images[class_name]
            train_paths.extend(paths)
            train_labels.extend([class_idx] * len(paths))
            print(f"  {class_name}: {len(paths)} train images")
        
        train_labels = np.array(train_labels)
        
        # Load validation
        print(f"\nLoading validation from: {val_folder}")
        val_class_images = find_images_by_class(val_folder, only_augmented)
        
        val_paths = []
        val_labels = []
        
        for class_idx, class_name in enumerate(class_names):
            if class_name in val_class_images:
                paths = val_class_images[class_name]
                val_paths.extend(paths)
                val_labels.extend([class_idx] * len(paths))
                print(f"  {class_name}: {len(paths)} validation images")
        
        val_labels = np.array(val_labels)
        
        # Load test
        print(f"\nLoading test from: {test_folder}")
        test_class_images = find_images_by_class(test_folder, only_augmented)
        
        test_paths = []
        test_labels = []
        
        for class_idx, class_name in enumerate(class_names):
            if class_name in test_class_images:
                paths = test_class_images[class_name]
                test_paths.extend(paths)
                test_labels.extend([class_idx] * len(paths))
                print(f"  {class_name}: {len(paths)} test images")
        
        test_labels = np.array(test_labels)
        
        print(f"\n✓ Using existing splits:")
        print(f"  Train: {len(train_paths)}")
        print(f"  Validation: {len(val_paths)}")
        print(f"  Test: {len(test_paths)}")
        
    elif split_type == 'train_test':
        print("\n✓ Found train/test split (will create validation from train)")
        
        train_folder, _, test_folder = find_split_folders(dataset_path)
        
        if not train_folder or not test_folder:
            print("✗ Could not locate train/test folders")
            return None
        
        # Load train
        print(f"\nLoading train from: {train_folder}")
        train_class_images = find_images_by_class(train_folder, only_augmented)
        
        train_paths = []
        train_labels = []
        class_names = sorted(train_class_images.keys())
        
        for class_idx, class_name in enumerate(class_names):
            paths = train_class_images[class_name]
            train_paths.extend(paths)
            train_labels.extend([class_idx] * len(paths))
            print(f"  {class_name}: {len(paths)} train images")
        
        train_labels = np.array(train_labels)
        
        # Load test
        print(f"\nLoading test from: {test_folder}")
        test_class_images = find_images_by_class(test_folder, only_augmented)
        
        test_paths = []
        test_labels = []
        
        for class_idx, class_name in enumerate(class_names):
            if class_name in test_class_images:
                paths = test_class_images[class_name]
                test_paths.extend(paths)
                test_labels.extend([class_idx] * len(paths))
                print(f"  {class_name}: {len(paths)} test images")
        
        test_labels = np.array(test_labels)
        
        # Split train into train + validation
        print(f"\n✓ Creating validation split from train (10-14% of train)...")
        val_size = min(0.14, max(0.1, 2.0 / len(train_paths)))
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=val_size, random_state=42,
            stratify=train_labels if len(np.unique(train_labels)) > 1 else None
        )
        
        print(f"  Train: {len(train_paths)}")
        print(f"  Validation: {len(val_paths)}")
        print(f"  Test: {len(test_paths)}")
        
    else:
        print("\n✗ No train/test/validation split - creating 70:10:20 split")
        
        class_images = find_images_by_class(dataset_path, only_augmented)
        
        if len(class_images) == 0:
            print("✗ No images found")
            return None
        
        print(f"Found {len(class_images)} classes:")
        for cls, imgs in class_images.items():
            print(f"  {cls}: {len(imgs)} images")
        
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
        
        print(f"\n✓ Created 70:10:20 split:")
        print(f"  Train: {len(train_paths)}")
        print(f"  Validation: {len(val_paths)}")
        print(f"  Test: {len(test_paths)}")
    
    # Extract features
    print("\n[Extracting train features...]")
    train_features = extract_features_batch(train_paths, model)
    
    print("\n[Extracting validation features...]")
    val_features = extract_features_batch(val_paths, model)
    
    print("\n[Extracting test features...]")
    test_features = extract_features_batch(test_paths, model)
    
    if train_features is None:
        print("✗ Failed to extract features")
        return None
    
    # Save
    suffix = '_augmented' if only_augmented else ''
    output_path = os.path.join(output_folder, f"{dataset_name}{suffix}_densenet121_features.npz")
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
                        only_augmented=False):
    """Process all datasets with smart split detection"""
    os.makedirs(output_folder, exist_ok=True)
    
    datasets = [d for d in os.listdir(input_folder) 
                if os.path.isdir(os.path.join(input_folder, d))]
    
    print(f"\n{'='*70}")
    print(f"DenseNet121 Feature Extraction (1024 features)")
    print(f"Datasets: {len(datasets)}")
    print(f"Mode: {'AUGMENTED ONLY' if only_augmented else 'ALL IMAGES'}")
    print(f"{'='*70}")
    
    processed = []
    for dataset_name in datasets:
        dataset_path = os.path.join(input_folder, dataset_name)
        result = process_dataset_direct(dataset_path, output_folder, dataset_name, only_augmented)
        if result:
            processed.append(result)
    
    print(f"\n{'='*70}")
    print(f"✓ Processed: {len(processed)}/{len(datasets)}")
    print(f"{'='*70}\n")
    
    return processed

if __name__ == "__main__":
    process_all_datasets(only_augmented=True)
