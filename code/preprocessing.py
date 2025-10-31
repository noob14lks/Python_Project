import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

def find_all_images(folder_path, skip_augmented=True):
    """Find all images, optionally skipping 'augmented' folders"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG')
    all_images = []
    
    for root, dirs, files in os.walk(folder_path):
        if skip_augmented and 'augmented' in root.lower():
            continue
        
        for file in files:
            if file.lower().endswith(image_extensions):
                all_images.append(os.path.join(root, file))
    
    return all_images

def load_images_with_labels(image_paths, img_size=(224, 224), batch_size=500):
    """Load images in batches to avoid memory issues"""
    images = []
    labels = []
    class_dict = {}
    class_counter = 0
    
    print(f"  Loading {len(image_paths)} images in batches of {batch_size}...")
    
    # Process in batches
    for batch_start in range(0, len(image_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_paths = image_paths[batch_start:batch_end]
        
        print(f"    Batch {batch_start//batch_size + 1}: {batch_start}-{batch_end}", end='\r')
        
        for img_path in batch_paths:
            # Get class from parent folder
            parent_dir = os.path.basename(os.path.dirname(img_path))
            
            # Skip generic folder names
            if parent_dir.lower() in ['original', 'augmented', 'images']:
                parent_dir = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            
            if parent_dir not in class_dict:
                class_dict[parent_dir] = class_counter
                print(f"\n    Class {class_counter}: {parent_dir}")
                class_counter += 1
            
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                # Use float32 instead of float64 to save memory
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                labels.append(class_dict[parent_dir])
            except Exception as e:
                continue
    
    print(f"\n  ✓ Loaded {len(images)} images")
    
    if len(images) == 0:
        return None, None, None
    
    class_names = sorted(class_dict.keys(), key=lambda x: class_dict[x])
    
    # Convert to numpy arrays with float32 (saves 50% memory vs float64)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32), class_names

def split_and_prepare_dataset(dataset_path, output_folder, dataset_name, skip_augmented=True):
    """Process dataset with memory-efficient loading"""
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}")
    print(f"Skip augmented: {skip_augmented}")
    print(f"{'='*70}")
    
    # Check for train/test structure
    has_train = any('train' in d.lower() for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d)))
    has_test = any('test' in d.lower() for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d)))
    
    if has_train and has_test:
        print("✓ Found train/test split")
        
        # Find folders
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
            print("✗ Could not locate folders")
            return None
        
        # Load train
        print(f"\nLoading train from: {train_folder}")
        train_images = find_all_images(train_folder, skip_augmented)
        X_train, y_train, class_names = load_images_with_labels(train_images)
        
        if X_train is None:
            return None
        
        # Load test
        print(f"\nLoading test from: {test_folder}")
        test_images = find_all_images(test_folder, skip_augmented)
        X_test, y_test, _ = load_images_with_labels(test_images)
        
        if X_test is None:
            return None
        
        # Validation split
        val_size = min(0.14, max(0.1, 2.0 / X_train.shape[0]))
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42,
            stratify=y_train if len(np.unique(y_train)) > 1 else None
        )
        
    else:
        print("✗ No train/test split - creating 70:10:20")
        
        all_images = find_all_images(dataset_path, skip_augmented)
        
        if len(all_images) == 0:
            print("✗ No images found")
            return None
        
        X, y, class_names = load_images_with_labels(all_images)
        
        if X is None or X.shape[0] < 10:
            return None
        
        # Split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.67, random_state=42,
            stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )
        
        print(f"\n  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Save with compression
    output_path = os.path.join(output_folder, f"{dataset_name}_preprocessed.npz")
    np.savez_compressed(output_path,
                       X_train=X_train, y_train=y_train,
                       X_val=X_val, y_val=y_val,
                       X_test=X_test, y_test=y_test,
                       class_names=class_names)
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  Classes: {class_names}")
    
    # Free memory
    del X_train, X_val, X_test, y_train, y_val, y_test
    
    return output_path

def preprocess_all_datasets(input_folder='./data/extracteddata', 
                           output_folder='./preprocessed',
                           skip_augmented=True):
    """Process all datasets"""
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(input_folder):
        print(f"✗ Error: '{input_folder}' not found!")
        return []
    
    datasets = [d for d in os.listdir(input_folder) 
                if os.path.isdir(os.path.join(input_folder, d))]
    
    print(f"\n{'='*70}")
    print(f"Found {len(datasets)} datasets")
    print(f"Memory-efficient mode enabled (float32, batched loading)")
    print(f"{'='*70}")
    
    processed_files = []
    for dataset_name in datasets:
        try:
            dataset_path = os.path.join(input_folder, dataset_name)
            output_path = split_and_prepare_dataset(
                dataset_path, output_folder, dataset_name, skip_augmented
            )
            if output_path:
                processed_files.append(output_path)
        except MemoryError as e:
            print(f"\n✗ Memory error processing {dataset_name}: {e}")
            print("Try setting skip_augmented=True to reduce data size")
            continue
    
    print(f"\n{'='*70}")
    print(f"✓ Preprocessed: {len(processed_files)}/{len(datasets)}")
    print(f"{'='*70}\n")
    
    return processed_files

if __name__ == "__main__":
    preprocess_all_datasets(skip_augmented=False)
