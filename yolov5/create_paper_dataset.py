import os
import random
import shutil
from pathlib import Path

# --- CONFIGURATION ---

# 1. Define the classes we want to extract (matching paper)
#    and their folder names in the source dataset
CLASSES_TO_EXTRACT = {
    'persons': 2,   # Original class index in synrailobs (assuming 0:animal, 1:moto, 2:person, 3:rock, 4:vehicle)
    'rocks': 3,
    'vehicles': 4
}

# 2. Define the new class mapping for our 3-class dataset
#    (We will remap the .txt files to these new indices)
NEW_CLASS_MAP = {
    2: 0,  # 'persons' (original index 2) will become index 0
    3: 1,  # 'rocks' (original index 3) will become index 1
    4: 2   # 'vehicles' (original index 4) will become index 2
}
NEW_CLASS_NAMES = ['person', 'rock', 'vehicle'] # The data.yaml names

# 3. Define the 80/20 split for the paper's 2,534 images
#    (Total 2534 = 2027 train + 507 val)
#    We will sample equally from our 3 classes.
NUM_TRAIN_PER_CLASS = 676  # (676 * 3 = 2028, close enough to 2027)
NUM_VAL_PER_CLASS = 169    # (169 * 3 = 507)

# --- SCRIPT ---

def remap_label_file(src_path, dest_path):
    """
    Reads a source .txt file, remaps the class indices,
    and writes to a new destination file.
    """
    new_lines = []
    with open(src_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        original_class_index = int(parts[0])
        
        # Check if this class is one we want to keep
        if original_class_index in NEW_CLASS_MAP:
            new_class_index = NEW_CLASS_MAP[original_class_index]
            # Reconstruct the line with the new class index
            new_line = f"{new_class_index} {' '.join(parts[1:])}\n"
            new_lines.append(new_line)
        # If it's not in the map (e.g., 'animal' or 'moto'), we discard it.

    # Only write the file if it still contains valid labels
    if new_lines:
        with open(dest_path, 'w') as f:
            f.writelines(new_lines)
        return True
    return False

def process_class(class_name, original_index, base_src_path, base_dest_path):
    """
    Samples and processes all files for a single class.
    """
    print(f"--- Processing class: {class_name} ---")
    
    src_img_dir = os.path.join(base_src_path, class_name, 'imgs')
    src_lbl_dir = os.path.join(base_src_path, class_name, 'anno')
    
    if not os.path.isdir(src_img_dir):
        print(f"Error: Source directory not found: {src_img_dir}")
        return

    all_images = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_images)
    
    # Check if we have enough images
    total_needed = NUM_TRAIN_PER_CLASS + NUM_VAL_PER_CLASS
    if len(all_images) < total_needed:
        print(f"Warning: Not enough images for class '{class_name}'.")
        print(f"Need {total_needed}, but only have {len(all_images)}.")
        # Adjust numbers if we don't have enough
        available_train = int(len(all_images) * 0.8)
        available_val = len(all_images) - available_train
    else:
        available_train = NUM_TRAIN_PER_CLASS
        available_val = NUM_VAL_PER_CLASS
        
    # Split the list
    train_images = all_images[:available_train]
    val_images = all_images[available_train : available_train + available_val]
    
    # Process training set
    print(f"Copying {len(train_images)} train images for '{class_name}'...")
    copy_files(train_images, src_img_dir, src_lbl_dir, os.path.join(base_dest_path, 'images', 'train'), os.path.join(base_dest_path, 'labels', 'train'))
    
    # Process validation set
    print(f"Copying {len(val_images)} val images for '{class_name}'...")
    copy_files(val_images, src_img_dir, src_lbl_dir, os.path.join(base_dest_path, 'images', 'val'), os.path.join(base_dest_path, 'labels', 'val'))

def copy_files(image_list, src_img_dir, src_lbl_dir, dest_img_dir, dest_lbl_dir):
    """
    Helper function to copy and remap files.
    """
    for img_name in image_list:
        base_name = Path(img_name).stem
        src_img_path = os.path.join(src_img_dir, img_name)
        src_lbl_path = os.path.join(src_lbl_dir, base_name + '.txt')
        
        dest_img_path = os.path.join(dest_img_dir, img_name)
        dest_lbl_path = os.path.join(dest_lbl_dir, base_name + '.txt')
        
        if os.path.exists(src_lbl_path):
            # Remap and write the new label file
            if remap_label_file(src_lbl_path, dest_lbl_path):
                # Only copy the image if the label file wasn't empty
                shutil.copy2(src_img_path, dest_img_path)
        else:
            print(f"Skipping {img_name}: No label found.")

def main():
    # Path to your 'gen' folder in the synrailobs dataset
    base_src_path = "/content/drive/MyDrive/yolo/gen" 
    
    # Path for the new dataset
    base_dest_path = "/content/drive/MyDrive/yolo/paper_dataset_2534"
    
    print(f"Creating new dataset at: {base_dest_path}")
    
    # Create the new directory structure
    Path(base_dest_path, 'images', 'train').mkdir(parents=True, exist_ok=True)
    Path(base_dest_path, 'images', 'val').mkdir(parents=True, exist_ok=True)
    Path(base_dest_path, 'labels', 'train').mkdir(parents=True, exist_ok=True)
    Path(base_dest_path, 'labels', 'val').mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_name, original_index in CLASSES_TO_EXTRACT.items():
        process_class(class_name, original_index, base_src_path, base_dest_path)
        
    print("--- Dataset Creation Complete ---")
    print("Next, create your 'paper_data.yaml' file.")

if __name__ == "__main__":
    main()