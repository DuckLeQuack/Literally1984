import os
import shutil
import random

# Function to split the dataset into 80% and 20% and create new folders
def split_dataset(dataset_path, new_dataset_path):
    # Create new folders for 80% and 20%
    data_80_path = os.path.join(new_dataset_path, 'data_80')
    data_20_path = os.path.join(new_dataset_path, 'data_20')
    
    os.makedirs(data_80_path, exist_ok=True)
    os.makedirs(data_20_path, exist_ok=True)
    
    # Loop through each label folder in the original dataset
    for label in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, label)
        
        # Skip if it's not a directory (e.g., hidden files)
        if not os.path.isdir(folder_path):
            continue
        
        # Create corresponding label folders in the new data_80 and data_20
        label_data_80_path = os.path.join(data_80_path, label)
        label_data_20_path = os.path.join(data_20_path, label)
        
        os.makedirs(label_data_80_path, exist_ok=True)
        os.makedirs(label_data_20_path, exist_ok=True)
        
        # Get all files that don't end with _bb.png
        files = [f for f in os.listdir(folder_path) if f.endswith('.png') and not f.endswith('_bb.png')]
        
        # Select 20% of these files randomly for the data_20 set
        files_20 = random.sample(files, int(0.2 * len(files)))  # 20% of the files
        
        # The remaining files go into the 80% set
        files_80 = [f for f in files if f not in files_20]
        
        # Move the 20% files to data_20 and the 80% files to data_80
        for file in files_20:
            file_path = os.path.join(folder_path, file)
            new_file_path = os.path.join(label_data_20_path, file)
            
            # Copy the image file
            shutil.copy(file_path, new_file_path)
            
            # Also copy the corresponding .txt file, if it exists
            txt_file = file.replace('.png', '.txt')
            txt_file_path = os.path.join(folder_path, txt_file)
            if os.path.exists(txt_file_path):
                new_txt_file_path = os.path.join(label_data_20_path, txt_file)
                shutil.copy(txt_file_path, new_txt_file_path)
        
        # Now copy the 80% files to data_80
        for file in files_80:
            file_path = os.path.join(folder_path, file)
            new_file_path = os.path.join(label_data_80_path, file)
            
            # Copy the image file
            shutil.copy(file_path, new_file_path)
            
            # Also copy the corresponding .txt file, if it exists
            txt_file = file.replace('.png', '.txt')
            txt_file_path = os.path.join(folder_path, txt_file)
            if os.path.exists(txt_file_path):
                new_txt_file_path = os.path.join(label_data_80_path, txt_file)
                shutil.copy(txt_file_path, new_txt_file_path)

# Example usage
DATASET_PATH = 'data'  # Your original dataset folder path
NEW_DATASET_PATH = 'new_data'  # The new folder where the split dataset will be stored
split_dataset(DATASET_PATH, NEW_DATASET_PATH)
