import os
import shutil
import random

def subsample():
    base_dir = "ai_training/dataset/LumbarSpinalStenosis"
    target_dir = "ai_training/dataset/subset_LumbarSpinalStenosis"
    
    train_count = 200
    test_count = 50
    
    classes = ['Herniated Disc', 'No Stenosis', 'Thecal Sac']
    
    for split in ['train', 'test']:
        count = train_count if split == 'train' else test_count
        for cls in classes:
            src_path = os.path.join(base_dir, split, cls)
            dst_path = os.path.join(target_dir, split, cls)
            
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            
            files = os.listdir(src_path)
            selected_files = random.sample(files, min(len(files), count))
            
            print(f"Copying {len(selected_files)} files for {split}/{cls}...")
            for f in selected_files:
                shutil.copy(os.path.join(src_path, f), os.path.join(dst_path, f))

if __name__ == "__main__":
    subsample()
