import os
import shutil

# =============================================================================
# 1. DIRECTORY CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'data', '0_downloaded_dataset') 
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', '1_raw_clean')
SPLITS = ['train', 'test']

# =============================================================================
# 2. DATA AGGREGATION LOOP
# =============================================================================
print("=" * 60)
print("STEP 0: PREPARING RAW DATASET (AGGREGATION)")
print(f"Reading from: {INPUT_DIR}")
print(f"Extracting to:  {OUTPUT_DIR}")
print("=" * 60)

if not os.path.exists(INPUT_DIR):
    print(f"ERROR: Downloaded dataset folder not found at {INPUT_DIR}")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)
total_files_copied = 0

for split in SPLITS:
    split_path = os.path.join(INPUT_DIR, split)
    
    if not os.path.exists(split_path):
        continue
        
    print(f"\n--> Processing '{split}' split...")
    
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        
        if not os.path.isdir(class_path):
            continue
            
        target_class_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(target_class_path, exist_ok=True)
        
        files = [f for f in os.listdir(class_path) if f.endswith('.mat')]
        
        for file in files:
            source_file = os.path.join(class_path, file)
            target_file = os.path.join(target_class_path, file)
            
            # Sadece kopyala, isme kesinlikle dokunma
            shutil.copy2(source_file, target_file)
            total_files_copied += 1
            
        print(f"    - Copied {len(files)} files for class: {class_name}")

print("\n" + "=" * 60)
print(f"✅ STEP 0 COMPLETED!")
print(f"Total {total_files_copied} .mat files successfully merged.")
