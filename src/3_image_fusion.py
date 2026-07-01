import os
from PIL import Image

# =============================================================================
# 1. DIRECTORY CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_BASE_ROOT = os.path.join(BASE_DIR, 'data', '3_single_channel_images')
OUTPUT_ROOT_FOLDER = os.path.join(BASE_DIR, 'data', '4_final_datasets')

# Processing both Clean and Noisy directories
DATASETS_TO_PROCESS = ["Clean", "15dB", "10dB", "05dB"]

TARGET_WIDTH, TARGET_HEIGHT = 224, 224
CHANNELS = ["GADF_Images", "GASF_Images", "RP_Images"]

try:
    resample_mode = Image.Resampling.LANCZOS
except AttributeError:
    resample_mode = Image.ANTIALIAS

# =============================================================================
# 2. MAIN PROCESSING LOOP (SENSOR FUSION)
# =============================================================================
print("=" * 60)
print("STEP 3: SENSOR FUSION (COMBINING TO MULTI-CHANNEL RGB)")
print("=" * 60)

if not os.path.exists(INPUT_BASE_ROOT):
    print(f"ERROR: Input directory '{INPUT_BASE_ROOT}' does not exist.")
    exit()

for dataset_name in DATASETS_TO_PROCESS:
    input_base_folder = os.path.join(INPUT_BASE_ROOT, dataset_name)
    output_folder = os.path.join(OUTPUT_ROOT_FOLDER, f"{dataset_name}_Combined")
    
    ref_channel_path = os.path.join(input_base_folder, CHANNELS[0])
    
    if not os.path.exists(ref_channel_path):
        print(f"  WARNING: Skipping {dataset_name}. Folder not found at {ref_channel_path}")
        continue
        
    print(f"\n---> Merging Channels for: {dataset_name} ...")
    print(f"     Target Size: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"     Output -> {output_folder}")
    
    for class_folder in os.listdir(ref_channel_path):
        class_path = os.path.join(ref_channel_path, class_folder)
        
        if not os.path.isdir(class_path): 
            continue
        
        print(f"       Processing Class: {class_folder}")
        
        file_count = 0
        for image_file in os.listdir(class_path):
            if not image_file.endswith(".png"): 
                continue
            
            image_paths = [os.path.join(input_base_folder, channel, class_folder, image_file) for channel in CHANNELS]
            
            if all(os.path.exists(p) for p in image_paths):
                try:
                    images = [Image.open(p).convert('L') for p in image_paths]
                    rgb_image = Image.merge("RGB", tuple(images))
                    rgb_image_resized = rgb_image.resize((TARGET_WIDTH, TARGET_HEIGHT), resample_mode)

                    file_name = os.path.splitext(image_file)[0]
                    output_path = os.path.join(output_folder, class_folder, f"{file_name}.jpg")
                    
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    rgb_image_resized.save(output_path)
                    
                    file_count += 1
                    
                    if file_count % 1000 == 0:
                        print(f"         [{file_count}] images merged.")
                        
                except Exception as e:
                    print(f"         Error processing {image_file}: {e}")
            else:
                print(f"         Missing channels for {image_file}, skipping.")

print("\n" + "=" * 60)
print("✅ STEP 3 COMPLETED! All fused RGB datasets (Clean & Noisy) are ready.")