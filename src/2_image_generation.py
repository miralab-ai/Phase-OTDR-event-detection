import os
import scipy.io
import numpy as np
from pyts.image import GramianAngularField, RecurrencePlot
from PIL import Image

# =============================================================================
# 1. DIRECTORY CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input directories
CLEAN_INPUT_DIR = os.path.join(BASE_DIR, 'data', '1_raw_clean')
NOISY_INPUT_DIR = os.path.join(BASE_DIR, 'data', '2_noisy_mat')

# Output directory
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'data', '3_single_channel_images')

# Define processing tasks: (Input_Path, Output_Folder_Name)
TASKS = [
    (CLEAN_INPUT_DIR, "Clean") # Add the clean, noiseless dataset first
]

# Add the noisy datasets to the task list
SNR_LEVELS = [15, 10, 5]
for snr in SNR_LEVELS:
    snr_path = os.path.join(NOISY_INPUT_DIR, f"Field_Data_{snr:02d}dB")
    TASKS.append((snr_path, f"{snr:02d}dB"))

# Grid parameters
CHOSEN_SIZE = 500   
ROWS, COLS = 3, 4

# =============================================================================
# 2. MAIN PROCESSING LOOP
# =============================================================================
print("=" * 60)
print("STEP 2: TRANSFORMING 1D SIGNALS TO 2D IMAGES (INCLUDING CLEAN DATA)")
print("=" * 60)

for input_dir, output_name in TASKS:
    output_dir = os.path.join(OUTPUT_BASE_DIR, output_name)
    
    if not os.path.exists(input_dir):
        print(f"  WARNING: Skipping {output_name}. Input folder not found at {input_dir}")
        continue
        
    print(f"\n---> Transforming Data for: {output_name} ...")
    
    for class_folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_folder)
        
        if not os.path.isdir(class_path): 
            continue
        
        print(f"     Processing Class: {class_folder}")
        
        os.makedirs(os.path.join(output_dir, "GASF_Images", class_folder), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "GADF_Images", class_folder), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "RP_Images", class_folder), exist_ok=True)
        
        file_count = 0
        for file in os.listdir(class_path):
            if file.endswith(".mat"):
                file_count += 1
                file_path = os.path.join(class_path, file)
                
                try:
                    data = scipy.io.loadmat(file_path)['data']

                    # ---------------------------------------------------------
                    # PART 1: GASF TRANSFORMATION
                    # ---------------------------------------------------------
                    gasf_images = []
                    for i in range(12):
                        sensor_data = data[:, i]
                        gasf = GramianAngularField(image_size=CHOSEN_SIZE, method='summation')
                        gasf_images.append(gasf.fit_transform(sensor_data.reshape(1, -1))[0])

                    gasf_combined = np.zeros((ROWS * CHOSEN_SIZE, COLS * CHOSEN_SIZE), dtype=np.uint8)
                    for i in range(ROWS):
                        for j in range(COLS):
                            x_start, y_start = i * CHOSEN_SIZE, j * CHOSEN_SIZE
                            pixel_val = (gasf_images[i * COLS + j] * 255).astype(np.uint8)
                            gasf_combined[x_start:x_start + CHOSEN_SIZE, y_start:y_start + CHOSEN_SIZE] = pixel_val

                    # ---------------------------------------------------------
                    # PART 2: GADF TRANSFORMATION
                    # ---------------------------------------------------------
                    gadf_images = []
                    for i in range(12):
                        sensor_data = data[:, i]
                        gadf = GramianAngularField(image_size=CHOSEN_SIZE, method='difference')
                        gadf_images.append(gadf.fit_transform(sensor_data.reshape(1, -1))[0])

                    gadf_combined = np.zeros((ROWS * CHOSEN_SIZE, COLS * CHOSEN_SIZE), dtype=np.uint8)
                    for i in range(ROWS):
                        for j in range(COLS):
                            x_start, y_start = i * CHOSEN_SIZE, j * CHOSEN_SIZE
                            pixel_val = (gadf_images[i * COLS + j] * 255).astype(np.uint8)
                            gadf_combined[x_start:x_start + CHOSEN_SIZE, y_start:y_start + CHOSEN_SIZE] = pixel_val

                    # ---------------------------------------------------------
                    # PART 3: RP TRANSFORMATION (WITH DOWNSAMPLING)
                    # ---------------------------------------------------------
                    rp_images = []
                    for i in range(12):
                        sensor_data = data[:, i]
                        step_count = len(sensor_data) // CHOSEN_SIZE
                        if step_count == 0: step_count = 1
                        selected_indices = np.arange(0, len(sensor_data), step_count)
                        downsampled_data = np.array(sensor_data)[selected_indices]

                        rp = RecurrencePlot(dimension=1)
                        rp_images.append(rp.fit_transform(downsampled_data.reshape(1, -1))[0])

                    rp_combined = np.zeros((ROWS * CHOSEN_SIZE, COLS * CHOSEN_SIZE), dtype=np.uint8)
                    for i in range(ROWS):
                        for j in range(COLS):
                            x_start, y_start = i * CHOSEN_SIZE, j * CHOSEN_SIZE
                            if (i * COLS + j) < len(rp_images):
                                img_vals = rp_images[i * COLS + j]
                                h, w = img_vals.shape
                                h_end = min(CHOSEN_SIZE, h)
                                w_end = min(CHOSEN_SIZE, w)
                                pixel_val = (img_vals[:h_end, :w_end] * 255).astype(np.uint8)
                                rp_combined[x_start:x_start + h_end, y_start:y_start + w_end] = pixel_val

                    # ---------------------------------------------------------
                    # SAVE IMAGES
                    # ---------------------------------------------------------
                    file_raw_name = os.path.splitext(file)[0]
                    Image.fromarray(gasf_combined, mode='L').save(os.path.join(output_dir, "GASF_Images", class_folder, f"{file_raw_name}.png"))
                    Image.fromarray(gadf_combined, mode='L').save(os.path.join(output_dir, "GADF_Images", class_folder, f"{file_raw_name}.png"))
                    Image.fromarray(rp_combined, mode='L').save(os.path.join(output_dir, "RP_Images", class_folder, f"{file_raw_name}.png"))

                    if file_count % 100 == 0:
                        print(f"       [{file_count}] images successfully generated.")

                except Exception as e:
                    print(f"      ERROR processing {file}: {e}")

print("\n" + "=" * 60)
print("✅ STEP 2 COMPLETED! Datasets (Clean & Noisy) are successfully transformed.")