import os
import numpy as np
import scipy.io

# =============================================================================
# 1. DIRECTORY CONFIGURATION
# =============================================================================
# Dynamically determine the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input directory (Clean data generated in Step 0)
SOURCE_ROOT_DIR = os.path.join(BASE_DIR, 'data', '1_raw_clean')

# Output directory for the newly generated noisy data
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'data', '2_noisy_mat')

# Target Signal-to-Noise Ratio (SNR) levels in dB
SNR_LEVELS = [15, 10, 5] 

# ---> CRITICAL: Global Random Seed
# Ensures that the generated noise is reproducible across different runs
np.random.seed(42) 

# =============================================================================
# 2. PINK NOISE GENERATOR (1/f)
# =============================================================================
def generate_pink_noise(n_samples):
    """
    Generates 1/f Pink Noise to simulate realistic environmental acoustic background.
    """
    white = np.random.randn(n_samples)
    X_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    
    # Avoid division by zero at DC (0 Hz)
    freqs[0] = 1 
    
    # Scale spectrum by 1/sqrt(f) to obtain a 1/f power spectrum
    S_pink = X_white / np.sqrt(freqs)
    S_pink[0] = 0 # Remove DC offset
    
    pink = np.fft.irfft(S_pink)
    
    # Fix potential length mismatch caused by FFT
    if len(pink) < n_samples:
        pink = np.append(pink, pink[-1])
        
    return pink[:n_samples]

# =============================================================================
# 3. MAIN PROCESSING LOOP
# =============================================================================
print("=" * 60)
print(f"STEP 1: PINK NOISE INJECTION FOR SNRs: {SNR_LEVELS} dB")
print("=" * 60)

# Check if the clean dataset is prepared
if not os.path.exists(SOURCE_ROOT_DIR):
    print(f"ERROR: Source directory '{SOURCE_ROOT_DIR}' does not exist.")
    print("Please run Step 0 (0_prepare_raw_data.py) first.")
    exit()

for snr in SNR_LEVELS:
    target_root_dir = os.path.join(OUTPUT_BASE_DIR, f"Field_Data_{snr:02d}dB")
    print(f"\n---> Processing SNR: {snr} dB...")
    
    file_count = 0
    
    # Traverse through the clean dataset directory
    for root, dirs, files in os.walk(SOURCE_ROOT_DIR):
        # Replicate the folder structure for the noisy output
        relative_path = os.path.relpath(root, SOURCE_ROOT_DIR)
        target_dir = os.path.join(target_root_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            if file.endswith(".mat"):
                file_count += 1
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)
                
                try:
                    # 1. Load Original Data
                    mat = scipy.io.loadmat(source_path)
                    data = mat['data']
                    rows, cols = data.shape
                    
                    noisy_data = np.zeros_like(data)
                    
                    # Process each channel/column
                    for col in range(cols):
                        signal = data[:, col]
                        
                        # Generate unique noise sequence for the current signal
                        noise = generate_pink_noise(rows)
                        
                        # Calculate signal and noise power
                        s_power = np.mean(signal ** 2)
                        n_power_needed = s_power / (10 ** (snr / 10))
                        current_n_power = np.mean(noise ** 2)
                        
                        if current_n_power > 0:
                            # Scale the noise to match the target SNR and inject it
                            scale = np.sqrt(n_power_needed / current_n_power)
                            noisy_data[:, col] = signal + (noise * scale)
                        else:
                            noisy_data[:, col] = signal
                    
                    # 2. Save the New Noisy File
                    mat['data'] = noisy_data
                    scipy.io.savemat(target_path, mat)
                    
                    # Logging progress
                    if file_count % 1000 == 0:
                        print(f"     [{file_count}] files processed.")
                        
                except Exception as e:
                    print(f"     ERROR processing {file}: {e}")

    print(f"     DONE. Total {file_count} files created for {snr} dB.")

print("\n" + "=" * 60)
print("✅ STEP 1 COMPLETED! Noisy datasets are ready.")