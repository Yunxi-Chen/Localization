import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, ifftshift
from scipy.signal import correlate, find_peaks
import scipy.io as sio  # Used to export directly to MATLAB .mat files
import csv              # Used to export universal .csv files

# ==========================================
# 1. 5G PRS WAVEFORM GENERATION FUNCTIONS
# ==========================================
def generate_prs_sequence(n_res, c_init):
    n_bits = 2 * n_res
    Nc = 1600
    seq_length = n_bits + Nc
    x1, x2 = np.zeros(seq_length, dtype=int), np.zeros(seq_length, dtype=int)
    x1[0] = 1
    for i in range(31): 
        x2[i] = (c_init >> i) & 1
    for n in range(seq_length - 31):
        x1[n+31] = (x1[n+3] + x1[n]) % 2
        x2[n+31] = (x2[n+3] + x2[n+2] + x2[n+1] + x2[n]) % 2
    c = (x1[Nc:] + x2[Nc:]) % 2
    i_data = (1 - 2 * c[0::2]) / np.sqrt(2)
    q_data = (1 - 2 * c[1::2]) / np.sqrt(2)
    return i_data + 1j * q_data

def modulate_and_add_cp(prs_seq, n_fft, cp_length, n_res):
    fft_grid = np.zeros(n_fft, dtype=complex)
    start = (n_fft - n_res) // 2
    fft_grid[start : start + n_res : 2] = prs_seq 
    time_domain_symbol = ifft(ifftshift(fft_grid)) * np.sqrt(n_fft)
    cp = time_domain_symbol[-cp_length:]
    return np.concatenate((cp, time_domain_symbol))

def get_local_ref_half_subframe(n_id, hs_idx, n_fft, cp_first, cp_normal, n_res, mu=2):
    hs_waveform = np.array([], dtype=complex)
    slots_per_hs = 2 ** (mu - 1) 
    start_slot = hs_idx * slots_per_hs
    
    for s_offset in range(slots_per_hs):
        slot_idx = start_slot + s_offset
        for l in range(14):
            part1 = (2**22) * (n_id // 1024)
            part2 = (2**10) * (14 * slot_idx + l + 1) * (2 * (n_id % 1024) + 1)
            part3 = (n_id % 1024)
            c_init_true = (part1 + part2 + part3) % (2**31)
            prs_seq = generate_prs_sequence(n_res // 2, c_init=c_init_true)
            
            # 3GPP Rule: Long CP is only applied to the very first symbol in a 0.5ms block
            if s_offset == 0 and l == 0:
                current_cp = cp_first
            else:
                current_cp = cp_normal
                
            ofdm_sym = modulate_and_add_cp(prs_seq, n_fft, current_cp, n_res)
            hs_waveform = np.concatenate((hs_waveform, ofdm_sym))
            
    return hs_waveform

# ==========================================
# 2. FREQUENCY-DOMAIN OVER-SAMPLING & PEAK DETECTION
# ==========================================
def global_zero_padding(signal, factor=10):
    N = len(signal)
    N_new = N * factor
    X = fft(signal)
    X_padded = np.zeros(N_new, dtype=complex)
    half = N // 2
    X_padded[:half] = X[:half]
    X_padded[-half:] = X[-half:]
    upsampled_signal = ifft(X_padded) * factor
    return upsampled_signal

def extract_first_peak_in_window(window_data, threshold_ratio=0.7, min_dist=20):
    if len(window_data) == 0:
        return 0
    max_val = np.max(window_data)
    if max_val == 0:
        return 0
    
    # Detect the first prominent peak (Line-of-Sight) exceeding the amplitude threshold
    peaks, _ = find_peaks(window_data, height=max_val * threshold_ratio, distance=min_dist)
    if len(peaks) > 0:
        return peaks[0]
    return np.argmax(window_data)

# ==========================================
# 3. CONFIGURATION & CALIBRATION
# ==========================================
FS = 122.88e6
HALF_SUBFRAME_LEN = 61440 
N_FFT, N_RES, N_CP_F, N_CP_N = 2048, 1620, 208, 144
MU = 2
K_OS = 10
SPEED_OF_LIGHT = 299792458.0 
TX_PERIOD_HS = 20  # Modulo index for a 10ms frame length

nIDs = [100, 200, 300, 400]
ant_labels = ['Antenna A (ch0)', 'Antenna B (ch1)', 'Antenna C (ch2)', 'Antenna D (ch3)']
TARGET_BLOCK_COUNT = 200  # Number of valid snapshots to extract
SEARCH_RANGE = 2000 

# UPDATED: Using SPDE High-Precision Offsets
CALIBRATION_OFFSETS = {
    100: 207.313,  # No extension cable
    200: 210.165,  # With extension cable
    300: 210.165,  # With extension cable
    400: 207.313   # No extension cable
}

# Parameters for CIR visualization
pre_samples_cir = 5    
post_samples_cir = 35  
cir_snapshot_db = {}   

# ==========================================
# 4. DATA LOADING (TWO-FILE SYSTEM)
# ==========================================
skip_samples = int(FS * 1)

FILE_AB = '/home/antlab/Desktop/12_2.9455, 7.817_7g.bin' 
FILE_CD = '/home/antlab/Desktop/34_2.9455, 7.817_7G.bin'

print(f"Loading Signal A & B from: {FILE_AB}")
rx_memmap_AB = np.memmap(FILE_AB, dtype=np.complex64, mode='r', offset=skip_samples * 8)

print(f"Loading Signal C & D from: {FILE_CD}")
rx_memmap_CD = np.memmap(FILE_CD, dtype=np.complex64, mode='r', offset=skip_samples * 8)

# ==========================================
# 5. CYCLIC PROCESSING LOOP (A, B, C, D)
# ==========================================
data_store = {nid: {'raw_dist': [], 'fine_dist': []} for nid in nIDs}
counts = {nid: 0 for nid in nIDs}

print(f"\nProcessing {TARGET_BLOCK_COUNT} Blocks for ALL Antennas (A, B, C, D)...\n")

abs_hs_idx = 0

while any(c < TARGET_BLOCK_COUNT for c in counts.values()):
    ant_idx = abs_hs_idx % 4
    nid = nIDs[ant_idx]
    
    # Skip if we already have enough measurements for this specific antenna ID
    if counts[nid] >= TARGET_BLOCK_COUNT:
        abs_hs_idx += 1
        continue
        
    search_start_abs = abs_hs_idx * HALF_SUBFRAME_LEN
    search_end_abs = search_start_abs + (2 * HALF_SUBFRAME_LEN) - 1
    
    # --- DYNAMIC FILE ROUTING ---
    if nid in [100, 200]:  # Route Antennas A and B to FILE_AB
        if search_end_abs > len(rx_memmap_AB):
            print(f"\nWarning: Reached end of FILE_AB early for Antenna {nid}!")
            counts[nid] = TARGET_BLOCK_COUNT  # Force loop completion for this antenna
            abs_hs_idx += 1
            continue
        rx_chunk = rx_memmap_AB[search_start_abs:search_end_abs]
    else:                  # Route Antennas C and D to FILE_CD
        if search_end_abs > len(rx_memmap_CD):
            print(f"\nWarning: Reached end of FILE_CD early for Antenna {nid}!")
            counts[nid] = TARGET_BLOCK_COUNT  # Force loop completion for this antenna
            abs_hs_idx += 1
            continue
        rx_chunk = rx_memmap_CD[search_start_abs:search_end_abs]
        
    hs_in_tx_period = abs_hs_idx % TX_PERIOD_HS
    cal_offset = CALIBRATION_OFFSETS.get(nid, 0)
    
    ref_hs = get_local_ref_half_subframe(nid, hs_in_tx_period, N_FFT, N_CP_F, N_CP_N, N_RES, mu=MU)
    
    # --- 1. RAW 1x Grid ---
    res_hs = correlate(rx_chunk, ref_hs, mode='valid')
    res_abs = np.abs(res_hs)
    window_data_raw = res_abs[:SEARCH_RANGE]
    
    rel_peak_raw = extract_first_peak_in_window(window_data_raw, threshold_ratio=0.7, min_dist=20)
    toa_sec_raw = (rel_peak_raw - cal_offset) / FS
    dist_raw = toa_sec_raw * SPEED_OF_LIGHT
    
    # Extract the CIR slice for visualization
    if nid not in cir_snapshot_db:
        s_start = max(0, rel_peak_raw - pre_samples_cir)
        s_end = min(len(res_abs), rel_peak_raw + post_samples_cir)
        cir_slice = res_abs[s_start:s_end]
        
        slice_indices = np.arange(s_start, s_end)
        slice_distances = ((slice_indices - cal_offset) / FS) * SPEED_OF_LIGHT
            
        norm_peak = np.max(res_abs) + 1e-12
        power_db = 20 * np.log10(cir_slice / norm_peak + 1e-10)
        
        cir_snapshot_db[nid] = {
            'distance': slice_distances,
            'power_db': power_db
        }

    # --- 2. 10x Oversampled Grid ---
    corr_os_complex = global_zero_padding(res_hs[:SEARCH_RANGE], factor=K_OS)
    window_data_os = np.abs(corr_os_complex)
    
    rel_peak_os = extract_first_peak_in_window(window_data_os, threshold_ratio=0.7, min_dist=20 * K_OS)
    fine_rel_peak = rel_peak_os / float(K_OS)
    toa_sec_fine = (fine_rel_peak - cal_offset) / FS
    dist_fine = toa_sec_fine * SPEED_OF_LIGHT
    
    # --- 3. Save to Memory ---
    data_store[nid]['raw_dist'].append(dist_raw)
    data_store[nid]['fine_dist'].append(dist_fine)
    counts[nid] += 1
    
    # --- 4. Terminal Output (Throttled to avoid flooding) ---
    if counts[nid] <= 5: 
        label = ant_labels[ant_idx]
        print(f"[{counts[nid]:3d}/{TARGET_BLOCK_COUNT}] Antenna {label} (ID {nid})")
        print(f"  [RAW ] Smp: {rel_peak_raw:6.2f} | TOA: {toa_sec_raw * 1e6:6.4f} us | Dist: {dist_raw:7.3f} m")
        print(f"  [FINE] Smp: {fine_rel_peak:6.2f} | TOA: {toa_sec_fine * 1e6:6.4f} us | Dist: {dist_fine:7.3f} m")
        print("-" * 85)
    elif counts[nid] == 6:
        print(f"... Silently processing remaining blocks for Antenna {ant_labels[ant_idx]} ...")
        
    abs_hs_idx += 1

# ==========================================
# 6. DATA EXPORT FOR LOCALIZATION ALGORITHMS
# ==========================================
print("\n" + "="*85)
print("SAVING MEASUREMENT DATA TO DISK...")
print("="*85)

# Ensure rows are aligned by limiting loops to the minimum valid count collected.
# This ensures that if one file is shorter, the exported matrix remains a perfect rectangle.
safe_count = min(counts.values())

csv_filename = "localization_distances.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Measurement_Index', 
                     'Dist_A_Raw', 'Dist_B_Raw', 'Dist_C_Raw', 'Dist_D_Raw', 
                     'Dist_A_Fine', 'Dist_B_Fine', 'Dist_C_Fine', 'Dist_D_Fine'])
    
    for i in range(safe_count):
        writer.writerow([
            i + 1,
            data_store[100]['raw_dist'][i], data_store[200]['raw_dist'][i], 
            data_store[300]['raw_dist'][i], data_store[400]['raw_dist'][i],
            data_store[100]['fine_dist'][i], data_store[200]['fine_dist'][i], 
            data_store[300]['fine_dist'][i], data_store[400]['fine_dist'][i]
        ])
print(f" -> Successfully saved CSV dataset: {csv_filename} ({safe_count} snapshots)")

mat_filename = "localization_distances.mat"
mat_data = {
    'Dist_A_Raw': np.array(data_store[100]['raw_dist'][:safe_count]),
    'Dist_B_Raw': np.array(data_store[200]['raw_dist'][:safe_count]),
    'Dist_C_Raw': np.array(data_store[300]['raw_dist'][:safe_count]),
    'Dist_D_Raw': np.array(data_store[400]['raw_dist'][:safe_count]),
    'Dist_A_Fine': np.array(data_store[100]['fine_dist'][:safe_count]),
    'Dist_B_Fine': np.array(data_store[200]['fine_dist'][:safe_count]),
    'Dist_C_Fine': np.array(data_store[300]['fine_dist'][:safe_count]),
    'Dist_D_Fine': np.array(data_store[400]['fine_dist'][:safe_count])
}
sio.savemat(mat_filename, mat_data)
print(f" -> Successfully saved MATLAB dataset: {mat_filename} ({safe_count} snapshots)")
print("="*85 + "\n")

# ==========================================
# 7. 2D CIR MULTIPATH VISUALIZATION
# ==========================================
print("Rendering 2D CIR Multipath Plot for 4 Antennas (Absolute Distance)...")

plt.figure(figsize=(12, 6))
colors = ['blue', 'red', 'green', 'purple']

min_plot_dist = np.inf
max_plot_dist = -np.inf

for ant_idx in range(4):
    nid = nIDs[ant_idx]
    if nid in cir_snapshot_db:
        dist_axis = cir_snapshot_db[nid]['distance']
        power_db = cir_snapshot_db[nid]['power_db']
        
        plt.plot(
            dist_axis, 
            power_db, 
            label=f"{ant_labels[ant_idx]}", 
            color=colors[ant_idx], 
            linewidth=2.0, 
            alpha=0.85
        )
        
        min_plot_dist = min(min_plot_dist, np.min(dist_axis))
        max_plot_dist = max(max_plot_dist, np.max(dist_axis))

plt.title("5G PRS UE Positioning Channel Impulse Response (Absolute Distance)", fontsize=14, fontweight='bold')
plt.xlabel("Absolute Calibrated Distance (Meters)", fontsize=12)
plt.ylabel("Relative Power (dB)", fontsize=12)

if min_plot_dist != np.inf and max_plot_dist != -np.inf:
    plt.xlim([min_plot_dist, max_plot_dist])
    
plt.ylim([-70, 2])
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.show()