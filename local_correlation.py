import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.fft import fft, ifft, ifftshift
from scipy.signal import correlate, find_peaks
import scipy.io as sio  # Used to export directly to MATLAB .mat files
import csv              # Used to export universal .csv files

# ==========================================
# 1. 3GPP 5G PRS WAVEFORM GENERATION FUNCTIONS
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
# 2. SIGNAL PROCESSING FUNCTIONS (FINE & SPDE)
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
    if len(window_data) == 0: return 0
    max_val = np.max(window_data)
    if max_val == 0: return 0
    peaks, _ = find_peaks(window_data, height=max_val * threshold_ratio, distance=min_dist)
    if len(peaks) > 0: return peaks[0]
    return np.argmax(window_data)

def run_spde_single_antenna(cfr_tensor, delta_f, M, expected_paths, tau_grid):
    K = cfr_tensor.shape[0]
    device = cfr_tensor.device
    
    # Forward Spatial Smoothing (Toeplitz Construction)
    Z = torch.zeros((M, K - M + 1), dtype=torch.complex64, device=device)
    for i in range(K - M + 1):
        Z[:, i] = cfr_tensor[i : i + M]
    R_z = (Z @ Z.mH) / (K - M + 1) 
    
    # Eigendecomposition & Noise Subspace Extraction
    eigenvalues, eigenvectors = torch.linalg.eigh(R_z)
    
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, idx]
    E_n = eigenvectors[:, expected_paths:] 
    
    # Vectorized Pseudospectrum Projection
    k_indices = torch.arange(M, device=device).unsqueeze(1) 
    steering_matrix = torch.exp(-1j * 2 * torch.pi * delta_f * k_indices * tau_grid)
    projection = E_n.mH @ steering_matrix
    pseudospectrum = 1.0 / torch.sum(torch.abs(projection)**2, dim=0)
    
    # Extract the pseudospectrum to CPU for peak analysis
    pseudo_np = pseudospectrum.cpu().numpy()
    
    # Find all peaks that are at least 10% of the maximum peak's height
    peak_indices, _ = find_peaks(pseudo_np, height=np.max(pseudo_np) * 0.1)
    if len(peak_indices) > 0:
        first_peak_idx = peak_indices[0] # FAP Logic: Earliest arrival
        delta_tau = tau_grid[first_peak_idx].item()
    else:
        peak_idx = torch.argmax(pseudospectrum)
        delta_tau = tau_grid[peak_idx].item()
        
    return delta_tau, pseudo_np

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

# SPDE Config
DELTA_F = 60e3
DELTA_F_ACTIVE = DELTA_F * 2 # 120 kHz for comb-2 structure
ADVANCE_MARGIN = N_CP_F // 4 # Advance the FFT window by 1/4 CP length
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute Device for SPDE: {device}")
tau_grid = torch.linspace(0e-9, 700e-9, 20000, device=device) # Pre-allocate grid on GPU

# Hardware Calibration Offsets (Based on SPDE measurements)
CALIBRATION_OFFSETS = {
    100: 207.313,  # No extension cable
    200: 210.165,  # With extension cable
    300: 210.165,  # With extension cable
    400: 207.313   # No extension cable
}

# --- RESTORED: Parameters for CIR visualization ---
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
data_store = {nid: {'raw_dist': [], 'fine_dist': [], 'spde_dist': []} for nid in nIDs}
counts = {nid: 0 for nid in nIDs}

print(f"\nProcessing {TARGET_BLOCK_COUNT} Blocks for ALL Antennas (A, B, C, D)...\n")

abs_hs_idx = 0

while any(c < TARGET_BLOCK_COUNT for c in counts.values()):
    ant_idx = abs_hs_idx % 4
    nid = nIDs[ant_idx]
    
    if counts[nid] >= TARGET_BLOCK_COUNT:
        abs_hs_idx += 1
        continue
        
    search_start_abs = abs_hs_idx * HALF_SUBFRAME_LEN
    search_end_abs = search_start_abs + (2 * HALF_SUBFRAME_LEN) - 1
    
    # --- DYNAMIC FILE ROUTING ---
    if nid in [100, 200]: 
        if search_end_abs > len(rx_memmap_AB):
            counts[nid] = TARGET_BLOCK_COUNT 
            abs_hs_idx += 1
            continue
        rx_chunk = rx_memmap_AB[search_start_abs:search_end_abs]
    else:                 
        if search_end_abs > len(rx_memmap_CD):
            counts[nid] = TARGET_BLOCK_COUNT 
            abs_hs_idx += 1
            continue
        rx_chunk = rx_memmap_CD[search_start_abs:search_end_abs]
        
    hs_in_tx_period = abs_hs_idx % TX_PERIOD_HS
    cal_offset = CALIBRATION_OFFSETS.get(nid, 0)
    
    ref_hs = get_local_ref_half_subframe(nid, hs_in_tx_period, N_FFT, N_CP_F, N_CP_N, N_RES, mu=MU)
    
    # ---------------------------------------------------------
    # METHOD 1: RAW 1x Grid 
    # ---------------------------------------------------------
    res_hs = correlate(rx_chunk, ref_hs, mode='valid')
    res_abs = np.abs(res_hs)
    window_data_raw = res_abs[:SEARCH_RANGE]
    
    rel_peak_raw = extract_first_peak_in_window(window_data_raw, threshold_ratio=0.7, min_dist=20)
    toa_sec_raw = (rel_peak_raw - cal_offset) / FS
    dist_raw = toa_sec_raw * SPEED_OF_LIGHT
    
    # --- RESTORED: Extract the CIR slice for visualization ---
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

    # ---------------------------------------------------------
    # METHOD 2: 10x Oversampled Grid 
    # ---------------------------------------------------------
    corr_os_complex = global_zero_padding(res_hs[:SEARCH_RANGE], factor=K_OS)
    window_data_os = np.abs(corr_os_complex)
    
    rel_peak_os = extract_first_peak_in_window(window_data_os, threshold_ratio=0.7, min_dist=20 * K_OS)
    fine_rel_peak = rel_peak_os / float(K_OS)
    toa_sec_fine = (fine_rel_peak - cal_offset) / FS
    dist_fine = toa_sec_fine * SPEED_OF_LIGHT
    
    # ---------------------------------------------------------
    # METHOD 3: SPDE (Subspace Super-Resolution)
    # ---------------------------------------------------------
    start_fft_rel = int(rel_peak_raw) + N_CP_F - ADVANCE_MARGIN
    rx_symbol = rx_chunk[start_fft_rel : start_fft_rel + N_FFT]
    
    rx_fft_shifted = np.fft.fftshift(np.fft.fft(rx_symbol))
    start_sub = (N_FFT - N_RES) // 2
    rx_active_subcarriers = rx_fft_shifted[start_sub : start_sub + N_RES : 2]
    
    slots_per_hs = 2 ** (MU - 1) 
    start_slot = hs_in_tx_period * slots_per_hs
    part1 = (2**22) * (nid // 1024)
    part2 = (2**10) * (14 * start_slot + 0 + 1) * (2 * (nid % 1024) + 1)
    part3 = (nid % 1024)
    c_init_true = (part1 + part2 + part3) % (2**31)
    tx_freq_seq = generate_prs_sequence(N_RES // 2, c_init=c_init_true)
    
    cfr_numpy = rx_active_subcarriers / tx_freq_seq
    dc_idx = len(cfr_numpy) // 2 
    cfr_numpy[dc_idx] = (cfr_numpy[dc_idx - 1] + cfr_numpy[dc_idx + 1]) / 2.0
    
    cfr_tensor = torch.tensor(cfr_numpy, dtype=torch.complex64, device=device)
    M = len(cfr_tensor) // 2
    delta_tau, _ = run_spde_single_antenna(cfr_tensor, delta_f=DELTA_F_ACTIVE, M=M, expected_paths=3, tau_grid=tau_grid)
    
    t_window_start_sec = start_fft_rel / float(FS)
    symbol_toa = t_window_start_sec + delta_tau
    frame_toa = symbol_toa - (N_CP_F / float(FS))
    toa_sec_spde = frame_toa - (cal_offset / float(FS))
    dist_spde = toa_sec_spde * SPEED_OF_LIGHT

    # --- Save to Memory ---
    data_store[nid]['raw_dist'].append(dist_raw)
    data_store[nid]['fine_dist'].append(dist_fine)
    data_store[nid]['spde_dist'].append(dist_spde)
    counts[nid] += 1
    
    # --- Terminal Output ---
    if counts[nid] <= 5: 
        label = ant_labels[ant_idx]
        print(f"[{counts[nid]:3d}/{TARGET_BLOCK_COUNT}] {label} (ID {nid})")
        print(f"  [RAW ] Dist: {dist_raw:7.4f} m | TOA: {toa_sec_raw * 1e6:6.4f} us")
        print(f"  [FINE] Dist: {dist_fine:7.4f} m | TOA: {toa_sec_fine * 1e6:6.4f} us")
        print(f"  [SPDE] Dist: {dist_spde:7.4f} m | TOA: {toa_sec_spde * 1e6:6.4f} us")
        print("-" * 85)
    elif counts[nid] == 6:
        print(f"... Silently processing remaining blocks for {ant_labels[ant_idx]} ...")
        
    abs_hs_idx += 1

# ==========================================
# 6. DATA EXPORT FOR LOCALIZATION ALGORITHMS
# ==========================================
print("\n" + "="*85)
print("SAVING MEASUREMENT DATA TO DISK...")
print("="*85)

safe_count = min(counts.values())

csv_filename = "localization_distances.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Measurement_Index', 
        'Dist_A_Raw', 'Dist_B_Raw', 'Dist_C_Raw', 'Dist_D_Raw', 
        'Dist_A_Fine', 'Dist_B_Fine', 'Dist_C_Fine', 'Dist_D_Fine',
        'Dist_A_SPDE', 'Dist_B_SPDE', 'Dist_C_SPDE', 'Dist_D_SPDE'
    ])
    
    for i in range(safe_count):
        writer.writerow([
            i + 1,
            data_store[100]['raw_dist'][i], data_store[200]['raw_dist'][i], 
            data_store[300]['raw_dist'][i], data_store[400]['raw_dist'][i],
            data_store[100]['fine_dist'][i], data_store[200]['fine_dist'][i], 
            data_store[300]['fine_dist'][i], data_store[400]['fine_dist'][i],
            data_store[100]['spde_dist'][i], data_store[200]['spde_dist'][i], 
            data_store[300]['spde_dist'][i], data_store[400]['spde_dist'][i]
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
    'Dist_D_Fine': np.array(data_store[400]['fine_dist'][:safe_count]),
    
    'Dist_A_SPDE': np.array(data_store[100]['spde_dist'][:safe_count]),
    'Dist_B_SPDE': np.array(data_store[200]['spde_dist'][:safe_count]),
    'Dist_C_SPDE': np.array(data_store[300]['spde_dist'][:safe_count]),
    'Dist_D_SPDE': np.array(data_store[400]['spde_dist'][:safe_count])
}
sio.savemat(mat_filename, mat_data)
print(f" -> Successfully saved MATLAB dataset: {mat_filename} ({safe_count} snapshots)")
print("="*85 + "\n")

# ==========================================
# 7. RESTORED: 2D CIR MULTIPATH VISUALIZATION
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