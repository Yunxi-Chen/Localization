import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.fft import ifft, ifftshift
from scipy.signal import correlate, find_peaks

# ==========================================
# 1. 3GPP 5G PRS WAVEFORM GENERATION (TX)
# ==========================================
def generate_prs_sequence(n_res, c_init):
    n_bits = 2 * n_res
    Nc = 1600
    seq_length = n_bits + Nc
    
    x1 = np.zeros(seq_length, dtype=int)
    x2 = np.zeros(seq_length, dtype=int)
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

def generate_reference_signal(n_id, hs_idx, n_fft, cp_first, cp_normal, n_res, mu=2):
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
            current_cp = cp_first if (l == 0 and slot_idx % slots_per_hs == 0) else cp_normal
            ofdm_sym = modulate_and_add_cp(prs_seq, n_fft, current_cp, n_res)
            hs_waveform = np.concatenate((hs_waveform, ofdm_sym))
    return hs_waveform

# ==========================================
# 2. SPDE ALGORITHM (GPU ACCELERATED)
# ==========================================
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
    print("Top 5 Eigenvalues:", eigenvalues[-5:].cpu().numpy())
    
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
    from scipy.signal import find_peaks
    peak_indices, _ = find_peaks(pseudo_np, height=np.max(pseudo_np) * 0.1)
    
    if len(peak_indices) > 0:
        # FAP Logic: Always pick the FIRST peak in time (earliest arrival)
        first_peak_idx = peak_indices[0]
        delta_tau = tau_grid[first_peak_idx].item()
    else:
        # Fallback if no clean peaks are found
        peak_idx = torch.argmax(pseudospectrum)
        delta_tau = tau_grid[peak_idx].item()
        
    return delta_tau, pseudo_np

# ==========================================
# 3. CONFIGURATION & DATA LOADING
# ==========================================
FILE_PATH = '/dev/shm/rx_wave_mu_2.bin' 
FS = 122.88e6  
DURATION = 10.0  
DELTA_F = 60e3  # 60 kHz Subcarrier Spacing for mu=2

N_FFT = 2048
N_RES = 1620
N_CP_F = 208
N_CP_N = 144
MU = 2

TARGET_NID = 100
TARGET_HS = 0  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute Device: {device}")

print("Loading raw signal data...")
num_samples = int(FS * DURATION)
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"File {FILE_PATH} not found!")
rx_data = np.fromfile(FILE_PATH, dtype=np.complex64, count=num_samples)

# ==========================================
# 4. COARSE SYNCHRONIZATION (WITH DIAGNOSTICS)
# ==========================================
print(f"Generating Tx Reference Signal for Antenna {TARGET_NID}...")
tx_sig = generate_reference_signal(
    n_id=TARGET_NID, hs_idx=TARGET_HS, n_fft=N_FFT, cp_first=N_CP_F, cp_normal=N_CP_N, n_res=N_RES, mu=MU
)

start_index = int(1.0 * FS) 
micro_samples = int(FS * 0.02) # Analyze a 20ms window
rx_window = rx_data[start_index : start_index + micro_samples]

print("Running Coarse Synchronization...")
corr = correlate(rx_window, tx_sig, mode='valid', method='fft')
mag = np.abs(corr)

# --- DIAGNOSTIC PLOT: WE MUST SEE THIS ---
plt.figure(figsize=(12, 4))
plt.plot(mag, color='blue')
plt.title(f"Cross-Correlation for NID {TARGET_NID}, HS {TARGET_HS}", fontweight='bold')
plt.xlabel("Sample Index")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()
# -----------------------------------------

peaks, _ = find_peaks(mag, height=np.max(mag) * 0.5, distance=100)
if len(peaks) == 0:
    raise ValueError("ERROR: No peaks found! The target signal is NOT in this 20ms window.")

coarse_peak_idx = peaks[0]
print(f"Peak selected at index: {coarse_peak_idx}")

# ==========================================
# 5. CFR EXTRACTION & FINE TOA (WITH WINDOW ADVANCE)
# ==========================================
print("Extracting CFR and running SPDE...")

# --- ENGINEERING UPDATE: Window Advance Strategy ---
# Advance the window by 1/4 of the CP to catch early LoS paths and avoid ISI
ADVANCE_MARGIN = N_CP_F // 4 

# Define absolute start index of the FFT window
start_fft = coarse_peak_idx + N_CP_F - ADVANCE_MARGIN
rx_symbol = rx_window[start_fft : start_fft + N_FFT]

# Convert to Frequency Domain
rx_fft = np.fft.fft(rx_symbol)
rx_fft_shifted = np.fft.fftshift(rx_fft)

# Extract only the active subcarriers
start_sub = (N_FFT - N_RES) // 2
rx_active_subcarriers = rx_fft_shifted[start_sub : start_sub + N_RES : 2]

# Regenerate pure frequency-domain PRS sequence for l=0 of this slot
slots_per_hs = 2 ** (MU - 1) 
start_slot = TARGET_HS * slots_per_hs
part1 = (2**22) * (TARGET_NID // 1024)
part2 = (2**10) * (14 * start_slot + 0 + 1) * (2 * (TARGET_NID % 1024) + 1)
part3 = (TARGET_NID % 1024)
c_init_true = (part1 + part2 + part3) % (2**31)

tx_freq_seq = generate_prs_sequence(N_RES // 2, c_init=c_init_true)

# Zero-Forcing to extract the Channel Frequency Response (CFR)
cfr_numpy = rx_active_subcarriers / tx_freq_seq
# --- FIX: REMOVE LO LEAKAGE (DC OFFSET) ---
dc_idx = len(cfr_numpy) // 2  # This will be 405
# Interpolate the DC subcarrier using its adjacent neighbors
cfr_numpy[dc_idx] = (cfr_numpy[dc_idx - 1] + cfr_numpy[dc_idx + 1]) / 2.0

cfr_tensor = torch.tensor(cfr_numpy, dtype=torch.complex64, device=device)

# Calculate the effective active subcarrier spacing (Comb-2 = step of 2)
DELTA_F_ACTIVE = DELTA_F * 2

# Run SPDE using the ACTIVE subcarrier spacing
M = len(cfr_tensor) // 2
tau_grid = torch.linspace(0e-9, 700e-9, 20000, device=device) 

delta_tau, pseudospectrum = run_spde_single_antenna(
    cfr_tensor, delta_f=DELTA_F_ACTIVE, M=M, expected_paths=3, tau_grid=tau_grid
)

# Calculate Final Absolute TOA
t_window_start = start_fft / FS
final_toa = t_window_start + delta_tau

print(f"\n--- LOCALIZATION RESULTS ---")
print(f"Coarse Peak Index:     {coarse_peak_idx}")
print(f"FFT Window Start Idx:  {start_fft} (Margin: -{ADVANCE_MARGIN})")
print(f"Base T_window_start:   {t_window_start * 1e6:.6f} us")
print(f"SPDE Sub-sample Delta: {delta_tau * 1e9:.4f} ns")
print(f"Final Absolute TOA:    {final_toa * 1e6:.6f} us")

# ==========================================
# 6. VISUALIZATION (NORMALIZED)
# ==========================================
# Normalize the amplitude so the maximum peak is exactly 1.0
pseudo_norm = pseudospectrum / np.max(pseudospectrum)

plt.figure(figsize=(10, 5))
plt.plot(tau_grid.cpu().numpy() * 1e9, pseudo_norm, color='purple', linewidth=1.5)
plt.axvline(delta_tau * 1e9, color='red', linestyle='--', label=f'Relative Delay: {delta_tau*1e9:.2f} ns')

plt.title(f"SPDE Pseudospectrum (Antenna {TARGET_NID})", fontweight='bold')
plt.xlabel("Relative Time Delay (Delta Tau) in ns")
plt.ylabel("Normalized Pseudospectrum")
plt.ylim([-0.05, 1.1]) # Lock Y-axis to 0-1 scale
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()