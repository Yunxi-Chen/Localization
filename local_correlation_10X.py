import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, ifftshift
from scipy.signal import correlate

# ==========================================
# 1. 5G PRS WAVEFORM GENERATION FUNCTIONS
# ==========================================
def generate_prs_sequence(n_res, c_init):
    n_bits = 2 * n_res
    Nc = 1600
    seq_length = n_bits + Nc
    x1, x2 = np.zeros(seq_length, dtype=int), np.zeros(seq_length, dtype=int)
    x1[0] = 1
    for i in range(31): x2[i] = (c_init >> i) & 1
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
            
            # 3GPP Rule: Long CP only for the very first symbol in 0.5ms block
            if s_offset == 0 and l == 0:
                current_cp = cp_first
            else:
                current_cp = cp_normal
                
            ofdm_sym = modulate_and_add_cp(prs_seq, n_fft, current_cp, n_res)
            hs_waveform = np.concatenate((hs_waveform, ofdm_sym))
            
    return hs_waveform

# ==========================================
# 2. GLOBAL OAI ZERO-PADDING FUNCTION
# ==========================================
def global_zero_padding(signal, factor=10):
    """
    Applies exact frequency-domain zero-padding to the ENTIRE signal array.
    """
    N = len(signal)
    N_new = N * factor
    
    X = fft(signal)
    X_padded = np.zeros(N_new, dtype=complex)
    half = N // 2
    X_padded[:half] = X[:half]
    X_padded[-half:] = X[-half:]
    
    upsampled_signal = ifft(X_padded) * factor
    return upsampled_signal

# ==========================================
# 3. CONFIGURATION 
# ==========================================
FS = 122.88e6
HALF_SUBFRAME_LEN = 61440 
N_FFT, N_RES, N_CP_F, N_CP_N = 2048, 1620, 208, 144
MU = 2
K_OS = 10  # 10x Oversampling factor
FS_OS = FS * K_OS  # 1.2288 GHz
SPEED_OF_LIGHT = 299792458.0 

# ==========================================
# 4. DATA LOADING
# ==========================================
skip_samples = int(FS * 1)
search_samples = int(FS * 0.005) 

print("Loading raw signal from /dev/shm/rx_wave_mu_2.bin...")
rx_data = np.fromfile('/dev/shm/rx_wave_mu_2.bin', dtype=np.complex64, 
                      offset=skip_samples * 8, count=search_samples)

# ==========================================
# 5. CORRELATION AND OVERSAMPLING PROCESSING
# ==========================================
result_len = search_samples - HALF_SUBFRAME_LEN + 1
result_len_os = result_len * K_OS

corr_raw = [np.zeros(result_len) for _ in range(4)]
corr_os = [np.zeros(result_len_os) for _ in range(4)]

nIDs = [100, 200, 300, 400]
ant_labels = ['A', 'B', 'C', 'D']
colors = ['blue', 'red', 'green', 'purple']

total_hs = 8
hs_assignments = [
    np.arange(0, total_hs, 4), 
    np.arange(1, total_hs, 4), 
    np.arange(2, total_hs, 4), 
    np.arange(3, total_hs, 4)  
]

print("Computing base correlation and applying global 10x zero-padding...")
for ant_idx in range(4):
    for hs_idx in hs_assignments[ant_idx]:
        ref_hs = get_local_ref_half_subframe(nIDs[ant_idx], hs_idx, N_FFT, N_CP_F, N_CP_N, N_RES, mu=MU)
        res_hs = correlate(rx_data, ref_hs, mode='valid')
        corr_raw[ant_idx] += np.abs(res_hs)
        
    corr_os_complex = global_zero_padding(corr_raw[ant_idx], factor=K_OS)
    corr_os[ant_idx] = np.abs(corr_os_complex)

# ==========================================
# 6. EXTRACT AND COMPARE TOA/DISTANCE
# ==========================================
print("\n" + "="*85)
print("EXTRACTING TOA AND DISTANCE (RAW vs. 10X OVERSAMPLED COMPARISON)")
print("="*85)

# --- ENGINEERING UPDATE: Tighter Search Window ---
# Hardware baseline ~210 samples + 11m physical LoS (~4.5 samples) + 35 samples buffer
MAX_SEARCH_SAMPLES = 250
MAX_SEARCH_SAMPLES_OS = MAX_SEARCH_SAMPLES * K_OS

for ant_idx in range(4):
    for hs_idx in hs_assignments[ant_idx]:
        # Boundaries for Raw 1x Grid (Restricted to MAX_SEARCH_SAMPLES)
        search_start = hs_idx * HALF_SUBFRAME_LEN
        search_end = min(search_start + MAX_SEARCH_SAMPLES, result_len)
        
        # Boundaries for Oversampled 10x Grid (Restricted accordingly)
        search_start_os = hs_idx * HALF_SUBFRAME_LEN * K_OS
        search_end_os = min(search_start_os + MAX_SEARCH_SAMPLES_OS, result_len_os)
        
        if search_start < result_len:
            # --- 1. Peak Extraction on RAW Grid ---
            window_data_raw = corr_raw[ant_idx][search_start:search_end]
            if len(window_data_raw) > 0:
                raw_peak_offset = np.argmax(window_data_raw)
                toa_sec_raw = raw_peak_offset / FS
                dist_raw = toa_sec_raw * SPEED_OF_LIGHT
                
            # --- 2. Peak Extraction on 10x OVERSAMPLED Grid ---
            window_data_os = corr_os[ant_idx][search_start_os:search_end_os]
            if len(window_data_os) > 0:
                os_peak_offset = np.argmax(window_data_os)
                # Convert the 10x grid index back to the base 122.88MHz float format
                fine_peak_offset = os_peak_offset / K_OS  
                toa_sec_fine = fine_peak_offset / FS
                dist_fine = toa_sec_fine * SPEED_OF_LIGHT
            
            # --- 3. Print Side-by-Side Comparison ---
            if hs_idx < 16:
                print(f"Antenna {ant_labels[ant_idx]} | Block {hs_idx:3d} | [RAW ] Rel Peak Sample: {float(raw_peak_offset):6.2f} | Rel TOA: {toa_sec_raw * 1e6:6.4f} us | Distance: {dist_raw:7.3f} m")
                print(f"Antenna {ant_labels[ant_idx]} | Block {hs_idx:3d} | [FINE] Rel Peak Sample: {fine_peak_offset:6.2f} | Rel TOA: {toa_sec_fine * 1e6:6.4f} us | Distance: {dist_fine:7.3f} m")
                print("-" * 85)

# ==========================================
# 7. VISUALIZATION (FULL MACROSCOPIC VIEW)
# ==========================================
t_ms_raw = np.arange(result_len) / FS * 1000 
t_ms_os = np.arange(result_len_os) / FS_OS * 1000 

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

for ant_idx in range(4):
    ax1.plot(t_ms_raw, corr_raw[ant_idx], color=colors[ant_idx], label=f'Antenna {ant_labels[ant_idx]}', linewidth=1.0)
ax1.set_title("Original Correlation Envelope (122.88 MHz Grid)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Magnitude", fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

for ant_idx in range(4):
    ax2.plot(t_ms_os, corr_os[ant_idx], color=colors[ant_idx], label=f'Antenna {ant_labels[ant_idx]} (10x OS)', linewidth=1.0)
ax2.set_title("10x Zero-Padded Correlation Envelope (1.2288 GHz Grid)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Relative Time in Search Window (ms)", fontsize=12)
ax2.set_ylabel("Magnitude", fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.xlim(0, 4.5)
plt.tight_layout()
plt.show()