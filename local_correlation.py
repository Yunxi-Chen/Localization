import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft, ifftshift
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

# UPDATED: Replaced per-slot generation with a strict 0.5ms half-subframe block generator
def get_local_ref_half_subframe(n_id, hs_idx, n_fft, cp_first, cp_normal, n_res, mu=2):
    hs_waveform = np.array([], dtype=complex)
    
    # Calculate how many slots are in 0.5ms (mu=1 -> 1 slot, mu=2 -> 2 slots)
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
            
            # STRICT 3GPP CP RULE: Only apply long CP to the very first symbol of the 0.5ms block
            if s_offset == 0 and l == 0:
                current_cp = cp_first
            else:
                current_cp = cp_normal
                
            ofdm_sym = modulate_and_add_cp(prs_seq, n_fft, current_cp, n_res)
            hs_waveform = np.concatenate((hs_waveform, ofdm_sym))
            
    # This guarantees exactly 61,440 samples for a 122.88MHz sample rate
    return hs_waveform

# ==========================================
# 2. CONFIGURATION (mu=2, 4 Antennas)
# ==========================================
FS = 122.88e6
# Replaced SLOT_LEN with HALF_SUBFRAME_LEN for the 0.5ms grid
HALF_SUBFRAME_LEN = 61440 
N_FFT, N_RES, N_CP_F, N_CP_N = 2048, 1620, 208, 144
MU = 2

# ==========================================
# 3. DATA LOADING
# ==========================================
skip_samples = int(FS * 1)
search_samples = int(FS * 0.005) 

print("Loading 110ms of raw signal from /dev/shm/rx_wave_mu_2.bin...")
rx_data = np.fromfile('/dev/shm/rx_wave_mu_2.bin', dtype=np.complex64, 
                      offset=skip_samples * 8, count=search_samples)

# ==========================================
# 4. BLOCK-BY-BLOCK SLIDING CORRELATION & TOA
# ==========================================
result_len = search_samples - HALF_SUBFRAME_LEN + 1
corr_total = [np.zeros(result_len) for _ in range(4)]
toa_results = {0: [], 1: [], 2: [], 3: []}

nIDs = [100, 200, 300, 400]
ant_labels = ['A', 'B', 'C', 'D']

# Dynamically generate the 4-block TDM pattern across the 110ms window
# 110ms contains 220 half-subframes (0.5ms each)
total_hs = 8
hs_assignments = [
    np.arange(0, total_hs, 4), # Antenna A 
    np.arange(1, total_hs, 4), # Antenna B 
    np.arange(2, total_hs, 4), # Antenna C 
    np.arange(3, total_hs, 4)  # Antenna D 
]

print("Performing Sliding Match and TOA Extraction (0.5ms Blocks)...")

# Speed of light in meters per second
SPEED_OF_LIGHT = 299792458.0 

for ant_idx in range(4):
    for hs_idx in hs_assignments[ant_idx]:
        # Generate 0.5ms block
        ref_hs = get_local_ref_half_subframe(nIDs[ant_idx], hs_idx, N_FFT, N_CP_F, N_CP_N, N_RES, mu=MU)
        res_hs = correlate(rx_data, ref_hs, mode='valid')
        res_abs = np.abs(res_hs)
        
        # Safe strict superposition
        corr_total[ant_idx] += res_abs
        
        # TOA Targeted Search Window (0.5ms)
        search_start = hs_idx * HALF_SUBFRAME_LEN
        search_end = min(search_start + HALF_SUBFRAME_LEN, result_len)
        
        if search_start < result_len:
            window_data = res_abs[search_start:search_end]
            if len(window_data) > 0:
                # 1. RELATIVE Peak Sample (Offset from the start of this specific block)
                peak_idx_relative = np.argmax(window_data)
                
                # 2. ABSOLUTE metrics (for accumulation/plotting)
                absolute_peak_idx = search_start + peak_idx_relative
                toa_sec_absolute = absolute_peak_idx / FS
                
                # 3. RELATIVE metrics (for Distance Calculation)
                toa_sec_relative = peak_idx_relative / FS
                distance_meters = toa_sec_relative * SPEED_OF_LIGHT
                
                toa_results[ant_idx].append(toa_sec_absolute)
                
                # Print first few results with the new formatting
                if hs_idx < 16:
                    print(f"Antenna {ant_labels[ant_idx]} | Block {hs_idx:3d} | Rel Sample: {peak_idx_relative:4d} | Rel TOA: {toa_sec_relative * 1e6:.2f} us | Distance: {distance_meters:.2f} m")

# ==========================================
# 5. VISUALIZATION
# ==========================================
t_ms = np.arange(result_len) / FS * 1000 
plt.figure(figsize=(15, 7))

colors = ['blue', 'red', 'green', 'purple']

for ant_idx in range(4):
    plt.plot(t_ms, corr_total[ant_idx], color=colors[ant_idx], 
             label=f'Antenna {ant_labels[ant_idx]} (nID {nIDs[ant_idx]})', 
             linewidth=1.2, alpha=0.7)

# Draw 10ms frame boundary grid lines
for i in range(12):
    plt.axvline(i * 10.0, color='black', linestyle='-', alpha=0.4)

plt.title("Pure Correlation Magnitude: 4-Channel TDM (0.5ms Half-Subframe Blocks)", fontsize=14)
plt.xlabel("Relative Time in Search Window (ms)", fontsize=12)
plt.ylabel("Superimposed Correlation Magnitude", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# Display the first 25ms 
plt.xlim(0, 4.5) 
plt.tight_layout()
plt.show()