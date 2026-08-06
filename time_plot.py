import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import ifft, ifftshift
from scipy.signal import correlate, find_peaks

# ==========================================
# 1. 5G PRS WAVEFORM GENERATION FUNCTIONS
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
            
            if s_offset == 0 and l == 0:
                current_cp = cp_first
            else:
                current_cp = cp_normal
                
            ofdm_sym = modulate_and_add_cp(prs_seq, n_fft, current_cp, n_res)
            hs_waveform = np.concatenate((hs_waveform, ofdm_sym))
            
    return hs_waveform

# ==========================================
# 2. CONFIGURATION
# ==========================================
FILE_PATH = '/dev/shm/rx_wave_mu_2.bin' 
FS = 122.88e6  
DURATION = 10.0  

N_FFT = 2048
N_RES = 1620
N_CP_F = 208
N_CP_N = 144
MU = 2

# Antenna assignments (distinct slots per antenna in a 20-slot frame)
HS_ASSIGNMENTS = {
    100: [0, 4, 8, 12, 16],  # Antenna A
    200: [1, 5, 9, 13, 17],  # Antenna B
    300: [2, 6, 10, 14, 18], # Antenna C
    400: [3, 7, 11, 15, 19]  # Antenna D
}

# ==========================================
# 3. DATA LOADING & DYNAMIC MULTI-SLOT TEMPLATES
# ==========================================
num_samples = int(FS * DURATION)
print(f"Loading {DURATION} seconds of data from {FILE_PATH}...")
if not os.path.exists(FILE_PATH):
    print(f"ERROR: File {FILE_PATH} not found!")
    exit()

rx_data = np.fromfile(FILE_PATH, dtype=np.complex64, count=num_samples)

print("Generating 5G PRS Tx Reference Signals dynamically for all active slots...")
# Structure: tx_signals[nid][hs_idx] = complex_waveform
tx_signals = {100: {}, 200: {}, 300: {}, 400: {}}

for nid, slots in HS_ASSIGNMENTS.items():
    for hs in slots:
        tx_signals[nid][hs] = generate_reference_signal(
            n_id=nid, hs_idx=hs, n_fft=N_FFT, cp_first=N_CP_F, cp_normal=N_CP_N, n_res=N_RES, mu=MU
        )
    print(f"  -> Generated {len(slots)} reference templates for Antenna ID {nid}")

# ==========================================
# 4. ENVELOPE DETECTION (Macro View)
# ==========================================
print("Calculating macroscopic magnitude envelope...")
chunk_size = int(FS * 0.001) 
num_chunks = len(rx_data) // chunk_size
reshaped_data = rx_data[:num_chunks * chunk_size].reshape(num_chunks, chunk_size)
envelope = np.max(np.abs(reshaped_data), axis=1)
t_env = np.arange(num_chunks) * 0.001 

# ==========================================
# 5. 4-CHANNEL CORRELATION (Multi-Slot Superposition)
# ==========================================
print("Extracting microscopic 20ms view and running cross-correlation...")
start_time_sec = 1.0
start_index = int(start_time_sec * FS)
micro_samples = int(FS * 0.02) # First 20ms
rx_window = rx_data[start_index : start_index + micro_samples]

results = {}
for nid, slot_templates in tx_signals.items():
    print(f"  Correlating Antenna {nid} across all {len(slot_templates)} slots...")
    
    combined_mag = None
    
    # Correlate the 20ms Rx window against EVERY active slot template for this antenna
    for hs, tx_sig in slot_templates.items():
        corr = correlate(rx_window, tx_sig, mode='valid', method='fft')
        mag = np.abs(corr)
        
        # Superimpose the correlation magnitudes by taking the maximum across all slot templates
        if combined_mag is None:
            combined_mag = mag
        else:
            combined_mag = np.maximum(combined_mag, mag)
            
    # Run peak detection on the superimposed magnitude array
    max_val = np.max(combined_mag)
    peaks, _ = find_peaks(combined_mag, height=max_val * 0.5, distance=100)
    t_corr = np.arange(len(combined_mag)) / FS * 1000
    
    results[nid] = {
        'mag': combined_mag,
        'peaks': peaks,
        't_corr': t_corr,
        'threshold': max_val * 0.5
    }

# ==========================================
# 6. VISUALIZATION
# ==========================================
plt.figure(figsize=(15, 10))

# --- Plot 1: Macroscopic 10-Second Envelope ---
plt.subplot(2, 1, 1)
plt.plot(t_env, envelope, color='darkblue', linewidth=1)
plt.title("Macroscopic View: 10-Second Signal Envelope (1ms Resolution)", fontsize=14, fontweight='bold')
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Peak Magnitude", fontsize=12)
plt.grid(True, alpha=0.4)
plt.xlim(0, DURATION)

# --- Plot 2: Microscopic 20ms 4-Channel Peaks ---
plt.subplot(2, 1, 2)
colors = {100: 'blue', 200: 'red', 300: 'green', 400: 'orange'}

for nid, res in results.items():
    t_c = res['t_corr']
    mag = res['mag']
    peaks = res['peaks']
    color = colors.get(nid, 'black')
    
    # Plot the superimposed multi-slot correlation magnitude line
    plt.plot(t_c, mag, color=color, linewidth=1.2, alpha=0.8, label=f'Antenna {nid//100} (ID {nid})')
    
    # Mark the peaks
    plt.plot(t_c[peaks], mag[peaks], marker='*', color=color, linestyle='None', 
             markersize=10, markeredgecolor='black', zorder=5)

plt.title("4-Channel Correlation Magnitude & Peak Detection (Multi-Slot Superposition)", fontsize=14, fontweight='bold')
plt.xlabel("Relative Time in Search Window (ms)", fontsize=12)
plt.ylabel("Superimposed Correlation Magnitude", fontsize=12)
plt.grid(True, alpha=0.4)
plt.xlim(0, 20.0) 
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

print("Plotting complete!")