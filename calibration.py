import numpy as np
from scipy.fft import fft, ifft, ifftshift
from scipy.signal import correlate

# ==========================================
# 1. 5G PRS WAVEFORM GENERATION FUNCTIONS
# ==========================================
def generate_prs_sequence(n_res, c_init):
    """
    Generates 3GPP TS 38.211 compliant Gold sequence for PRS.
    """
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
    """
    Maps PRS sequence onto Comb-2 subcarriers, applies IFFT and prepends Cyclic Prefix (CP).
    """
    fft_grid = np.zeros(n_fft, dtype=complex)
    start = (n_fft - n_res) // 2
    fft_grid[start : start + n_res : 2] = prs_seq 
    time_domain_symbol = ifft(ifftshift(fft_grid)) * np.sqrt(n_fft)
    cp = time_domain_symbol[-cp_length:]
    return np.concatenate((cp, time_domain_symbol))

def get_local_ref_half_subframe(n_id, hs_idx, n_fft, cp_first, cp_normal, n_res, mu=2):
    """
    Synthesizes a 0.5ms half-subframe reference waveform based on 3GPP slot indexing rules.
    """
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
            
            # 3GPP Rule: First symbol in 0.5ms block uses long CP
            if s_offset == 0 and l == 0:
                current_cp = cp_first
            else:
                current_cp = cp_normal
                
            ofdm_sym = modulate_and_add_cp(prs_seq, n_fft, current_cp, n_res)
            hs_waveform = np.concatenate((hs_waveform, ofdm_sym))
            
    return hs_waveform

def global_zero_padding(signal, factor=10):
    """
    Applies exact frequency-domain zero-padding to the complex correlation signal.
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
# 2. CONFIGURATION & SYSTEM PARAMETERS
# ==========================================
FS = 122.88e6                           # Sampling frequency: 122.88 MHz
HALF_SUBFRAME_LEN = 61440               # Samples per 0.5ms half-subframe
N_FFT, N_RES, N_CP_F, N_CP_N = 2048, 1620, 208, 144
MU = 2                                  # Subcarrier spacing parameter (60 kHz SCS)
K_OS = 10                               # 10x Oversampling factor
SPEED_OF_LIGHT = 299792458.0            # Speed of light in vacuum (m/s)

N_ID_B = 200                            # Antenna B ID
BLOCKS = [1, 5]                         # Antenna B TDM assigned half-subframe slots
TARGET_BLOCK_COUNT = 100                # Limit processing to the first 100 blocks
SEARCH_RANGE = 1000                     # Restrict peak search window to first 1000 samples

# Core Fix: 5G NR frame period is 10ms = 20 half-subframes
TX_PERIOD_HS = 20

# ==========================================
# 3. DATA LOADING VIA MEMORY MAP
# ==========================================
skip_samples = int(FS * 1)             # Skip 1 second of initial RX noise
rx_memmap = np.memmap('/home/antlab/Desktop/10m_calibration.bin', dtype=np.complex64, mode='r', offset=skip_samples * 8)

# ==========================================
# 4. PROCESSING LOOP WITH DYNAMIC SLOT MATCHING
# ==========================================
raw_peaks_hist, raw_toas_hist, raw_dists_hist = [], [], []
fine_peaks_hist, fine_toas_hist, fine_dists_hist = [], [], []

block_count = 0
cycle = 0

print(f"\nProcessing first {TARGET_BLOCK_COUNT} Blocks for Antenna B (TX_PERIOD_HS = 20)...\n")

while block_count < TARGET_BLOCK_COUNT:
    for hs_idx in BLOCKS:
        if block_count >= TARGET_BLOCK_COUNT:
            break
            
        abs_hs_idx = (cycle * 8) + hs_idx
        search_start_abs = abs_hs_idx * HALF_SUBFRAME_LEN
        search_end_abs = search_start_abs + (2 * HALF_SUBFRAME_LEN) - 1
        
        if search_end_abs > len(rx_memmap):
            print("Warning: Reached end of binary file early!")
            break
            
        rx_chunk = rx_memmap[search_start_abs:search_end_abs]
        
        # Calculate slot reference index modulo 20
        hs_in_tx_period = abs_hs_idx % TX_PERIOD_HS
        ref_hs = get_local_ref_half_subframe(N_ID_B, hs_in_tx_period, N_FFT, N_CP_F, N_CP_N, N_RES, mu=MU)
        
        # --- 1. RAW 1x Grid Correlation ---
        res_hs = correlate(rx_chunk, ref_hs, mode='valid')
        window_data_raw = np.abs(res_hs)[:SEARCH_RANGE]
        
        raw_peak_offset = np.argmax(window_data_raw)
        toa_sec_raw = raw_peak_offset / FS
        dist_raw = toa_sec_raw * SPEED_OF_LIGHT
        
        # --- 2. 10x Oversampled Grid Correlation ---
        corr_os_complex = global_zero_padding(res_hs[:SEARCH_RANGE], factor=K_OS)
        window_data_os = np.abs(corr_os_complex)
        
        os_peak_offset = np.argmax(window_data_os)
        fine_peak_offset = os_peak_offset / float(K_OS)
        toa_sec_fine = fine_peak_offset / FS
        dist_fine = toa_sec_fine * SPEED_OF_LIGHT
        
        # Record statistical metrics
        raw_peaks_hist.append(raw_peak_offset)
        raw_toas_hist.append(toa_sec_raw)
        raw_dists_hist.append(dist_raw)
        
        fine_peaks_hist.append(fine_peak_offset)
        fine_toas_hist.append(toa_sec_fine)
        fine_dists_hist.append(dist_fine)
        
        block_count += 1
        
        print(f"[{block_count:3d}/100] Antenna B | Block {hs_idx:3d} (Slot Ref: {hs_in_tx_period:2d}) | [RAW ] Rel Peak: {float(raw_peak_offset):6.2f} | Rel TOA: {toa_sec_raw * 1e6:6.4f} us | Distance: {dist_raw:7.3f} m")
        print(f"[{block_count:3d}/100] Antenna B | Block {hs_idx:3d} (Slot Ref: {hs_in_tx_period:2d}) | [FINE] Rel Peak: {fine_peak_offset:6.2f} | Rel TOA: {toa_sec_fine * 1e6:6.4f} us | Distance: {dist_fine:7.3f} m")
        print("-" * 85)
        
    cycle += 1
    if search_end_abs > len(rx_memmap):
        break

# ==========================================
# 5. STATISTICAL AVERAGES
# ==========================================
print("\n" + "="*85)
print(f"STATISTICAL AVERAGES FOR ANTENNA B OVER {len(raw_peaks_hist)} BLOCKS")
print("="*85)

print(f"  [RAW  AVERAGE] Rel Peak Sample: {np.mean(raw_peaks_hist):6.2f} | Rel TOA: {np.mean(raw_toas_hist) * 1e6:6.4f} us | Distance: {np.mean(raw_dists_hist):7.3f} m")
print(f"  [FINE AVERAGE] Rel Peak Sample: {np.mean(fine_peaks_hist):6.2f} | Rel TOA: {np.mean(fine_toas_hist) * 1e6:6.4f} us | Distance: {np.mean(fine_dists_hist):7.3f} m")
print("="*85)