import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks

# ==========================================
# 1. CONFIGURATION & CALIBRATION
# ==========================================
sample_rate_hz = 122.88e6  
# EXACT defined speed of light in vacuum (299,792,458 m/s)
speed_of_light = 299792458    

# Apply individual calibration offsets to account for the 5m RF extension cables on AP2 and AP3
# Base offset: 205. Cable delay: 3 samples.
CALIBRATION_OFFSETS = {
    100: 205,      # AP1 (Master)
    200: 205 + 3,  # AP2 (with 5m cable delay compensation)
    300: 205 + 3,  # AP3 (with 5m cable delay compensation)
    400: 205       # AP4
}

# --- RX EARLY START HANDLING ---
RX_EARLY_START_SEC = 1.0
SKIP_SAMPLES = int(RX_EARLY_START_SEC * sample_rate_hz)

# File paths for 4 transmitters
tx_paths = {
    100: "/home/antlab/Desktop/ch0_mu=2.bin",
    200: "/home/antlab/Desktop/ch1_mu=2.bin",
    300: "/home/antlab/Desktop/ch2_mu=2.bin",
    400: "/home/antlab/Desktop/ch3_mu=2.bin"
}
rx_path = "/dev/shm/rx_wave_mu_2.bin" 

# ==========================================
# 2. DATA LOADING
# ==========================================
print("Loading 4-channel reference binary files...")
tx_signals = {nid: np.fromfile(path, dtype=np.complex64) for nid, path in tx_paths.items()}

# Calculate search window
search_len = max([len(sig) for sig in tx_signals.values()]) + int(sample_rate_hz * 0.1)

print(f"Skipping first {RX_EARLY_START_SEC}s ({SKIP_SAMPLES} samples) of RX noise...")
search_window = np.fromfile(rx_path, dtype=np.complex64, offset=SKIP_SAMPLES * 8, count=search_len)

# ==========================================
# 3. CORRELATION & PEAK DETECTION FUNCTION
# ==========================================
# Added 'nid' to the function parameters to fetch the correct offset
def extract_toa_and_cir(tx_sig, rx_win, nid, name):
    print(f"\nProcessing {name}...")
    
    correlation = correlate(rx_win, tx_sig, mode='valid', method='fft')
    magnitude = np.abs(correlation)

    max_val = np.max(magnitude)
    peaks, _ = find_peaks(magnitude, height=max_val * 0.5, distance=100)

    if len(peaks) == 0:
        print(f"ERROR: No peaks found for {name}!")
        return None, None

    raw_peak_idx = peaks[0]
    print(f"  => Found {len(peaks)} repeating frames. Locking onto FIRST arrival at index {raw_peak_idx}.")
    
    # Retrieve the specific offset for this antenna ID
    current_offset = CALIBRATION_OFFSETS[nid]
    
    # Mathematical deduction using exact speed of light and the specific calibration offset
    true_delay_samples = raw_peak_idx - current_offset
    toa_seconds = true_delay_samples / sample_rate_hz
    distance_meters = toa_seconds * speed_of_light

    # Detailed terminal reporting structure
    print(f"  Raw Peak Index        : {raw_peak_idx}")
    print(f"  Calibration Offset    : -{current_offset}")
    print(f"  True OTA Delay        : {true_delay_samples} samples")
    print(f"  Time of Arrival (ToA) : {toa_seconds:.9f} seconds")
    print(f"  ABSOLUTE DISTANCE     : {distance_meters:.4f} Meters")

    # Extract CIR array
    cir_mag = magnitude[raw_peak_idx - 10 : raw_peak_idx + 100]
    cir_norm = cir_mag / np.max(cir_mag)
    cir_db = 20 * np.log10(cir_norm + 1e-10)
    
    return cir_db, distance_meters

# ==========================================
# 4. EXECUTE FOR ALL 4 gNBs
# ==========================================
print("\n" + "="*60)
results = {}
for nid in [100, 200, 300, 400]:
    # Pass the antenna ID (nid) into the function
    results[nid] = extract_toa_and_cir(tx_signals[nid], search_window, nid, f"Antenna ID {nid}")
print("="*60 + "\n")

# ==========================================
# 5. VISUALIZATION
# ==========================================
delay_axis_samples = np.arange(-10, 100)
distance_axis_meters = (delay_axis_samples / sample_rate_hz) * speed_of_light

plt.figure(figsize=(14, 7))
colors = {100: 'blue', 200: 'red', 300: 'green', 400: 'orange'}

for nid, (cir_db, dist) in results.items():
    if cir_db is not None:
        plt.plot(distance_axis_meters, cir_db, color=colors[nid], linewidth=1.5, 
                 label=f'ID {nid} (Dist: {dist:.4f}m)')

plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Detected LoS')
plt.xlim([-10, 80]) 
plt.ylim([-60, 5]) 

plt.title("4-Channel Channel Impulse Response (CIR) - dB Normalized")
plt.xlabel("Relative Multipath Distance (Meters)")
plt.ylabel("Normalized Magnitude (dB)")
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()