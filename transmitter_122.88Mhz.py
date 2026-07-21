import numpy as np
from scipy.fft import ifft, ifftshift

# STEP 1: Gold Sequence Generator (The "Barcode" Maker)
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
    
    c_even = c[0::2]
    c_odd  = c[1::2]
    
    i_data = (1 - 2 * c_even) / np.sqrt(2)
    q_data = (1 - 2 * c_odd) / np.sqrt(2)
    
    return i_data + 1j * q_data

# STEP 2 & 3: OFDM Modulator (The Frequency-to-Time Converter)
def modulate_and_add_cp(prs_seq, n_fft, cp_length, n_res):
    fft_grid = np.zeros(n_fft, dtype=complex)
    
    start_idx = (n_fft - n_res) // 2
    end_idx = start_idx + n_res
    fft_grid[start_idx:end_idx:2] = prs_seq # Comb-2 mapping
    
    time_domain_symbol = ifft(ifftshift(fft_grid)) * np.sqrt(n_fft)
    
    # Dynamic CP insertion
    cp = time_domain_symbol[-cp_length:]
    return np.concatenate((cp, time_domain_symbol))

# STEP 4: 3GPP Compliant Frame Assembler (CORRECTED)
def generate_compliant_5g_frame(n_id, n_sym_prs, n_fft, cp_len_first, cp_len_normal, n_res, slots_per_half_subframe):
    frame_waveform = np.array([], dtype=complex)
    n_slots = n_sym_prs // 14 
    
    for slot_idx in range(n_slots):
        for l in range(14):
            # Calculate the unique 3GPP seed for THIS exact symbol
            part1 = (2**22) * (n_id // 1024)
            part2 = (2**10) * (14 * slot_idx + l + 1) * (2 * (n_id % 1024) + 1)
            part3 = (n_id % 1024)
            c_init_true = (part1 + part2 + part3) % (2**31)
            
            # Generate the sequence
            prs_seq = generate_prs_sequence(n_res // 2, c_init=c_init_true)
            
            # CORRECTED 3GPP CP RULE: 
            # The long CP is only applied to l=0 IF the slot is the start of a 0.5ms half-subframe
            if l == 0 and (slot_idx % slots_per_half_subframe == 0):
                current_cp = cp_len_first
            else:
                current_cp = cp_len_normal
            
            # Modulate and append to the long wave
            ofdm_sym = modulate_and_add_cp(prs_seq, n_fft, current_cp, n_res)
            frame_waveform = np.concatenate((frame_waveform, ofdm_sym))
            
    return frame_waveform

# STEP 5: File Exporter (Normalization and repeating ONLY)
def format_and_save_for_sdr(waveform_1d, filename, repeat_count=100):
    # 1. Normalization
    max_amp = np.max(np.abs(waveform_1d))
    
    if max_amp > 0:
        waveform_1d = (waveform_1d / max_amp) * 0.95
        print(f"   [!] Signal normalized. Original peak: {max_amp:.4f} -> Target peak: 0.95")
    
    # 2. Repeat the frame to extend file length
    print(f"   [+] Repeating the frame {repeat_count} times to generate base 1-second data...")
    long_waveform = np.tile(waveform_1d, repeat_count)
    
    # 3. Convert to Complex64 and save
    signal_out = long_waveform.astype(np.complex64)
    signal_out.tofile(filename)
    return len(signal_out)


if __name__ == "__main__":
    # --- Configuration Options ---
    MU = 2 
    
    if MU == 1:
        print(">> Configuring for mu=1 (30 kHz SCS, 10ms frame)")
        # WARNING: N_FFT=2048 at 30kHz SCS equals 61.44 MHz sample rate, NOT 122.88 MHz!
        # If you want 122.88 MHz for mu=1, N_FFT must be 4096 (CPs must be doubled).
        N_FFT = 2048
        N_RES = 1596          
        N_CP_FIRST = 176      
        N_CP = 144            
        N_SYM_PRS = 280         
        SLOTS_PER_HALF_SUBFRAME = 1 # In mu=1, 0.5ms is exactly 1 slot
        
    elif MU == 2:
        print(">> Configuring for mu=2 (60 kHz SCS, 10ms frame)")
        N_FFT = 2048
        N_RES = 1620          
        N_CP_FIRST = 208      
        N_CP = 144            
        N_SYM_PRS = 560         
        SLOTS_PER_HALF_SUBFRAME = 2 # In mu=2, 0.5ms is exactly 2 slots
        
    else:
        raise ValueError("This script only supports MU=1 or MU=2")

    antenna_configs = [
        {"n_id": 100, "filename": f"my_5g_prs_mu{MU}_nID100_1s_122.88M.bin"},
        {"n_id": 200, "filename": f"my_5g_prs_mu{MU}_nID200_1s_122.88M.bin"},
        {"n_id": 300, "filename": f"my_5g_prs_mu{MU}_nID300_1s_122.88M.bin"},
        {"n_id": 400, "filename": f"my_5g_prs_mu{MU}_nID400_1s_122.88M.bin"}
    ]

    for config in antenna_configs:
        print(f"\n-> Building 3GPP frame (10ms) for nID = {config['n_id']}...")
        
        # Pass the subframe timing parameter
        tx_waveform_base = generate_compliant_5g_frame(
            n_id=config["n_id"], 
            n_sym_prs=N_SYM_PRS, 
            n_fft=N_FFT, 
            cp_len_first=N_CP_FIRST, 
            cp_len_normal=N_CP, 
            n_res=N_RES,
            slots_per_half_subframe=SLOTS_PER_HALF_SUBFRAME
        )
        
        print(f"-> Processing and saving to {config['filename']}...")
        total_samples = format_and_save_for_sdr(
            tx_waveform_base, 
            config["filename"], 
            repeat_count=100
        )
        
        print(f"   [DONE] Generated {total_samples} samples.")