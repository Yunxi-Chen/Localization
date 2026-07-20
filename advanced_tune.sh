#!/bin/bash

# --- 1. System-wide Kernel Tuning (sysctl) ---
# Persistently optimizes network buffers and TCP stack behavior 
# by writing to /etc/sysctl.d/99-performance.conf
echo "--- Applying System Kernel Tuning ---"
cat <<EOF | sudo tee /etc/sysctl.d/99-performance.conf
# Increase TCP read/write buffer sizes for high-speed throughput
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# Increase maximum kernel network buffer limits
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.netdev_max_backlog = 5000

# Enable TCP features for high-bandwidth/latency efficiency
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_sack = 1
net.ipv4.tcp_timestamps = 1

# Increase max open files limit to prevent bottlenecks under high load
fs.file-max = 2097152
EOF

# Apply the new settings immediately
sudo sysctl --system
echo "Kernel parameters updated and persisted."

# --- 2. Automated IRQ Affinity Binding ---
# Dynamically detects high-speed interfaces and binds network interrupts 
# to a specific CPU core to reduce cache thrashing and latency.
echo "--- Applying IRQ Affinity ---"

# Identify all network interfaces starting with 'enp1s0'
interfaces=()
for iface in $(ls /sys/class/net); do
    if [[ "$iface" == enp1s0* ]]; then
        interfaces+=("$iface")
    fi
done

# Bind interrupts for each detected interface
for iface in "${interfaces[@]}"; do
    echo "Processing interface: $iface"
    
    # Extract IRQ numbers associated with the interface
    irqs=$(grep "$iface" /proc/interrupts | cut -d: -f1 | tr -d ' ')
    
    for irq in $irqs; do
        # Bind to CPU core 1 (bitmask 2 equals CPU 1)
        echo "Binding IRQ $irq to CPU core 1"
        echo 2 | sudo tee /proc/irq/"$irq"/smp_affinity > /dev/null
    done
done

echo "--- Advanced Optimization Complete ---"
