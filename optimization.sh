#!/bin/bash

# --- CPU Governor Tuning ---
echo -e "Would you like to set the CPU governor to performance?\n1 - Performance\n2 - Powersave"
read yn
case $yn in
    1)
        for ((i=0; i<$(nproc --all); i++)); do
            sudo cpufreq-set -c $i -r -g performance
        done
        ;;
    2)
        for ((i=0; i<$(nproc --all); i++)); do
            sudo cpufreq-set -c $i -r -g powersave
        done
        ;;
esac

# --- Network Buffer Tuning ---
read -p "Would you like to adjust the network buffers? (y/n) " yn
case $yn in
    y)
        echo -e "Choose buffer values:\n1 - 62500000\n2 - 33554432" 
        read buffer_values
        case $buffer_values in 
            1)
                sudo sysctl -w net.core.wmem_max=62500000
                sudo sysctl -w net.core.rmem_max=62500000
                sudo sysctl -w net.core.wmem_default=62500000
                sudo sysctl -w net.core.rmem_default=62500000
                ;;
            2)
                sudo sysctl -w net.core.wmem_max=33554432
                sudo sysctl -w net.core.rmem_max=33554432
                sudo sysctl -w net.core.wmem_default=33554432
                sudo sysctl -w net.core.rmem_default=33554432
                ;;
        esac
        ;;
esac

# --- Ulimit Tuning ---
read -p "Would you like to set ulimit? (y/n) " yn
case $yn in
    y)
        ulimit -n 65536
        ulimit -l unlimited
        ulimit -r 99
        echo "Ulimit settings applied for current session."
        ;;
esac

# --- Interface Tuning (enp1s0*) ---
interfaces=()
for iface in $(ls /sys/class/net); do
    if [[ "$iface" == enp1s0* ]]; then
        interfaces+=("$iface")
    fi
done

echo "Detected 10Gb Ethernet interfaces: ${interfaces[@]}"

for iface in "${interfaces[@]}"; do
    read -p "Would you like to set the tx and rx ring buffer sizes for interface $iface? (y/n) " yn
    case $yn in
        y) sudo ethtool -G "$iface" tx 4096 rx 4096 ;;
    esac
    
    read -p "Would you like to set the MTU size to 9000 for interface $iface? (y/n) " yn
    case $yn in
        y) sudo ifconfig "$iface" mtu 9000 up ;;
    esac
done
