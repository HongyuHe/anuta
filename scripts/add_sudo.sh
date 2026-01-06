#!/usr/bin/env bash

set -euo pipefail

# --- Check arguments ---
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <username>"
    exit 1
fi

USERNAME="$1"

# --- Must be run as root ---
if [[ "$EUID" -ne 0 ]]; then
    echo "This script must be run as root (use sudo)."
    exit 1
fi

# --- Create user if it does not exist ---
if id "$USERNAME" &>/dev/null; then
    echo "User '$USERNAME' already exists."
else
    echo "Creating user '$USERNAME'..."
    useradd -m -s /bin/bash "$USERNAME"
    passwd "$USERNAME"
fi

# --- Grant sudo privileges ---
echo "Adding '$USERNAME' to sudo group..."
usermod -aG sudo "$USERNAME"

# --- Enable password-based SSH login ---
SSHD_CONFIG="/etc/ssh/sshd_config"

echo "Configuring SSH to allow password authentication..."

# Backup sshd_config once
if [[ ! -f "${SSHD_CONFIG}.bak" ]]; then
    cp "$SSHD_CONFIG" "${SSHD_CONFIG}.bak"
fi

# Ensure settings are present and correct
sed -i \
    -e 's/^#\?PasswordAuthentication .*/PasswordAuthentication yes/' \
    -e 's/^#\?ChallengeResponseAuthentication .*/ChallengeResponseAuthentication no/' \
    -e 's/^#\?UsePAM .*/UsePAM yes/' \
    "$SSHD_CONFIG"

# --- Restart SSH safely ---
echo "Restarting SSH service..."
systemctl restart ssh

# --- Final output ---
echo "========================================"
echo "User '$USERNAME' created and configured:"
echo "  - Home directory: /home/$USERNAME"
echo "  - Shell: /bin/bash"
echo "  - Sudo access: enabled"
echo "  - SSH password login: enabled"
echo "========================================"
