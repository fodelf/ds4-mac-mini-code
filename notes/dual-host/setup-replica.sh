#!/bin/bash
# DS4 dual-host setup — run this ONCE on the MacBook (replica side).
#
# What it does:
#   1. Adds the Mac Mini's SSH public key to ~/.ssh/authorized_keys (idempotent).
#   2. Enables macOS Remote Login (SSH server) — needs sudo, will prompt locally.
#   3. Prints all info the host side needs to take over: username, IP addresses,
#      hostname, Thunderbolt link status, macOS version.
#
# Safety:
#   - No password leaves this machine; sudo prompt runs locally.
#   - Only the one ed25519 public key embedded below is granted access.
#   - To revoke later: edit ~/.ssh/authorized_keys and delete that one line.
#
# Usage:
#   chmod +x setup-replica.sh
#   ./setup-replica.sh
#
# (or just: bash setup-replica.sh)

set -u

HOST_PUBKEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJJoUjU5RR2rQIQu3uuHxz5TYmuUMx7h8dAD3dbntgpX 310066827@qq.com'

echo "================================================================"
echo "DS4 replica setup — installing Mac Mini's SSH key on this machine"
echo "================================================================"
echo

# --- Step 1: install authorized_keys ---------------------------------------
echo "[1/3] Installing host SSH public key into ~/.ssh/authorized_keys ..."
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

if grep -qxF "$HOST_PUBKEY" ~/.ssh/authorized_keys 2>/dev/null; then
    echo "      key already present, no change."
else
    echo "$HOST_PUBKEY" >> ~/.ssh/authorized_keys
    echo "      key installed."
fi
echo

# --- Step 2: enable Remote Login (SSH) -------------------------------------
echo "[2/3] Enabling macOS Remote Login (SSH server) ..."
echo "      sudo will prompt for YOUR LOCAL password on this MacBook."
echo "      Do not type your password into any chat — only into this terminal."
echo

# systemsetup needs sudo. On modern macOS it may also require Full Disk Access
# for Terminal in System Settings → Privacy. If it fails, the script falls back
# to printing manual instructions instead of dying silently.
if sudo systemsetup -setremotelogin on 2>/tmp/setremotelogin.err; then
    echo "      Remote Login enabled via systemsetup."
elif sudo launchctl load -w /System/Library/LaunchDaemons/ssh.plist 2>/dev/null; then
    echo "      Remote Login enabled via launchctl (fallback)."
else
    echo "      WARNING: could not enable Remote Login automatically."
    echo "      Open System Settings → General → Sharing → Remote Login and turn it ON."
    echo "      systemsetup stderr was:"
    sed 's/^/        /' /tmp/setremotelogin.err 2>/dev/null
fi
echo

# Verify sshd is actually listening.
if pgrep -x sshd >/dev/null || launchctl list 2>/dev/null | grep -q com.openssh.sshd; then
    echo "      sshd is running."
else
    echo "      NOTE: sshd not detected as running yet — give macOS ~5s then re-check."
fi
echo

# --- Step 3: print info the host side needs --------------------------------
echo "[3/3] Information for the host side"
echo "----------------------------------------------------------------"
echo "Username:           $USER"
echo "Hostname:           $(hostname)"
echo "Hostname (.local):  $(hostname -s).local"
echo "macOS version:      $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
echo "Architecture:       $(uname -m) / $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo '?')"
echo "Total RAM:          $(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.1f GiB", $1/1024/1024/1024}')"
echo
echo "All non-loopback IPv4 addresses:"
ifconfig 2>/dev/null | awk '
  /^[a-z]/ { iface = $1; sub(":", "", iface) }
  /inet / && $2 != "127.0.0.1" { printf "  %-10s %s\n", iface, $2 }
'
echo
echo "Thunderbolt-related interfaces (look for one with an IP above):"
networksetup -listallhardwareports 2>/dev/null | awk '
  /Hardware Port:.*[Tt]hunderbolt/ { port=$0; getline; iface=$0; print "  " port " → " iface; next }
  /Hardware Port:.*[Bb]ridge/      { port=$0; getline; iface=$0; print "  " port " → " iface }
'
echo
echo "Bridge interfaces (TB direct connect usually creates one):"
ifconfig 2>/dev/null | awk '/^bridge[0-9]+:/ {print "  " $0}' | head -5
echo

echo "================================================================"
echo "DONE."
echo
echo "Copy the username + IP lines above and paste them back to Claude on the"
echo "Mac Mini. Claude will take over from there (scp the binary, run the"
echo "smoke test, collect RTT, clean up)."
echo "================================================================"
