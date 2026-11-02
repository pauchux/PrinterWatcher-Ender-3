#!/usr/bin/env bash
# ==============================================================================
# TDR AI Discord Bot & TinyML Monitor - Setup & Installation Script
# Raspberry Pi 4B (Raspberry Pi OS 64-bit Debian Bookworm / Trixie)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${CYAN}${BOLD}"
echo "======================================================================"
echo "    🖨️  TDR TinyML Print Failure Monitor & Discord Bot Setup"
echo "        Ender 3 + Klipper + Moonraker + Raspberry Pi 4B"
echo "======================================================================"
echo -e "${RESET}"

# 1. System prerequisites check
echo -e "${BOLD}[1/5] Checking system prerequisites...${RESET}"
if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}❌ python3 is not installed. Please install Python 3.10+ first.${RESET}"
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "  Found Python: ${GREEN}${PY_VERSION}${RESET}"

# Check for apt packages on Debian/Raspberry Pi OS
MISSING_PKGS=()
for pkg in python3-venv python3-pip; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo -e "${YELLOW}  Missing recommended packages: ${MISSING_PKGS[*]}${RESET}"
    if [ -t 0 ] && command -v sudo >/dev/null 2>&1; then
        read -rp "  Install missing packages with apt? [Y/n] " install_apt
        if [[ "${install_apt:-y}" =~ ^[Yy]$ ]]; then
            echo "  Updating package index and installing ${MISSING_PKGS[*]}..."
            sudo apt-get update -qq && sudo apt-get install -y "${MISSING_PKGS[@]}"
        fi
    fi
fi

# Check for python3-picamera2 (system camera package on Pi)
if dpkg -s python3-picamera2 >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓ python3-picamera2 is installed (system-wide)${RESET}"
else
    echo -e "  ${YELLOW}ℹ python3-picamera2 not detected.${RESET}"
    echo "    (If using Crowsnest snapshot endpoint, this is optional. For raw hardware fallback: 'sudo apt install -y python3-picamera2')"
fi

# 2. Virtual environment setup
echo ""
echo -e "${BOLD}[2/5] Setting up virtual environment (with --system-site-packages)...${RESET}"
VENV_DIR="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
    echo -e "  ${GREEN}✓ Created virtual environment at $VENV_DIR${RESET}"
else
    echo -e "  ${GREEN}✓ Virtual environment already exists at $VENV_DIR${RESET}"
fi

VENV_PIP="$VENV_DIR/bin/pip"
VENV_PYTHON="$VENV_DIR/bin/python"

echo "  Upgrading pip, setuptools, wheel..."
"$VENV_PIP" install --quiet --upgrade pip setuptools wheel

# 3. Installing dependencies
echo ""
echo -e "${BOLD}[3/5] Installing lightweight Python dependencies...${RESET}"
"$VENV_PIP" install -r "$SCRIPT_DIR/requirements.txt"

# Verify TFLite runtime
echo -n "  Verifying TFLite / LiteRT interpreter runtime: "
RUNTIME_CHECK=$("$VENV_PYTHON" -c "
try:
    from ai_edge_litert.interpreter import Interpreter
    print('ai-edge-litert')
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        print('tflite-runtime')
    except ImportError:
        try:
            import tensorflow as tf
            print('tensorflow')
        except ImportError:
            print('none')
" 2>/dev/null || echo "none")

if [ "$RUNTIME_CHECK" = "none" ]; then
    echo -e "${YELLOW}None found yet.${RESET}"
    echo "  Attempting fallback install: tflite-runtime..."
    "$VENV_PIP" install tflite-runtime || true
else
    echo -e "${GREEN}✓ ${RUNTIME_CHECK}${RESET}"
fi

if [ -f "$SCRIPT_DIR/printWatcher_efficientnet_v2s.tflite" ] && [ -f "$SCRIPT_DIR/printer_failure_detector.tflite" ]; then
    echo -e "  Bundled TinyML models: ${GREEN}✓ Dual-Brain Ensemble (EfficientNetV2S + MobileNetV2 INT8)${RESET}"
elif [ -f "$SCRIPT_DIR/printWatcher_efficientnet_v2s.tflite" ]; then
    echo -e "  Bundled TinyML model:  ${GREEN}✓ printWatcher_efficientnet_v2s.tflite${RESET}"
elif [ -f "$SCRIPT_DIR/printer_failure_detector.tflite" ]; then
    echo -e "  Bundled TinyML model:  ${GREEN}✓ printer_failure_detector.tflite${RESET}"
else
    echo -e "  ${YELLOW}⚠ No .tflite model found in $SCRIPT_DIR${RESET}"
fi

# 4. Interactive Configuration (.env)
echo ""
echo -e "${BOLD}[4/5] Configuring environment (.env)...${RESET}"

ENV_FILE="$SCRIPT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    cp "$SCRIPT_DIR/.env.example" "$ENV_FILE"
    echo -e "  ${GREEN}✓ Created .env from .env.example${RESET}"
fi

CURRENT_TOKEN=$(grep -E '^DISCORD_BOT_TOKEN=' "$ENV_FILE" | cut -d '=' -f2- | tr -d ' ' || true)
CURRENT_USER_ID=$(grep -E '^DISCORD_USER_ID=' "$ENV_FILE" | cut -d '=' -f2- | tr -d ' ' || true)

if [ -t 0 ]; then
    # Interactive mode
    echo ""
    echo -e "  ${CYAN}Discord Credentials Setup:${RESET}"
    
    if [[ -z "$CURRENT_TOKEN" || "$CURRENT_TOKEN" == "paste-your-bot-token-here" ]]; then
        read -rp "  Enter DISCORD_BOT_TOKEN: " input_token
        if [ -n "$input_token" ]; then
            sed -i "s|^DISCORD_BOT_TOKEN=.*|DISCORD_BOT_TOKEN=${input_token}|" "$ENV_FILE"
            echo -e "  ${GREEN}✓ Token updated${RESET}"
        fi
    else
        echo -e "  DISCORD_BOT_TOKEN is already set (${CURRENT_TOKEN:0:6}...)."
        read -rp "  Would you like to change it? [y/N] " change_token
        if [[ "$change_token" =~ ^[Yy]$ ]]; then
            read -rp "  Enter new DISCORD_BOT_TOKEN: " input_token
            if [ -n "$input_token" ]; then
                sed -i "s|^DISCORD_BOT_TOKEN=.*|DISCORD_BOT_TOKEN=${input_token}|" "$ENV_FILE"
                echo -e "  ${GREEN}✓ Token updated${RESET}"
            fi
        fi
    fi

    if [[ -z "$CURRENT_USER_ID" || "$CURRENT_USER_ID" == "123456789012345678" ]]; then
        read -rp "  Enter DISCORD_USER_ID (your personal Discord User ID for DMs): " input_uid
        if [ -n "$input_uid" ]; then
            sed -i "s|^DISCORD_USER_ID=.*|DISCORD_USER_ID=${input_uid}|" "$ENV_FILE"
            echo -e "  ${GREEN}✓ User ID updated${RESET}"
        fi
    else
        echo -e "  DISCORD_USER_ID is already set ($CURRENT_USER_ID)."
        read -rp "  Would you like to change it? [y/N] " change_uid
        if [[ "$change_uid" =~ ^[Yy]$ ]]; then
            read -rp "  Enter new DISCORD_USER_ID: " input_uid
            if [ -n "$input_uid" ]; then
                sed -i "s|^DISCORD_USER_ID=.*|DISCORD_USER_ID=${input_uid}|" "$ENV_FILE"
                echo -e "  ${GREEN}✓ User ID updated${RESET}"
            fi
        fi
    fi
else
    echo -e "  ${YELLOW}Non-interactive shell detected. Please ensure $ENV_FILE contains valid credentials.${RESET}"
fi

# 5. Systemd Service Installation
echo ""
echo -e "${BOLD}[5/5] Systemd service configuration...${RESET}"
TARGET_SERVICE="/etc/systemd/system/tdr-ai-bot.service"
SERVICE_TEMPLATE="$SCRIPT_DIR/tdr-ai-bot.service"

INSTALL_SERVICE="n"
if [ -t 0 ]; then
    read -rp "  Do you want to install tdr-ai-bot as a systemd service (auto-start on boot)? [Y/n] " resp
    INSTALL_SERVICE="${resp:-y}"
fi

if [[ "$INSTALL_SERVICE" =~ ^[Yy]$ ]]; then
    CURRENT_USER=$(id -un)
    echo "  Generating unit file for user: $CURRENT_USER, path: $SCRIPT_DIR..."
    
    TEMP_SERVICE=$(mktemp)
    sed -e "s|__USER__|$CURRENT_USER|g" \
        -e "s|__WORKDIR__|$SCRIPT_DIR|g" \
        -e "s|__PYTHON__|$VENV_PYTHON|g" \
        "$SERVICE_TEMPLATE" > "$TEMP_SERVICE"
    
    if command -v sudo >/dev/null 2>&1; then
        sudo cp "$TEMP_SERVICE" "$TARGET_SERVICE"
        sudo chmod 644 "$TARGET_SERVICE"
        sudo systemctl daemon-reload
        echo -e "  ${GREEN}✓ Installed service to $TARGET_SERVICE${RESET}"
        
        read -rp "  Enable and start tdr-ai-bot.service now? [Y/n] " start_resp
        if [[ "${start_resp:-y}" =~ ^[Yy]$ ]]; then
            sudo systemctl enable tdr-ai-bot.service
            sudo systemctl restart tdr-ai-bot.service
            echo -e "  ${GREEN}✓ Service enabled and started!${RESET}"
            echo -e "  Check status anytime with: ${CYAN}sudo systemctl status tdr-ai-bot.service${RESET}"
            echo -e "  View logs anytime with:   ${CYAN}journalctl -u tdr-ai-bot.service -f${RESET}"
        fi
    else
        echo -e "${YELLOW}  sudo not available. You can manually copy the unit file to /etc/systemd/system/:${RESET}"
        echo "  sudo cp $TEMP_SERVICE /etc/systemd/system/tdr-ai-bot.service"
        echo "  sudo systemctl daemon-reload && sudo systemctl enable --now tdr-ai-bot.service"
    fi
    rm -f "$TEMP_SERVICE"
else
    echo "  Skipping systemd installation."
    echo -e "  To start the bot manually:"
    echo -e "    ${CYAN}source $VENV_DIR/bin/activate && python discord_controller.py${RESET}"
fi

echo ""
echo -e "${GREEN}${BOLD}======================================================================"
echo "    🎉 Setup Complete!"
echo "======================================================================${RESET}"
echo -e "  1. Verify settings in: ${CYAN}$ENV_FILE${RESET}"
echo -e "  2. Test manually:      ${CYAN}$VENV_DIR/bin/python $SCRIPT_DIR/discord_controller.py${RESET}"
echo -e "  3. Service commands:   ${CYAN}sudo systemctl status tdr-ai-bot${RESET}"
echo -e "                         ${CYAN}journalctl -u tdr-ai-bot -f${RESET}"
echo -e "${GREEN}======================================================================${RESET}"
