#!/usr/bin/env bash
# ==============================================================================
# TDR AI Model & Bot Updater (Run directly on Raspberry Pi 4B)
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
echo "    🚀  TDR Raspberry Pi Model Updater: printWatcher_efficientnet_v2s"
echo "======================================================================"
echo -e "${RESET}"

MODEL_NAME="printWatcher_efficientnet_v2s.tflite"
TARGET_MODEL="$SCRIPT_DIR/$MODEL_NAME"

# 1. Check if model file exists or search in common upload paths
echo -e "${BOLD}[1/4] Checking for $MODEL_NAME...${RESET}"
if [ ! -f "$TARGET_MODEL" ]; then
    echo -e "  ${YELLOW}Model not found in $SCRIPT_DIR. Searching common directories...${RESET}"
    FOUND_PATH=""
    for search_dir in "$HOME/Downloads" "$HOME" "/tmp" "$SCRIPT_DIR/.." "$HOME/printer_data/config"; do
        if [ -f "$search_dir/$MODEL_NAME" ]; then
            FOUND_PATH="$search_dir/$MODEL_NAME"
            break
        fi
    done

    if [ -n "$FOUND_PATH" ]; then
        echo -e "  ${GREEN}✓ Found model at: $FOUND_PATH${RESET}"
        echo "  Copying to $TARGET_MODEL..."
        cp "$FOUND_PATH" "$TARGET_MODEL"
    else
        echo -e "${RED}❌ Could not find $MODEL_NAME in $SCRIPT_DIR or common folders.${RESET}"
        echo -e "   Please upload it to the Raspberry Pi using SCP or SFTP:"
        echo -e "   ${CYAN}scp $MODEL_NAME $(id -un)@$(hostname -I | awk '{print $1}'):$SCRIPT_DIR/${RESET}"
        exit 1
    fi
fi

MODEL_SIZE_MB=$(du -m "$TARGET_MODEL" | cut -f1)
echo -e "  ${GREEN}✓ Model present: $TARGET_MODEL (${MODEL_SIZE_MB} MB)${RESET}"

# 2. Virtual environment and dependencies check
echo ""
echo -e "${BOLD}[2/4] Verifying Python environment & model integrity...${RESET}"
VENV_DIR="$SCRIPT_DIR/venv"
if [ -f "$VENV_DIR/bin/python" ]; then
    VENV_PYTHON="$VENV_DIR/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    VENV_PYTHON="python3"
else
    echo -e "${RED}❌ python3 not found.${RESET}"
    exit 1
fi

if [ -f "$SCRIPT_DIR/test_model.py" ]; then
    echo "  Running quick verification with test_model.py..."
    if "$VENV_PYTHON" "$SCRIPT_DIR/test_model.py" "$TARGET_MODEL" --runs 3; then
        echo -e "  ${GREEN}✓ Model passed integrity and inference tests!${RESET}"
    else
        echo -e "${RED}❌ Model failed verification test. Check output above.${RESET}"
        exit 1
    fi
else
    echo -e "  ${YELLOW}ℹ test_model.py not found, skipping dry-run.${RESET}"
fi

# 3. Update persistent active model state (and migrate v1/v2 settings -> v3)
echo ""
echo -e "${BOLD}[3/4] Updating bot_state.json (Dual-Brain Ensemble, 60% threshold, BLTouch-safe ROI & 4/6 window)...${RESET}"
if [ -f "$SCRIPT_DIR/.env" ]; then
    sed -i 's/^CONFIDENCE_THRESHOLD=0\.85$/CONFIDENCE_THRESHOLD=0.60/' "$SCRIPT_DIR/.env" || true
    sed -i 's/^CONSECUTIVE_FRAMES=3$/CONSECUTIVE_FRAMES=4/' "$SCRIPT_DIR/.env" || true
fi
STATE_FILE="$SCRIPT_DIR/bot_state.json"
"$VENV_PYTHON" -c "
import json
from pathlib import Path
p = Path('$STATE_FILE')
script_dir = Path('$SCRIPT_DIR')
try:
    data = json.loads(p.read_text()) if p.is_file() else {}
except Exception:
    data = {}
data.setdefault('enabled', True)
data.setdefault('mode', 'supervised')
data.setdefault('supervised_timeout', 90)
if data.get('schema_version', 1) < 2 or float(data.get('threshold', 0.85)) >= 0.78:
    data['threshold'] = 0.60
if data.get('schema_version', 1) < 3 or int(data.get('frames', 3)) < 4:
    data['frames'] = 4
data['roi_mode'] = 'roi'
data['roi_box'] = '0.25,0.32,0.86,0.96'
data['schema_version'] = 3
has_mob = (script_dir / 'printer_failure_detector.tflite').is_file()
has_eff = (script_dir / '$MODEL_NAME').is_file()
data['active_model'] = 'ensemble' if (has_mob and has_eff) else ('printer_failure_detector.tflite' if has_mob else '$MODEL_NAME')
p.write_text(json.dumps(data, indent=2))
print(f\"  ✓ Updated bot_state.json -> model={data['active_model']}, threshold={data['threshold']}, frames={data['frames']}/6, roi={data['roi_mode']} ({data['roi_box']})\")
"

# 4. Restart systemd service
echo ""
echo -e "${BOLD}[4/4] Restarting tdr-ai-bot.service...${RESET}"
SERVICE_NAME="tdr-ai-bot.service"

if command -v systemctl >/dev/null 2>&1; then
    if systemctl is-active --quiet "$SERVICE_NAME" || systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        echo "  Executing: sudo systemctl restart $SERVICE_NAME"
        if command -v sudo >/dev/null 2>&1; then
            sudo systemctl restart "$SERVICE_NAME"
        else
            systemctl restart "$SERVICE_NAME"
        fi
        sleep 2
        
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            echo -e "  ${GREEN}✓ $SERVICE_NAME is ACTIVE and RUNNING!${RESET}"
        else
            echo -e "  ${RED}⚠️ $SERVICE_NAME failed to start.${RESET}"
            echo "  Check service logs: journalctl -u $SERVICE_NAME -n 30 --no-pager"
            exit 1
        fi
    else
        echo -e "  ${YELLOW}ℹ $SERVICE_NAME is not enabled or running.${RESET}"
        echo "  You can start it manually or run setup.sh to configure systemd."
    fi
fi

echo ""
echo -e "${GREEN}${BOLD}======================================================================"
echo "    🎉  Model Update Complete!"
echo "======================================================================${RESET}"
echo -e "  Active Model: ${CYAN}Dual-Brain Ensemble (MobileNetV2 + EfficientNetV2S + Skyline Air-Gap)${RESET}"
echo -e "  Discord:      Use ${BOLD}/model${RESET} to inspect model telemetry or switch to Ensemble Mode"
echo -e "  Logs:         ${CYAN}journalctl -u tdr-ai-bot.service -f${RESET}"
echo -e "${GREEN}======================================================================${RESET}"
