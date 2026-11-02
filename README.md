# PrinterWatcher — Ender 3 (TinyML + Klipper Discord Bot)

Real-time AI 3D-print failure detector and Discord controller for a **Creality Ender 3** running **Klipper, Moonraker, and Crowsnest** on a **Raspberry Pi 4B**.

It runs a **Dual-Brain TinyML Ensemble** (`MobileNetV2 INT8` + `EfficientNetV2S`) paired with a lighting-invariant **Skyline Air-Gap** clog detector and live Moonraker motion/extrusion telemetry to classify prints into **`Correcte`** or **`Error`** and alert/pause automatically via Discord.

---

## Quick Install (`setup.sh`)

Run directly on your Raspberry Pi 4B:

```bash
git clone https://github.com/pauchux/PrinterWatcher-Ender-3.git ~/tdr-bot
cd ~/tdr-bot
chmod +x setup.sh && ./setup.sh
```

`./setup.sh` automatically:
1. Checks Python 3.10+ and system prerequisites (`python3-venv`, `python3-picamera2`).
2. Creates a lightweight `venv` (`--system-site-packages`) and installs `requirements.txt` (`ai-edge-litert`, `discord.py`, `aiohttp`, `opencv-python-headless`).
3. Creates `.env` from `.env.example` and prompts for your `DISCORD_BOT_TOKEN` and `DISCORD_USER_ID`.
4. Optionally installs and starts `tdr-ai-bot.service` via `systemd` (configured with `Nice=10` so Klipper always keeps real-time priority).

---

## Configuration (`.env`)

If you prefer editing `.env` manually (`cp .env.example .env`):

| Variable | Description | Default |
| :--- | :--- | :--- |
| `DISCORD_BOT_TOKEN` | Discord Bot token from the Discord Developer Portal | *(required)* |
| `DISCORD_USER_ID` | Your Discord User ID (receives DM alerts & authorizes buttons) | *(required)* |
| `MOONRAKER_URL` | Local Moonraker HTTP endpoint | `http://localhost:7125` |
| `CAMERA_SNAPSHOT_URL` | Crowsnest snapshot URL (falls back to `Picamera2` if offline) | `http://127.0.0.1:8080/?action=snapshot` |

---

## Discord Commands

| Command | Action |
| :--- | :--- |
| `/status` | Live snapshot, Klipper temperatures, layer/progress, and Raspberry Pi health |
| `/check` | Run an immediate AI inspection on the current camera frame |
| `/watch` | Live auto-refreshing print dashboard inside Discord |
| `/ai` | Configure mode (`supervised` / `autonomous`), alert threshold (`60%`), ROI, or toggle monitoring |
| `/model` | Inspect latency/telemetry or hot-swap between **Dual-Brain Ensemble** and individual `.tflite` models |
| `/pause` `/resume` `/cancel` | Direct Klipper print control (with confirmation before cancel) |

---

## Repository Contents

- `discord_controller.py` — Discord bot, Moonraker telemetry gate, camera grabber, and hybrid AI engine.
- `printWatcher_efficientnet_v2s.tflite` — Quantized **EfficientNetV2S** (`300×300`, 21.3 MB, trained on Kaggle + Cambridge CAXTON).
- `printer_failure_detector.tflite` — Full-INT8 **MobileNetV2** (`224×224`, 2.6 MB, motion & lighting hardened).
- `setup.sh` — Interactive Raspberry Pi installer and `systemd` service setup.
- `test_model.py` — Standalone CLI smoke test (`./venv/bin/python test_model.py`).
- `update_on_pi.sh` — Helper script to verify models, migrate runtime state, and restart the service.
- `tdr-ai-bot.service` — `systemd` unit template.

### Useful Service Commands

```bash
sudo systemctl status tdr-ai-bot.service
journalctl -u tdr-ai-bot.service -f
```
