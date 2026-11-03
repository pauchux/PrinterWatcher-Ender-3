# PrinterWatcher — Ender 3 (TinyML + Klipper Discord Bot)

Real-time AI 3D-print failure detector and Discord controller for a **Creality Ender 3** running **Klipper, Moonraker, Mainsail, and Crowsnest** on a **Raspberry Pi 4B**.

It runs a **Dual-Brain TinyML Ensemble** (`MobileNetV2 INT8` + `EfficientNetV2S`) paired with a lighting-invariant **Skyline Air-Gap** clog detector and live Moonraker motion/extrusion telemetry to classify prints into **`Correcte`** or **`Error`** and alert/pause automatically via Discord.

---

## 1. Required Hardware & 3D-Printed Mounts (STLs)

To run both Klipper and the nozzle-focused TinyML vision system in real time, you need:

- **Raspberry Pi 4B (4 GB RAM recommended)** + official **5.1V / 3A USB-C** power supply + **32 GB microSD** card *(hosts Klipper, Moonraker, Mainsail, Crowsnest, and the AI Discord bot)*.
- **Raspberry Pi Camera Module 3 Wide** *(or **Camera Module 3 Wide NoIR** for low-light operation)* + 15-pin CSI ribbon cable, mounted directly on the X-carriage aimed at the nozzle *(an overhead 24V/5V LED strip is recommended for consistent lighting)*.
- **Creality Ender 3 Hardware (matching [`printer.cfg`](./printer.cfg)):**
  - **BIGTREETECH SKR Mini E3 V3.0** mainboard (STM32G0B1 + silent TMC2209 UART drivers) connected via USB data cable to the Raspberry Pi 4B.
  - **BLTouch** auto-bed leveling sensor.
  - *(Configured in [`printer.cfg`](./printer.cfg)):* **BMG Dual-Drive Extruder (3:1)** + 23 mm pancake NEMA 17 stepper (Direct Drive) and a **5015 24V radial blower fan**.

### 3D-Printed Parts (STL Links)

| Part | Purpose | STL Download |
| :--- | :--- | :--- |
| **Nozzle Camera Mount (NEMA 17 Arm)** | Attaches to the X-carriage extruder motor and aims the **Camera Module 3** directly at the nozzle | [Thingiverse #3965589](https://www.thingiverse.com/thing:3965589) |
| **SpeedDrive v1 Direct Drive Mount** | Mounts the NEMA 17 extruder on the X-carriage *(anchor bracket for the camera mount above)* | [Thingiverse #3816051](https://www.thingiverse.com/thing:3816051) |
| **Petsfang Blockhead 5015 Duct + BLTouch (`X+48, Y-2`)** | High-airflow 5015 cooling duct with an open front sightline so the camera has a clear view of the nozzle | [Thingiverse #2759439](https://www.thingiverse.com/thing:2759439) |
| **Raspberry Pi 4B + Stock LCD Enclosure** | Replaces the stock metal LCD bracket to house the Raspberry Pi 4B with side airflow vents | [Thingiverse #4586351](https://www.thingiverse.com/thing:4586351) |
| **Ender 3 Cable Chains** | Keeps hotend, BLTouch, and CSI camera cables tidy and away from the bed leveling wheels | [Thingiverse #2920060](https://www.thingiverse.com/thing:2920060) |

---

## 2. Klipper Setup Guide on Raspberry Pi 4B

### Step 1 — Flash MainsailOS to the Raspberry Pi 4B
1. Open **[Raspberry Pi Imager](https://www.raspberrypi.com/software/)** on your computer.
2. Select:
   - **Device:** `Raspberry Pi 4`
   - **OS:** `Other specific-purpose OS` → `3D printing` → `Mainsail OS (64-bit)` *(bundles Klipper, Moonraker, Mainsail, and Crowsnest)*
   - **Storage:** Your 32 GB microSD card.
3. Click **Edit Settings** (OS Customisation) to set your **hostname**, **SSH username/password**, and **Wi-Fi** credentials, then flash the card.
4. Insert the microSD card into the Raspberry Pi 4B, connect the **Camera Module 3** CSI ribbon cable to the Pi's `CAMERA` port, power it via USB-C, and SSH in:
   ```bash
   ssh <your-user>@<raspberry-pi-ip>
   ```

### Step 2 — Compile & Flash Klipper Firmware (BIGTREETECH SKR Mini E3 V3.0)
1. On the Raspberry Pi via SSH, configure the Klipper firmware build:
   ```bash
   cd ~/klipper
   make menuconfig
   ```
   Set the following options:
   - **Micro-controller Architecture:** `STMicroelectronics STM32`
   - **Processor model:** `STM32G0B1`
   - **Bootloader offset:** `8KiB bootloader`
   - **Communication interface:** `USB (on PA11/PA12)`
2. Press `Q` then `Y` to save, and compile:
   ```bash
   make
   ```
3. Copy `~/klipper/out/klipper.bin` to a separate FAT32 microSD card (≤ 8–32 GB) and **rename it to `firmware.bin`** *(note: `make flash` over USB is not supported on the SKR Mini E3 V3.0)*.
4. With the Ender 3 powered **off**, insert that microSD card into the **SKR Mini E3 V3.0** onboard slot and power the printer **on** for ~10 seconds (the board flashes automatically and renames the file to `FIRMWARE.CUR`).
5. Connect the SKR Mini E3 V3.0 Micro-USB port to a Raspberry Pi 4B USB port and check the serial ID:
   ```bash
   ls /dev/serial/by-id/*
   ```

### Step 3 — Verify Camera Module 3 in Crowsnest
Open Mainsail in your browser (`http://<raspberry-pi-ip>`), go to **Machine → `crowsnest.conf`**, and ensure `[cam 1]` is configured on port `8080` (`mode: camera-streamer` or `ustreamer`) so the snapshot endpoint `http://127.0.0.1:8080/?action=snapshot` is active.

### Step 4 — Install `printer.cfg`

Copy [`printer.cfg`](./printer.cfg) into `~/printer_data/config/printer.cfg` (or paste it via **Mainsail → Machine → `printer.cfg`**):

> **Before your first print:**
> 1. Replace `serial:` under `[mcu]` with the exact path returned by `ls /dev/serial/by-id/*`.
> 2. Run PID tuning in the Mainsail console (`PID_CALIBRATE HEATER=extruder TARGET=215` and `PID_CALIBRATE HEATER=heater_bed TARGET=60`) and calibrate your BLTouch Z-offset (`PROBE_CALIBRATE`), then run `SAVE_CONFIG` *(or uncomment the `#control: pid` / `#z_offset` lines and add `min_temp: 0` / `max_temp: 265` under `[extruder]` before running `SAVE_CONFIG`)*.
> 3. If you are using the **stock Creality extruder** instead of the 3:1 BMG Direct Drive, remove `gear_ratio: 50:17` under `[extruder]` and set `rotation_distance: 33.500`.

<details>
<summary><b>Click to view / copy <code>printer.cfg</code> (SKR Mini E3 V3.0 + BLTouch + BMG Direct Drive)</b></summary>

```ini
[include mainsail.cfg]
# This file contains common pin mappings for the BIGTREETECH SKR mini
# E3 v3.0. To use this config, the firmware should be compiled for the
# STM32G0B1 with a "8KiB bootloader" and USB communication.

# The "make flash" command does not work on the SKR mini E3. Instead,
# after running "make", copy the generated "out/klipper.bin" file to a
# file named "firmware.bin" on an SD card and then restart the SKR
# mini E3 with that SD card.

# See docs/Config_Reference.md for a description of parameters.

[stepper_x]
step_pin: PB13
dir_pin: !PB12
enable_pin: !PB14
microsteps: 16
rotation_distance: 40
endstop_pin: ^PC0
position_endstop: 0
position_max: 200
homing_speed: 50

[tmc2209 stepper_x]
uart_pin: PC11
tx_pin: PC10
uart_address: 0
run_current: 0.580
stealthchop_threshold: 999999

[stepper_y]
step_pin: PB10
dir_pin: !PB2
enable_pin: !PB11
microsteps: 16
rotation_distance: 40
endstop_pin: ^PC1
position_endstop: 0
position_max: 235
homing_speed: 50
[tmc2209 stepper_y]
uart_pin: PC11
tx_pin: PC10
uart_address: 2
run_current: 0.580
stealthchop_threshold: 999999

[stepper_z]
step_pin: PB0
dir_pin: PC5
enable_pin: !PB1
microsteps: 16
rotation_distance: 8
endstop_pin: probe:z_virtual_endstop
position_max: 250
position_min: -5


[tmc2209 stepper_z]
uart_pin: PC11
tx_pin: PC10
uart_address: 1
run_current: 0.580
stealthchop_threshold: 999999

[extruder]
step_pin: PB3
dir_pin: !PB4
enable_pin: !PD1
microsteps: 16
rotation_distance: 22.6789
gear_ratio: 50:17
nozzle_diameter: 0.400
filament_diameter: 1.750
heater_pin: PC8
sensor_type: EPCOS 100K B57560G104F
sensor_pin: PA0
#control: pid
#pid_Kp: 21.527
#pid_Ki: 1.063

[tmc2209 extruder]
uart_pin: PC11
tx_pin: PC10
uart_address: 3
run_current: 0.450

[heater_bed]
heater_pin: PC9
sensor_type: EPCOS 100K B57560G104F
sensor_pin: PC4
#control: pid
#pid_Kp: 54.027
#pid_Ki: 0.770
#pid_Kd: 948.182
min_temp: 0
max_temp: 130

[heater_fan heatbreak_cooling_fan]
pin: PC7

[heater_fan controller_fan]
pin: PB15

[fan]
pin: PC6

[mcu]
serial: /dev/serial/by-id/usb-Klipper_stm32g0b1xx_230040000750304158373220-if00

[printer]
kinematics: cartesian
max_velocity: 300
max_accel: 3000
max_z_velocity: 5
max_z_accel: 100

[board_pins]
aliases:
    # EXP1 header
    EXP1_1=PB5,  EXP1_3=PA9,   EXP1_5=PA10, EXP1_7=PB8, EXP1_9=<GND>,
    EXP1_2=PA15, EXP1_4=<RST>, EXP1_6=PB9,  EXP1_8=PD6, EXP1_10=<5V>

# See the sample-lcd.cfg file for definitions of common LCD displays.
[display]
lcd_type: st7920
cs_pin: PB8
sclk_pin: PB9
sid_pin: PD6
encoder_pins: ^PA10, ^PA9
click_pin: ^!PA15

[bltouch]
sensor_pin: ^PC14
control_pin: PA1
x_offset: 48
y_offset: -2
#z_offset: 2.0
speed: 3
samples: 2
pin_up_touch_mode_reports_triggered: False
pin_up_reports_not_triggered: True
pin_move_time: 0.680

[safe_z_home]
home_xy_position: 69.5, 119.5
speed: 50
z_hop: 10
z_hop_speed: 5

[bed_mesh]
speed: 100
horizontal_move_z: 6
mesh_min: 49, 24
mesh_max: 190, 220
probe_count: 5, 5

[delayed_gcode KEEP_ALIVE]
gcode:
    {% if printer.idle_timeout.state == "Printing" or printer['gcode_macro START_DRYER'].active %}
        SET_IDLE_TIMEOUT TIMEOUT=86400
        UPDATE_DELAYED_GCODE ID=KEEP_ALIVE DURATION=3600
    {% endif %}
```

</details>

---

## 3. AI Discord Bot Quick Install (`setup.sh`)

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

### Configuration (`.env`)

If you prefer editing `.env` manually (`cp .env.example .env`):

| Variable | Description | Default |
| :--- | :--- | :--- |
| `DISCORD_BOT_TOKEN` | Discord Bot token from the Discord Developer Portal | *(required)* |
| `DISCORD_USER_ID` | Your Discord User ID (receives DM alerts & authorizes buttons) | *(required)* |
| `MOONRAKER_URL` | Local Moonraker HTTP endpoint | `http://localhost:7125` |
| `CAMERA_SNAPSHOT_URL` | Crowsnest snapshot URL (falls back to `Picamera2` if offline) | `http://127.0.0.1:8080/?action=snapshot` |

---

## 4. Discord Commands

| Command | Action |
| :--- | :--- |
| `/status` | Live snapshot, Klipper temperatures, layer/progress, and Raspberry Pi health |
| `/check` | Run an immediate AI inspection on the current camera frame |
| `/watch` | Live auto-refreshing print dashboard inside Discord |
| `/ai` | Configure mode (`supervised` / `autonomous`), alert threshold (`60%`), ROI, or toggle monitoring |
| `/model` | Inspect latency/telemetry or hot-swap between **Dual-Brain Ensemble** and individual `.tflite` models |
| `/pause` `/resume` `/cancel` | Direct Klipper print control (with confirmation before cancel) |

---

## 5. Repository Contents

- `discord_controller.py` — Discord bot, Moonraker telemetry gate, camera grabber, and hybrid AI engine.
- `printWatcher_efficientnet_v2s.tflite` — Quantized **EfficientNetV2S** (`300×300`, 21.3 MB, trained on Kaggle + Cambridge CAXTON).
- `printer_failure_detector.tflite` — Full-INT8 **MobileNetV2** (`224×224`, 2.6 MB, motion & lighting hardened).
- `printer.cfg` — Ready-to-copy Klipper configuration for Ender 3 + BIGTREETECH SKR Mini E3 V3.0 + BLTouch + BMG Direct Drive.
- `setup.sh` — Interactive Raspberry Pi installer and `systemd` service setup.
- `test_model.py` — Standalone CLI smoke test (`./venv/bin/python test_model.py`).
- `update_on_pi.sh` — Helper script to verify models, migrate runtime state, and restart the service.
- `tdr-ai-bot.service` — `systemd` unit template.

### Useful Service Commands

```bash
sudo systemctl status tdr-ai-bot.service
journalctl -u tdr-ai-bot.service -f
```
