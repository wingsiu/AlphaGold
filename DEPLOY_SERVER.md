# AlphaGold — Server Deployment Guide

## Prerequisites

- Linux server (Ubuntu 22.04+ recommended)
- Python 3.12
- Git access to the repo
- MySQL running with `gold_prices`, `aud_prices`, `prices` tables
- IG API credentials in `.env`

---

## 1. Clone & install

```bash
git clone https://github.com/wingsiu/AlphaGold.git
cd AlphaGold
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install joblib
pip install -r requirements.txt
```

---

## 2. Copy your `.env`

The `.env` is **not** committed (contains IG credentials + MySQL DSN).
Copy it from your Mac:

```bash
# run this on your Mac
scp /Users/alpha/Desktop/python/AlphaGold/.env user@your-server:~/AlphaGold/.env
```

---

## 3. Test one cycle

```bash
cd ~/AlphaGold
source .venv/bin/activate
python3 -u run_live_bot.py --once
tail -50 runtime/trading_bot.log
```

You should see in the log:
```
MODEL CONFIG: ... stage1_min_prob=0.55 stage2_min_prob=0.58 ...
```

---

## 4. Run as a systemd service (recommended)

Create the service file (replace `YOUR_LINUX_USER` with your actual username e.g. `ubuntu`):

```bash
sudo nano /etc/systemd/system/alphagold.service
```

Paste this content:

```ini
[Unit]
Description=AlphaGold Live Trading Bot
After=network.target mysql.service
Wants=mysql.service

[Service]
Type=simple
User=YOUR_LINUX_USER
WorkingDirectory=/home/YOUR_LINUX_USER/AlphaGold
ExecStart=/home/YOUR_LINUX_USER/AlphaGold/.venv/bin/python3 -u run_live_bot.py --sleep-seconds 5 --prediction-poll-second 5 --market-data-poll-second 30 --prediction-cache-max-rows 1200 --max-hold-minutes 60
Restart=on-failure
RestartSec=10
StandardOutput=append:/home/YOUR_LINUX_USER/AlphaGold/runtime/trading_bot.log
StandardError=append:/home/YOUR_LINUX_USER/AlphaGold/runtime/trading_bot.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable alphagold
sudo systemctl start alphagold
sudo systemctl status alphagold
```

---

## 5. Monitor

```bash
# live log
tail -f ~/AlphaGold/runtime/trading_bot.log

# service health
sudo systemctl status alphagold

# current open position / P&L
cat ~/AlphaGold/runtime/trading_bot_status.json | python3 -m json.tool
```

---

## 6. Pull updates from GitHub

After pushing changes from your Mac:

```bash
cd ~/AlphaGold
git pull
sudo systemctl restart alphagold
```

---

## 7. Stop / restart / disable

```bash
sudo systemctl stop alphagold        # stop
sudo systemctl restart alphagold     # restart
sudo systemctl disable alphagold     # don't auto-start on reboot
```

---

## 8. Log rotation (optional)

```bash
sudo nano /etc/logrotate.d/alphagold
```

```
/home/YOUR_LINUX_USER/AlphaGold/runtime/trading_bot.log {
    daily
    rotate 14
    compress
    missingok
    notifempty
    copytruncate
}
```

---

## Active parameters (Candidate E)

| Param | Value |
|---|---|
| stage1_min_prob | 0.55 |
| stage2_min_prob_up (long) | 0.65 |
| stage2_min_prob_down (short) | 0.62 |
| max_hold_minutes | 60 |
| sleep_seconds | 5 |
| weak_filter cells | 15 |

