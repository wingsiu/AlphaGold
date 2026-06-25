# Remote Desktop (Screen Sharing) on the Mac mini

The **iPhone app** uses HTTP port **8765**. **Remote Desktop / VNC** uses port **5900**. They are unrelated.

## Sudden hang while connected

This is usually **not** the router or AlphaGold API. While you are connected, macOS must **encode the whole screen in real time** (Screen Sharing → `WindowServer`). If the Mac is busy, the session **freezes**, then drops; port **5900** may look “off” afterward until you run `fix_remote_desktop.sh`.

On an **8 GB** Mac mini, typical load while you remote in:

| Process | Why it hurts VNC |
|---------|------------------|
| **Cursor** | High CPU/GPU, constant UI updates |
| **WindowServer** | Encodes every pixel for Remote Desktop |
| **Hybrid bot** | Large Python process, periodic CPU spikes |
| **Screen recording** (`replayd`) | Extra capture pipeline |

### What to do (best → good)

1. **Use Remote Desktop to a “quiet” Mac** — quit **Cursor** on the mini before connecting; develop via **SSH** from your laptop instead (`ssh alpha@192.168.0.4`).
2. **Use the iPhone app** for PnL/signals; use RD only when you need the full UI.
3. In **Microsoft Remote Desktop** (or your client): lower quality / disable wallpaper / smaller resolution.
4. **Ethernet** for the Mac mini if possible (Wi‑Fi + VNC often stutters).
5. After a hang, on the mini: `~/AlphaGold/scripts/fix_remote_desktop.sh`

### Check load during a hang

```bash
~/AlphaGold/scripts/check_remote_desktop.sh
top -o cpu -n 8
```

If `WindowServer` or `Cursor` is high CPU, expect VNC to hang.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/check_remote_desktop.sh` | Is port 5900 open? Legacy VNC? Load warning |
| `scripts/fix_remote_desktop.sh` | Re-enable Screen Sharing (admin password) |
| `scripts/check_remote_access.sh` | Mobile API / launchd (port 8765) |

## Away access

Forward **TCP 5900** → Mac mini LAN IP. Forwarding **8765** alone does **not** enable Remote Desktop.
