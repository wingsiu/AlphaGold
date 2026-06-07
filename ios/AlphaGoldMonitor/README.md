# AlphaGoldMonitor (iOS + Watch)

## Create Xcode project

**Automated (recommended):**

```bash
cd ios/AlphaGoldMonitor
./setup_xcode.sh
open AlphaGoldMonitor.xcodeproj
```

Then set **Signing & Capabilities → Team** in Xcode.

**Manual:**

1. Xcode → **File → New → Project → iOS App**
2. Product name: `AlphaGoldMonitor`, Interface: **SwiftUI**
3. Replace generated `ContentView.swift` and add `Models.swift`, `APIClient.swift` from this folder
4. **Signing & Capabilities** → enable your team
5. Use `Info.plist` from this folder (allows local HTTP via `NSAllowsLocalNetworking`)

## Apple Watch (optional)

1. **File → New Target → Watch App**
2. Add a minimal `WatchStatusView` that calls `APIClient.fetchStatus()`
3. Share `APIClient.swift` + `Models.swift` with Watch target membership

## Settings

On first launch open **Settings** tab:
- **Mac Mini URL**: `http://<lan-ip>:8765`
- **API Key**: same as `MOBILE_API_KEY` in Mac `.env`

Phone and Mac must be on the same Wi‑Fi (or Tailscale VPN).

## Tabs

| Tab | API |
|-----|-----|
| Signals | `GET /api/v1/signals?minutes=30` |
| Today | `GET /api/v1/trades/today` |
| Compare | `GET /api/v1/compare/today` |

Pull to refresh on Signals and Today.
