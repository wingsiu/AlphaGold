import SwiftUI
import UIKit

@main
struct AlphaGoldMonitorApp: App {
    @StateObject private var api = APIClient()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(api)
        }
    }
}

struct RootView: View {
    @EnvironmentObject var api: APIClient
    @State private var showSplash = true
    @State private var splashStatus = "Starting…"

    var body: some View {
        ZStack {
            ContentView()
                .opacity(showSplash ? 0 : 1)

            if showSplash {
                LaunchSplashView(status: splashStatus)
                    .transition(.opacity)
            }
        }
        .task { await bootstrap() }
    }

    private func bootstrap() async {
        // Show UI quickly — data loads in the header/tabs after launch.
        try? await Task.sleep(nanoseconds: 300_000_000)
        withAnimation(.easeOut(duration: 0.2)) {
            showSplash = false
        }
        Task {
            splashStatus = "Connecting…"
            _ = try? await api.fetchHealth()
        }
    }
}

struct LaunchSplashView: View {
    let status: String

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(red: 0.08, green: 0.07, blue: 0.05), Color(red: 0.16, green: 0.12, blue: 0.04)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            VStack(spacing: 20) {
                Image(systemName: "chart.line.uptrend.xyaxis.circle.fill")
                    .font(.system(size: 72))
                    .foregroundStyle(.yellow, .orange)
                    .symbolEffect(.pulse)

                VStack(spacing: 6) {
                    Text("AlphaGold")
                        .font(.largeTitle.bold())
                        .foregroundStyle(.white)
                    Text("Monitor")
                        .font(.title3)
                        .foregroundStyle(.white.opacity(0.75))
                }

                ProgressView()
                    .tint(.yellow)
                    .padding(.top, 8)

                Text(status)
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.7))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
            }
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var api: APIClient
    @StateObject private var statusStore = StatusStore()
    @State private var selectedTab = 0

    var body: some View {
        VStack(spacing: 0) {
            AccountStatusHeader(
                status: statusStore.status,
                isLoading: statusStore.isLoading,
                error: statusStore.lastError
            )
            TabView(selection: $selectedTab) {
                SignalsView(isActive: selectedTab == 0)
                    .tag(0)
                    .tabItem { Label("Signals", systemImage: "waveform.path.ecg") }
                TradesView(isActive: selectedTab == 1)
                    .tag(1)
                    .tabItem { Label("Today", systemImage: "list.bullet.rectangle") }
                CompareView(isActive: selectedTab == 2)
                    .tag(2)
                    .tabItem { Label("Compare", systemImage: "arrow.left.arrow.right") }
                SettingsView()
                    .tag(3)
                    .tabItem { Label("Settings", systemImage: "gear") }
            }
        }
        .task {
            try? await Task.sleep(nanoseconds: 200_000_000)
            await statusStore.runPolling(api: api)
        }
    }
}

struct AccountStatusHeader: View {
    let status: StatusResponse?
    let isLoading: Bool
    let error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let status {
                igAccountRow(status.ig_account, gold: status.gold)

                HStack(spacing: 12) {
                    statPill("Day", status.trading_day ?? "—")
                    statPill("Trades", "\(status.today.trade_count)")
                    statPill("PnL", String(format: "%+.1f", status.today.net_pnl),
                             color: status.today.net_pnl >= 0 ? .green : .red)
                    statPill("WR", String(format: "%.0f%%", status.today.win_rate))
                }
                .font(.caption)

                if status.is_fallback == true {
                    Text("Showing latest: \(status.trading_day ?? "") (\(status.source ?? ""))")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }

                if let pending = status.today.pending_pnl_count, pending > 0 {
                    Text("\(pending) trade(s) awaiting broker PnL")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }

                positionRow(status.open_position)
            } else if isLoading {
                HStack(spacing: 8) {
                    ProgressView().scaleEffect(0.8)
                    Text("Loading account…").font(.caption).foregroundStyle(.secondary)
                }
            } else if let error {
                Text(error).font(.caption2).foregroundStyle(.red).lineLimit(2)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Color(.secondarySystemBackground))
    }

    private static func friendlyIGError(_ raw: String) -> String {
        if raw.contains("keychain") || raw.contains("(-61") {
            return "Mac keychain blocked IG login (launchd). Restart mobile API on Mac."
        }
        return raw
    }

    @ViewBuilder
    private func igAccountRow(_ account: IGAccount?, gold: GoldPrice?) -> some View {
        if let account, account.status != "error", account.balance != nil {
            HStack(spacing: 10) {
                statPill("Balance", money(account.balance))
                statPill("Equity", money(account.equity))
                statPill("Avail", money(account.available))
                statPill("P&L", money(account.profit_loss),
                         color: (account.profit_loss ?? 0) >= 0 ? .green : .red)
                if let close = gold?.close {
                    statPill("Close", money(close, decimals: 2))
                }
            }
        } else if let err = account?.error {
            Text("IG: \(Self.friendlyIGError(err))").font(.caption2).foregroundStyle(.orange).lineLimit(2)
        } else {
            HStack(spacing: 10) {
                Text("IG account unavailable").font(.caption2).foregroundStyle(.secondary)
                if let close = gold?.close {
                    statPill("Close", money(close, decimals: 2))
                }
            }
        }
    }

    private func money(_ value: Double?, decimals: Int = 0) -> String {
        guard let value else { return "—" }
        if decimals == 0 {
            return String(format: "$%.0f", value)
        }
        return String(format: "$%.2f", value)
    }

    @ViewBuilder
    private func positionRow(_ pos: OpenPosition?) -> some View {
        if let pos, !pos.isFlat {
            HStack(spacing: 8) {
                Image(systemName: "circle.fill")
                    .foregroundStyle(pos.open_side ?? 0 > 0 ? .green : .red)
                    .font(.caption2)
                Text(pos.sideLabel)
                    .font(.subheadline.bold())
                if let src = pos.open_position_source {
                    Text(src).font(.caption).foregroundStyle(.secondary)
                }
                if let pat = pos.open_pattern_name {
                    Text(pat).font(.caption).lineLimit(1)
                }
                Spacer(minLength: 0)
                if let tp = pos.open_tp {
                    Text("TP \(Int(tp))").font(.caption2)
                }
                if let sl = pos.open_sl {
                    Text("SL \(Int(sl))").font(.caption2)
                }
                if let end = pos.expectedHorizonEndISO {
                    Text("End \(formatTradeTimeHKT(end))")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
            }
        } else {
            HStack(spacing: 6) {
                Image(systemName: "circle")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text("FLAT — no open position")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private func statPill(_ label: String, _ value: String, color: Color = .primary) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.caption.bold()).foregroundStyle(color)
        }
    }
}

struct LoadingOverlay: View {
    let message: String

    var body: some View {
        VStack(spacing: 12) {
            ProgressView()
            Text(message)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct FallbackBanner: View {
    let tradingDay: String?
    let source: String?

    var body: some View {
        if let tradingDay {
            HStack(spacing: 8) {
                Image(systemName: "clock.arrow.circlepath")
                Text(bannerText(tradingDay: tradingDay))
                    .font(.caption)
            }
            .foregroundStyle(.orange)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(Color.orange.opacity(0.12))
        }
    }

    private func bannerText(tradingDay: String) -> String {
        if tradingDay == "recent" {
            return "No signals in the last 30 min — showing latest available"
        }
        if source == "backtest" {
            return "No live data today — showing latest backtest: \(tradingDay)"
        }
        return "No data today — showing latest: \(tradingDay)"
    }
}

struct SignalsView: View {
    let isActive: Bool
    @EnvironmentObject var api: APIClient
    @State private var signals: [SignalRow] = []
    @State private var tradingWindow: TradingWindow?
    @State private var error: String?
    @State private var isLoading = true
    @State private var isFallback = false
    @State private var title = "Last 30 min"

    var body: some View {
        NavigationStack {
            Group {
                if isLoading && signals.isEmpty && error == nil {
                    LoadingOverlay(message: "Loading signals…")
                } else {
                    List {
                        if let tw = tradingWindow {
                            Section {
                                TradingWindowBanner(window: tw)
                            }
                        }
                        if isFallback {
                            Section {
                                FallbackBanner(tradingDay: "recent", source: "journal")
                            }
                        }
                        ForEach(signals, id: \.bar_time) { s in
                            SignalMinuteRow(signal: s)
                        }
                    }
                }
            }
            .navigationTitle(title)
            .refreshable { await load() }
            .task(id: isActive) {
                guard isActive else { return }
                await load()
            }
            .overlay {
                if let error {
                    ContentUnavailableView {
                        Label("Could not load", systemImage: "wifi.exclamationmark")
                    } description: {
                        Text(error)
                    }
                } else if !isLoading && signals.isEmpty {
                    ContentUnavailableView {
                        Label("No signals yet", systemImage: "waveform.path.ecg")
                    } description: {
                        Text("The bot will populate this once it starts scoring bars.")
                    }
                }
            }
        }
    }

    func load() async {
        isLoading = true
        defer { isLoading = false }
        do {
            let r = try await api.fetchSignals(minutes: 30)
            signals = r.signals
            tradingWindow = r.trading_window
            isFallback = r.is_fallback ?? false
            title = isFallback ? "Latest signals" : "Last 30 min"
            error = nil
        } catch {
            self.error = error.localizedDescription
        }
    }
}

struct TradingWindowBanner: View {
    let window: TradingWindow

    var body: some View {
        if window.in_window {
            let sess = window.active_sessions?.map { $0.uppercased() }.joined(separator: ", ")
            Label {
                Text(sess.map { "Trading window OPEN · \($0)" } ?? "Trading window OPEN")
                    .font(.subheadline)
            } icon: {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            }
        } else {
            VStack(alignment: .leading, spacing: 4) {
                Label {
                    Text("Trading window CLOSED")
                        .font(.subheadline.bold())
                } icon: {
                    Image(systemName: "pause.circle.fill").foregroundStyle(.orange)
                }
                if let next = window.next_window_label {
                    Text("Next: \(next)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if let start = window.next_window_start_utc {
                    Text("Next: \(formatHKT(start))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let reasons = window.blocked_reasons, !reasons.isEmpty {
                    Text(reasonText(reasons))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func reasonText(_ reasons: [String]) -> String {
        reasons.map {
            switch $0 {
            case "market_closed": return "Market closed"
            case "time_filter": return "Weak time slot"
            case "outside_session": return "Outside session hours"
            default: return $0
            }
        }.joined(separator: " · ")
    }

    private func formatHKT(_ iso: String) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        guard let d = f.date(from: iso) ?? ISO8601DateFormatter().date(from: iso) else {
            return iso
        }
        let out = DateFormatter()
        out.timeZone = TimeZone(identifier: "Asia/Hong_Kong")
        out.dateFormat = "EEE HH:mm"
        return out.string(from: d) + " HKT"
    }
}

struct SignalMinuteRow: View {
    let signal: SignalRow

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(formatBarTime(signal.bar_time))
                .font(.caption)
                .foregroundStyle(.secondary)

            if signal.action == "no_data" {
                Text("No data")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else if signal.action == "no_score" {
                HStack(spacing: 8) {
                    Text("No signal")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    goldPriceLabel
                }
            } else {
                if hasRoutedPattern {
                    patternLine
                }
                if showEnergeticLine {
                    energeticLine
                }
                resultLine
                if isTradeAction {
                    Text(actionLabel)
                        .font(.caption2.bold())
                        .foregroundStyle(actionColor(signal.action))
                    goldPriceLabel
                }
            }
        }
    }

    @ViewBuilder
    private var goldPriceLabel: some View {
        if let gold = signal.gold_price {
            Text("Gold: \(String(format: "%.2f", gold))")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    private var hasRoutedPattern: Bool {
        guard let name = signal.pattern_name, !name.isEmpty else { return false }
        return true
    }

    private var showEnergeticLine: Bool {
        signal.energetic_prob != nil || (signal.energetic_side ?? 0) != 0
            || (signal.detail ?? "").contains("energetic")
    }

    @ViewBuilder
    private var patternLine: some View {
        HStack(spacing: 6) {
            Text("Pattern").font(.caption2.bold()).foregroundStyle(.blue)
            Text(signal.pattern_name ?? "—").font(.subheadline.bold())
            if let side = signal.pattern_side, side != 0 {
                Text(side > 0 ? "LONG" : "SHORT")
                    .font(.subheadline)
                    .foregroundStyle(side > 0 ? .green : .red)
            }
            if let prob = signal.pattern_prob {
                Text(String(format: "prob %.1f%%", prob * 100))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var energeticLine: some View {
        HStack(spacing: 6) {
            Text("Energetic").font(.caption2.bold()).foregroundStyle(.purple)
            if let side = signal.energetic_side, side != 0 {
                Text(side > 0 ? "LONG" : "SHORT")
                    .font(.subheadline)
                    .foregroundStyle(side > 0 ? .green : .red)
            }
            if let prob = signal.energetic_prob {
                Text(String(format: "prob %.1f%%", prob * 100))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if (signal.detail ?? "").contains("energetic_no_signal") {
                Text("energetic bar")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    @ViewBuilder
    private var resultLine: some View {
        if isTradeAction {
            EmptyView()
        } else if hasRoutedPattern || showEnergeticLine {
            HStack(spacing: 8) {
                Text("No signal")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                goldPriceLabel
            }
        } else {
            HStack(spacing: 8) {
                Text("No signal")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                goldPriceLabel
            }
        }
    }

    private var isTradeAction: Bool {
        let a = signal.action
        return a == "entry" || a.contains("blocked") || a.contains("close")
    }

    private var actionLabel: String {
        switch signal.action {
        case "entry": return "Entry"
        case "blocked_time_filter": return "Blocked · time filter"
        case "blocked_market": return "Blocked · market"
        default:
            return signal.action.replacingOccurrences(of: "_", with: " ")
        }
    }

    private func formatBarTime(_ iso: String) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = f.date(from: iso) ?? ISO8601DateFormatter().date(from: iso) {
            let out = DateFormatter()
            out.timeStyle = .short
            out.dateStyle = .none
            out.timeZone = TimeZone(identifier: "Asia/Hong_Kong")
            return out.string(from: d) + " HKT"
        }
        return iso
    }

    private func actionColor(_ action: String) -> Color {
        if action.contains("blocked") { return .orange }
        if action == "entry" { return .green }
        return .secondary
    }
}

struct TradesView: View {
    let isActive: Bool
    @EnvironmentObject var api: APIClient
    @State private var trades: [TradeRow] = []
    @State private var summary: TradeSummary?
    @State private var isLoading = true
    @State private var tradingDay: String?
    @State private var isFallback = false
    @State private var source: String?

    var body: some View {
        NavigationStack {
            Group {
                if isLoading && trades.isEmpty && summary == nil {
                    LoadingOverlay(message: "Loading trades…")
                } else {
                    List {
                        if isFallback, let tradingDay {
                            Section {
                                FallbackBanner(tradingDay: tradingDay, source: source)
                            }
                        }
                        if let s = summary {
                            Section("Summary") {
                                LabeledContent("Trades", value: "\(s.trade_count)")
                                LabeledContent("Net PnL", value: String(format: "%+.1f", s.net_pnl))
                                LabeledContent("Win rate", value: String(format: "%.1f%%", s.win_rate))
                                if let pending = s.pending_pnl_count, pending > 0 {
                                    LabeledContent("Pending PnL", value: "\(pending)")
                                }
                            }
                        }
                        Section("Trades") {
                            ForEach(trades) { t in
                                TradeDetailRow(trade: t)
                            }
                        }
                    }
                }
            }
            .navigationTitle(isFallback ? (tradingDay ?? "Latest") : "Today")
            .refreshable { await load() }
            .task(id: isActive) {
                guard isActive else { return }
                await load()
            }
        }
    }

    func load() async {
        isLoading = true
        defer { isLoading = false }
        if let r = try? await api.fetchTradesToday() {
            trades = r.trades
            summary = r.summary
            tradingDay = r.trading_day
            isFallback = r.is_fallback ?? false
            source = r.source
        }
    }
}

struct CompareView: View {
    let isActive: Bool
    @EnvironmentObject var api: APIClient
    @State private var compare: CompareResponse?
    @State private var isLoading = false
    @State private var isRefreshingBT = false
    @State private var loadError: String?

    var body: some View {
        NavigationStack {
            Group {
                if isLoading && compare == nil {
                    LoadingOverlay(
                        message: isRefreshingBT
                            ? "Running backtest…\nThis can take 1–2 minutes."
                            : "Loading compare…"
                    )
                } else if let compare {
                    compareList(compare)
                } else if let loadError {
                    ContentUnavailableView {
                        Label("Could not load", systemImage: "wifi.exclamationmark")
                    } description: {
                        Text(loadError)
                    } actions: {
                        Button("Retry") { Task { await load(refresh: false) } }
                            .buttonStyle(.borderedProminent)
                    }
                } else {
                    ContentUnavailableView {
                        Label("Live vs Backtest", systemImage: "arrow.left.arrow.right")
                    } description: {
                        Text("Compare live trades with the hybrid backtest for today.")
                    } actions: {
                        Button("Load") { Task { await load(refresh: false) } }
                            .buttonStyle(.borderedProminent)
                    }
                }
            }
            .navigationTitle(navTitle)
            .toolbar {
                Button("Refresh BT") { Task { await load(refresh: true) } }
                    .disabled(isLoading)
            }
            .refreshable { await load(refresh: false) }
            .task(id: isActive) {
                guard isActive, compare == nil, !isLoading else { return }
                await load(refresh: false)
            }
        }
    }

    private var navTitle: String {
        if let d = compare?.trading_day {
            return "Compare \(d)"
        }
        return "Live vs BT"
    }

    @ViewBuilder
    private func compareList(_ c: CompareResponse) -> some View {
        List {
            Section("Trading day") {
                LabeledContent("Day", value: c.trading_day)
                if let win = c.trading_day_window_hkt {
                    LabeledContent("Window", value: win)
                }
                if c.is_fallback == true {
                    Text("Live: showing latest available (\(c.source ?? ""))")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }

            if c.both_empty == true {
                Section {
                    Label {
                        Text(c.note ?? "No trades yet — live and backtest both flat.")
                            .font(.subheadline)
                    } icon: {
                        Image(systemName: "equal.circle")
                            .foregroundStyle(.secondary)
                    }
                }
            } else if let note = c.note {
                Section {
                    Text(note)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section("Live") {
                metric("Trades", c.live.trade_count)
                metric("PnL", String(format: "%+.1f", c.live.net_pnl))
                metric("WR", String(format: "%.1f%%", c.live.win_rate))
            }
            if let liveTrades = c.live.trades, !liveTrades.isEmpty {
                Section("Live trades") {
                    ForEach(liveTrades) { t in
                        TradeDetailRow(trade: t)
                    }
                }
            }
            Section("Backtest") {
                metric("Trades", c.backtest.trade_count)
                metric("PnL", String(format: "%+.1f", c.backtest.net_pnl))
                metric("WR", String(format: "%.1f%%", c.backtest.win_rate))
            }
            if let btTrades = c.backtest.trades, !btTrades.isEmpty {
                Section("Backtest trades") {
                    ForEach(btTrades) { t in
                        TradeDetailRow(trade: t)
                    }
                }
            }
            Section("Delta (Live − BT)") {
                metric("Δ Trades", c.delta.trade_count)
                metric("Δ PnL", String(format: "%+.1f", c.delta.net_pnl))
                metric("Δ WR", String(format: "%.1f%%", c.delta.win_rate))
            }
        }
    }

    @ViewBuilder
    func metric(_ label: String, _ value: CustomStringConvertible) -> some View {
        LabeledContent(label, value: String(describing: value))
    }

    func load(refresh: Bool) async {
        isLoading = true
        isRefreshingBT = refresh
        loadError = nil
        defer {
            isLoading = false
            isRefreshingBT = false
        }
        do {
            compare = try await api.fetchCompare(refresh: refresh)
        } catch {
            loadError = error.localizedDescription
        }
    }
}

struct TradeDetailRow: View {
    let trade: TradeRow

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(trade.source ?? "?")
                    .font(.subheadline.bold())
                Text(sideLabel)
                    .font(.subheadline)
                    .foregroundStyle(sideColor)
                Spacer()
                pnlLabel
            }
            if let pattern = trade.pattern_name, !pattern.isEmpty {
                Text(pattern)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let entry = trade.entry_time {
                tradeLine("Entry", time: entry, price: trade.effectiveEntryPrice)
            }
            if let exit = trade.exit_time {
                tradeLine("Exit", time: exit, price: trade.exit_price)
            }
            if let deadline = trade.horizon_deadline {
                tradeLine("End", time: deadline, price: nil, labelColor: .orange)
            }
            if let reason = trade.exit_reason, !reason.isEmpty, trade.status == "closed" {
                Text(exitReasonLabel(reason))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private var pnlLabel: some View {
        if trade.status == "open" {
            Text("OPEN")
                .font(.caption.bold())
                .foregroundStyle(.orange)
        } else if trade.pnl_confirmed == false {
            Text("PnL pending")
                .font(.caption.bold())
                .foregroundStyle(.orange)
        } else if let pnl = trade.pnl {
            Text(String(format: "%+.2f", pnl))
                .font(.subheadline.bold())
                .foregroundStyle(pnl >= 0 ? .green : .red)
        } else {
            Text("PnL pending")
                .font(.caption.bold())
                .foregroundStyle(.orange)
        }
    }

    private func exitReasonLabel(_ reason: String) -> String {
        switch reason {
        case "pnl_pending": return "Awaiting broker confirmation"
        case "estimated_ohlc": return "Estimate removed — awaiting broker"
        default: return reason
        }
    }

    private var sideLabel: String {
        guard let s = trade.side, s != 0 else { return "—" }
        return s > 0 ? "LONG" : "SHORT"
    }

    private var sideColor: Color {
        guard let s = trade.side, s != 0 else { return .secondary }
        return s > 0 ? .green : .red
    }

    @ViewBuilder
    private func tradeLine(_ label: String, time: String, price: Double?, labelColor: Color = .secondary) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.caption2.bold())
                .foregroundStyle(labelColor)
                .frame(width: 34, alignment: .leading)
            Text(formatTradeTimeHKT(time))
                .font(.caption)
            if let price {
                Text(String(format: "@ %.2f", price))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private func formatTradeTimeHKT(_ iso: String) -> String {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    guard let d = f.date(from: iso) ?? ISO8601DateFormatter().date(from: iso) else {
        return iso
    }
    let out = DateFormatter()
    out.timeZone = TimeZone(identifier: "Asia/Hong_Kong")
    out.dateFormat = "EEE HH:mm"
    return out.string(from: d) + " HKT"
}

struct SettingsView: View {
    @EnvironmentObject var api: APIClient
    @State private var testResult: String?
    @State private var isTesting = false
    @State private var showAPIKey = true
    @FocusState private var focusedField: Field?

    private enum Field: Hashable {
        case url, apiKey
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("http://123.203.51.164:8765", text: $api.baseURL)
                        .font(.body.monospaced())
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                        .textContentType(.URL)
                        .focused($focusedField, equals: .url)
                        .textSelection(.enabled)

                    HStack {
                        Button("Use Public IP") { api.baseURL = APIClient.publicURL }
                        Button("Use Home IP") { api.baseURL = APIClient.homeURL }
                    }
                    .buttonStyle(.bordered)
                } header: {
                    Text("Mac Mini URL")
                } footer: {
                    Text("No path at the end — the app adds /api/v1/…")
                }

                Section {
                    if showAPIKey {
                        TextField("Paste MOBILE_API_KEY from Mac .env", text: $api.apiKey)
                            .font(.body.monospaced())
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .keyboardType(.asciiCapable)
                            .focused($focusedField, equals: .apiKey)
                            .textSelection(.enabled)
                    } else {
                        SecureField("API Key", text: $api.apiKey)
                            .font(.body.monospaced())
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .keyboardType(.asciiCapable)
                            .focused($focusedField, equals: .apiKey)
                    }

                    Toggle("Show API key", isOn: $showAPIKey)

                    HStack {
                        Button("Paste Key") { pasteAPIKey() }
                        Button("Copy Key") { copyAPIKey() }
                            .disabled(api.apiKey.isEmpty)
                        Button("Clear") { api.apiKey = "" }
                            .foregroundStyle(.red)
                    }
                    .buttonStyle(.bordered)

                    Text("Length: \(api.apiKey.count) chars (expect 43 from .env)")
                        .font(.caption)
                        .foregroundStyle(api.apiKey.count == 43 ? Color.secondary : Color.orange)
                } header: {
                    Text("API Key")
                } footer: {
                    Text("On Mac: grep MOBILE_API_KEY ~/AlphaGold/.env")
                }

                Section("Connection") {
                    Button(isTesting ? "Testing…" : "Test Connection") {
                        focusedField = nil
                        Task { await runTest() }
                    }
                    .disabled(isTesting)

                    if let testResult {
                        Text(testResult)
                            .font(.caption)
                            .foregroundStyle(testResult.contains("❌") ? .red : .green)
                            .textSelection(.enabled)
                    }
                }
            }
            .navigationTitle("Settings")
            .scrollDismissesKeyboard(.interactively)
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { focusedField = nil }
                }
            }
        }
    }

    private func pasteAPIKey() {
        guard let text = UIPasteboard.general.string else { return }
        api.apiKey = text.trimmingCharacters(in: .whitespacesAndNewlines)
        showAPIKey = true
    }

    private func copyAPIKey() {
        UIPasteboard.general.string = api.apiKey
    }

    func runTest() async {
        isTesting = true
        defer { isTesting = false }
        testResult = await api.testConnection()
    }
}
