import SwiftUI

@main
struct AlphaGoldMonitorApp: App {
    @StateObject private var api = APIClient()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(api)
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var api: APIClient

    var body: some View {
        TabView {
            SignalsView()
                .tabItem { Label("Signals", systemImage: "waveform.path.ecg") }
            TradesView()
                .tabItem { Label("Today", systemImage: "list.bullet.rectangle") }
            CompareView()
                .tabItem { Label("Compare", systemImage: "arrow.left.arrow.right") }
            SettingsView()
                .tabItem { Label("Settings", systemImage: "gear") }
        }
    }
}

struct SignalsView: View {
    @EnvironmentObject var api: APIClient
    @State private var signals: [SignalRow] = []
    @State private var error: String?

    var body: some View {
        NavigationStack {
            List(signals) { s in
                VStack(alignment: .leading, spacing: 4) {
                    Text(s.bar_time).font(.caption).foregroundStyle(.secondary)
                    HStack {
                        if let p = s.pattern_name {
                            Text("P: \(p)").font(.subheadline.bold())
                        }
                        if let es = s.energetic_side, es != 0 {
                            Text("E: \(es > 0 ? "LONG" : "SHORT")").font(.subheadline)
                        }
                    }
                    Text(s.action).font(.caption2).foregroundStyle(.tertiary)
                }
            }
            .navigationTitle("Last 30 min")
            .refreshable { await load() }
            .task { await load() }
            .overlay {
                if let error { Text(error).foregroundStyle(.red).padding() }
            }
        }
    }

    func load() async {
        do {
            let r = try await api.fetchSignals(minutes: 30)
            signals = r.signals
            error = nil
        } catch {
            self.error = error.localizedDescription
        }
    }
}

struct TradesView: View {
    @EnvironmentObject var api: APIClient
    @State private var trades: [TradeRow] = []
    @State private var summary: TradeSummary?

    var body: some View {
        NavigationStack {
            List {
                if let s = summary {
                    Section("Summary") {
                        LabeledContent("Trades", value: "\(s.trade_count)")
                        LabeledContent("Net PnL", value: String(format: "%+.1f", s.net_pnl))
                        LabeledContent("Win rate", value: String(format: "%.1f%%", s.win_rate))
                    }
                }
                Section("Trades") {
                    ForEach(trades) { t in
                        VStack(alignment: .leading) {
                            Text("\(t.source ?? "?") · \(t.side ?? 0 > 0 ? "LONG" : "SHORT")")
                            if let pnl = t.pnl {
                                Text(String(format: "PnL %+.2f", pnl)).foregroundStyle(pnl >= 0 ? .green : .red)
                            } else {
                                Text(t.status).foregroundStyle(.orange)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Today")
            .refreshable { await load() }
            .task { await load() }
        }
    }

    func load() async {
        if let r = try? await api.fetchTradesToday() {
            trades = r.trades
            summary = r.summary
        }
    }
}

struct CompareView: View {
    @EnvironmentObject var api: APIClient
    @State private var compare: CompareResponse?

    var body: some View {
        NavigationStack {
            List {
                if let c = compare {
                    Section("Live") {
                        metric("Trades", c.live.trade_count)
                        metric("PnL", String(format: "%+.1f", c.live.net_pnl))
                        metric("WR", String(format: "%.1f%%", c.live.win_rate))
                    }
                    Section("Backtest") {
                        metric("Trades", c.backtest.trade_count)
                        metric("PnL", String(format: "%+.1f", c.backtest.net_pnl))
                        metric("WR", String(format: "%.1f%%", c.backtest.win_rate))
                    }
                    Section("Delta") {
                        metric("Δ Trades", c.delta.trade_count)
                        metric("Δ PnL", String(format: "%+.1f", c.delta.net_pnl))
                    }
                }
            }
            .navigationTitle("Live vs BT")
            .toolbar {
                Button("Refresh BT") { Task { await load(refresh: true) } }
            }
            .task { await load(refresh: false) }
        }
    }

    @ViewBuilder
    func metric(_ label: String, _ value: CustomStringConvertible) -> some View {
        LabeledContent(label, value: String(describing: value))
    }

    func load(refresh: Bool) async {
        compare = try? await api.fetchCompare(refresh: refresh)
    }
}

struct SettingsView: View {
    @EnvironmentObject var api: APIClient

    var body: some View {
        Form {
            TextField("Mac Mini URL", text: $api.baseURL)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            SecureField("API Key", text: $api.apiKey)
            Text("Example: http://192.168.1.50:8765")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .navigationTitle("Settings")
    }
}
