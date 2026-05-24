import Foundation

struct SignalRow: Codable, Identifiable {
    var id: Int
    var bar_time: String
    var pattern_name: String?
    var pattern_side: Int?
    var pattern_prob: Double?
    var energetic_side: Int?
    var energetic_prob: Double?
    var action: String
    var detail: String?
    var open_source: String?
    var created_at: String
}

struct TradeRow: Codable, Identifiable {
    var id: Int
    var deal_id: String?
    var source: String?
    var pattern_name: String?
    var side: Int?
    var entry_time: String?
    var exit_time: String?
    var entry_price: Double?
    var exit_price: Double?
    var pnl: Double?
    var exit_reason: String?
    var status: String
}

struct TradeSummary: Codable {
    var trade_count: Int
    var closed_count: Int
    var open_count: Int
    var net_pnl: Double
    var win_rate: Double
}

struct SignalsResponse: Codable {
    var minutes: Int
    var count: Int
    var signals: [SignalRow]
}

struct TradesTodayResponse: Codable {
    var summary: TradeSummary
    var trades: [TradeRow]
}

struct CompareResponse: Codable {
    var trading_day: String
    var live: TradeSummary
    var backtest: TradeSummary
    var delta: DeltaSummary

    struct DeltaSummary: Codable {
        var trade_count: Int
        var net_pnl: Double
        var win_rate: Double
    }
}

struct StatusResponse: Codable {
    var server_time_utc: String
    var trading_day_start_utc: String
    var today: TradeSummary
    var recent_signals_30m: Int
}
