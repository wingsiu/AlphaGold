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
    var gold_price: Double?
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
    var horizon_deadline: String?
    var gold_price: Double?
}

struct TradeSummary: Codable {
    var trade_count: Int
    var closed_count: Int
    var open_count: Int
    var net_pnl: Double
    var win_rate: Double
    var closed_pnl: Double?
    var unrealized_pnl: Double?
    var trades: [TradeRow]?
}

struct TradingWindow: Codable {
    var in_window: Bool
    var market_status: String?
    var market_open: Bool?
    var time_filter_blocked: Bool?
    var active_sessions: [String]?
    var blocked_reasons: [String]?
    var next_window_start_utc: String?
    var next_window_label: String?
}

struct SignalsResponse: Codable {
    var minutes: Int
    var count: Int
    var signals: [SignalRow]
    var is_fallback: Bool?
    var source: String?
    var requested_minutes: Int?
    var trading_window: TradingWindow?
}

struct TradesTodayResponse: Codable {
    var summary: TradeSummary
    var trades: [TradeRow]
    var trading_day: String?
    var trading_day_start_utc: String?
    var is_fallback: Bool?
    var source: String?
}

struct CompareResponse: Codable {
    var trading_day: String
    var trading_day_start_utc: String?
    var trading_day_window_hkt: String?
    var live: TradeSummary
    var backtest: TradeSummary
    var delta: DeltaSummary
    var is_fallback: Bool?
    var source: String?
    var both_empty: Bool?
    var note: String?

    struct DeltaSummary: Codable {
        var trade_count: Int
        var net_pnl: Double
        var win_rate: Double
    }
}

struct IGAccount: Codable {
    var account_id: String?
    var account_name: String?
    var account_type: String?
    var status: String?
    var currency: String?
    var balance: Double?
    var equity: Double?
    var available: Double?
    var deposit: Double?
    var profit_loss: Double?
    var cached: Bool?
    var stale: Bool?
    var error: String?
}

struct GoldPrice: Codable {
    var close: Double?
    var bid: Double?
    var offer: Double?
    var market_status: String?
    var updated_at_utc: String?
    var cached: Bool?
    var stale: Bool?
    var error: String?
}

struct StatusResponse: Codable {
    var server_time_utc: String
    var trading_day_start_utc: String
    var today: TradeSummary
    var recent_signals_30m: Int
    var trading_day: String?
    var is_fallback: Bool?
    var source: String?
    var recent_signals_display: Int?
    var signals_is_fallback: Bool?
    var open_position: OpenPosition?
    var ig_account: IGAccount?
    var gold: GoldPrice?
}

struct OpenPosition: Codable {
    var open_deal_id: String?
    var open_entry_time: String?
    var open_position_source: String?
    var open_pattern_name: String?
    var open_side: Int?
    var open_tp: Double?
    var open_sl: Double?
    var open_horizon: Int?
    var consecutive_losses: Int?
    var last_pnl: Double?

    var isFlat: Bool {
        open_deal_id == nil && (open_side == nil || open_side == 0)
    }

    var sideLabel: String {
        guard let s = open_side, s != 0 else { return "FLAT" }
        return s > 0 ? "LONG" : "SHORT"
    }
}
