import Foundation

@MainActor
final class APIClient: ObservableObject {
    @Published var baseURL: String {
        didSet { UserDefaults.standard.set(baseURL, forKey: "apiBaseURL") }
    }
    @Published var apiKey: String {
        didSet { UserDefaults.standard.set(apiKey, forKey: "apiKey") }
    }

    init() {
        let saved = UserDefaults.standard.string(forKey: "apiBaseURL") ?? "http://192.168.0.4:8765"
        baseURL = saved.trimmingCharacters(in: .whitespacesAndNewlines)
        apiKey = UserDefaults.standard.string(forKey: "apiKey") ?? ""
    }

    var trimmedBaseURL: String {
        baseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/").union(.whitespacesAndNewlines))
    }

    func testConnection() async -> String {
        var lines = ["URL: \(trimmedBaseURL)", "API key chars: \(apiKey.count)"]
        switch await probeHealth(at: trimmedBaseURL) {
        case .success(let body):
            lines.append("✅ Health OK: \(body)")
        case .failure(let err):
            lines.append("❌ Health failed: \(err)")
            if trimmedBaseURL.contains("192.168.") {
                lines.append("")
                lines.append("192.168.x.x needs Local Network permission.")
                lines.append("Fix A: iPhone Settings → Privacy → Local Network → AlphaGold → ON")
                lines.append("Fix B: Use public IP instead (below)")
                let publicURL = Self.publicURL
                lines.append("Trying public IP \(publicURL) …")
                switch await probeHealth(at: publicURL) {
                case .success(let body):
                    lines.append("✅ Public IP works: \(body)")
                    lines.append("→ Tap 'Use Public IP' and save")
                case .failure(let err2):
                    lines.append("❌ Public IP also failed: \(err2)")
                }
            }
            return lines.joined(separator: "\n")
        }
        do {
            _ = try await fetchStatus()
            lines.append("✅ Auth OK (/api/v1/status)")
        } catch {
            lines.append("❌ Auth failed: \(error.localizedDescription)")
        }
        do {
            let trades = try await fetchTradesToday()
            lines.append("✅ Trades OK: \(trades.trades.count) rows, day \(trades.trading_day ?? "?")")
        } catch {
            lines.append("❌ Trades failed: \(error.localizedDescription)")
        }
        return lines.joined(separator: "\n")
    }

    static let publicURL = "http://123.203.51.164:8765"
    static let homeURL = "http://192.168.0.4:8765"

    private enum ProbeResult {
        case success(String)
        case failure(String)
    }

    private func probeHealth(at base: String) async -> ProbeResult {
        let urlString = base.trimmingCharacters(in: CharacterSet(charactersIn: "/").union(.whitespacesAndNewlines)) + "/api/v1/health"
        guard let url = URL(string: urlString) else {
            return .failure("Invalid URL: \(urlString)")
        }
        var req = URLRequest(url: url)
        req.timeoutInterval = 10
        req.cachePolicy = .reloadIgnoringLocalCacheData
        do {
            let (data, resp) = try await URLSession.shared.data(for: req)
            guard let http = resp as? HTTPURLResponse else {
                return .failure("No HTTP response")
            }
            guard http.statusCode == 200 else {
                return .failure("HTTP \(http.statusCode)")
            }
            return .success(String(data: data, encoding: .utf8) ?? "{}")
        } catch {
            return .failure(APIError.describe(error))
        }
    }

    private func fetchHealthBody() async throws -> String {
        switch await probeHealth(at: trimmedBaseURL) {
        case .success(let body):
            return body
        case .failure(let msg):
            throw APIError.connection(baseURL: trimmedBaseURL, underlying: URLError(.cannotConnectToHost, userInfo: [NSLocalizedDescriptionKey: msg]))
        }
    }

    private func request<T: Decodable>(
        _ path: String,
        refresh: Bool = false,
        timeout: TimeInterval? = nil
    ) async throws -> T {
        var urlString = trimmedBaseURL + path
        if refresh { urlString += urlString.contains("?") ? "&refresh=true" : "?refresh=true" }
        guard let url = URL(string: urlString) else { throw APIError.badURL(urlString) }
        var req = URLRequest(url: url)
        req.timeoutInterval = timeout ?? (refresh ? 180 : 15)
        if !apiKey.isEmpty { req.setValue(apiKey, forHTTPHeaderField: "X-API-Key") }
        let data: Data
        let resp: URLResponse
        do {
            (data, resp) = try await URLSession.shared.data(for: req)
        } catch {
            throw APIError.connection(baseURL: trimmedBaseURL, underlying: error)
        }
        guard let http = resp as? HTTPURLResponse else {
            throw APIError.connection(baseURL: trimmedBaseURL, underlying: URLError(.badServerResponse))
        }
        guard http.statusCode == 200 else {
            throw APIError.httpStatus(http.statusCode, baseURL: trimmedBaseURL)
        }
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw APIError.decode(underlying: error)
        }
    }

    func fetchSignals(minutes: Int = 30) async throws -> SignalsResponse {
        try await request("/api/v1/signals?minutes=\(minutes)")
    }

    func fetchTradesToday() async throws -> TradesTodayResponse {
        try await request("/api/v1/trades/today")
    }

    func fetchCompare(refresh: Bool = false) async throws -> CompareResponse {
        try await request("/api/v1/compare/today", refresh: refresh, timeout: refresh ? 180 : 15)
    }

    func fetchStatus() async throws -> StatusResponse {
        try await request("/api/v1/status")
    }

    func fetchHealth() async throws {
        _ = try await fetchHealthBody()
    }
}

enum APIError: LocalizedError {
    case badURL(String)
    case connection(baseURL: String, underlying: Error)
    case httpStatus(Int, baseURL: String)
    case decode(underlying: Error)

    static func describe(_ error: Error) -> String {
        if let urlError = error as? URLError {
            if urlError.code == .appTransportSecurityRequiresSecureConnection {
                return "URLError 1022: iOS blocked HTTP. Delete app → Clean Build → reinstall v1.0.1."
            }
            return "URLError \(urlError.code.rawValue): \(urlError.localizedDescription)"
        }
        return error.localizedDescription
    }

    var errorDescription: String? {
        switch self {
        case .badURL(let url):
            return "Invalid URL: \(url)"
        case .connection(let baseURL, let underlying):
            let code = (underlying as? URLError)?.code.rawValue
            let codeText = code.map { " (code \($0))" } ?? ""
            let hint: String
            if baseURL.contains("192.168.") {
                hint = "Enable Local Network for this app, or use public IP http://123.203.51.164:8765"
            } else {
                hint = "Check Mac mini is on and port 8765 is forwarded"
            }
            return "Cannot reach \(baseURL)\(codeText).\n\(hint)\n\(Self.describe(underlying))"
        case .httpStatus(let code, let baseURL):
            return "HTTP \(code) from \(baseURL). Check API key in Settings."
        case .decode(let underlying):
            return "Bad response: \(underlying.localizedDescription)"
        }
    }
}
