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
        baseURL = UserDefaults.standard.string(forKey: "apiBaseURL") ?? "http://192.168.1.100:8765"
        apiKey = UserDefaults.standard.string(forKey: "apiKey") ?? ""
    }

    private func request<T: Decodable>(_ path: String, refresh: Bool = false) async throws -> T {
        var urlString = baseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + path
        if refresh { urlString += urlString.contains("?") ? "&refresh=true" : "?refresh=true" }
        guard let url = URL(string: urlString) else { throw URLError(.badURL) }
        var req = URLRequest(url: url)
        if !apiKey.isEmpty { req.setValue(apiKey, forHTTPHeaderField: "X-API-Key") }
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(T.self, from: data)
    }

    func fetchSignals(minutes: Int = 30) async throws -> SignalsResponse {
        try await request("/api/v1/signals?minutes=\(minutes)")
    }

    func fetchTradesToday() async throws -> TradesTodayResponse {
        try await request("/api/v1/trades/today")
    }

    func fetchCompare(refresh: Bool = false) async throws -> CompareResponse {
        try await request("/api/v1/compare/today", refresh: refresh)
    }

    func fetchStatus() async throws -> StatusResponse {
        try await request("/api/v1/status")
    }
}
