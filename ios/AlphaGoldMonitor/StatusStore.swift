import Foundation

@MainActor
final class StatusStore: ObservableObject {
    @Published var status: StatusResponse?
    @Published var isLoading = false
    @Published var lastError: String?

    func runPolling(api: APIClient) async {
        while !Task.isCancelled {
            await refresh(api: api)
            try? await Task.sleep(nanoseconds: 30_000_000_000)
        }
    }

    func refresh(api: APIClient) async {
        isLoading = status == nil
        defer { isLoading = false }
        do {
            status = try await api.fetchStatus()
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
    }
}
