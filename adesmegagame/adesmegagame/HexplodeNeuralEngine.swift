import CoreML
import Foundation

final class HexplodeNeuralEngine {
    struct Inference {
        let policyScores: [AxialCoord: Double]
        let value: Double
    }

    private let modelName = "HexplodePolicy"
    private var model: MLModel?

    init() {
        loadModelIfAvailable()
    }

    var isAvailable: Bool {
        model != nil
    }

    func infer(
        owners: [AxialCoord: TileState],
        levels: [AxialCoord: Int],
        turnsPlayed: Int,
        currentPlayer: TileState,
        boardCoords: [AxialCoord],
        radius: Int,
        legalMoves: [AxialCoord]
    ) -> Inference? {
        guard let model else { return nil }
        guard radius == 3 else { return nil } // Large Hexplode board (4-per-side) only.

        let gridSize = radius * 2 + 1
        guard let input = try? MLMultiArray(shape: [1, 8, NSNumber(value: gridSize), NSNumber(value: gridSize)], dataType: .float32) else {
            return nil
        }

        let legalSet = Set(legalMoves)
        for coord in boardCoords {
            let row = coord.r + radius
            let col = coord.q + radius
            guard row >= 0, row < gridSize, col >= 0, col < gridSize else { continue }
            let owner = owners[coord] ?? .empty
            let level = min(6, max(0, levels[coord] ?? 0))
            let levelNorm = Float(level) / 6.0

            set(input, batch: 0, channel: 0, row: row, col: col, width: gridSize, value: owner == currentPlayer ? 1 : 0)
            set(input, batch: 0, channel: 1, row: row, col: col, width: gridSize, value: owner == currentPlayer.opposite ? 1 : 0)
            set(input, batch: 0, channel: 2, row: row, col: col, width: gridSize, value: owner == .empty ? 1 : 0)
            set(input, batch: 0, channel: 3, row: row, col: col, width: gridSize, value: owner == currentPlayer ? levelNorm : 0)
            set(input, batch: 0, channel: 4, row: row, col: col, width: gridSize, value: owner == currentPlayer.opposite ? levelNorm : 0)
            set(input, batch: 0, channel: 5, row: row, col: col, width: gridSize, value: owner == currentPlayer && level >= 5 ? 1 : 0)
            set(input, batch: 0, channel: 6, row: row, col: col, width: gridSize, value: owner == currentPlayer.opposite && level >= 5 ? 1 : 0)
            set(input, batch: 0, channel: 7, row: row, col: col, width: gridSize, value: legalSet.contains(coord) ? 1 : 0)
        }

        let turnScalar = min(1.0, Float(turnsPlayed) / 40.0)
        for coord in boardCoords {
            let row = coord.r + radius
            let col = coord.q + radius
            guard row >= 0, row < gridSize, col >= 0, col < gridSize else { continue }
            let existing = get(input, batch: 0, channel: 7, row: row, col: col, width: gridSize)
            set(input, batch: 0, channel: 7, row: row, col: col, width: gridSize, value: min(1, existing * (0.8 + 0.2 * turnScalar)))
        }

        let provider = try? MLDictionaryFeatureProvider(dictionary: ["input": input])
        guard let provider else { return nil }
        guard let output = try? model.prediction(from: provider) else { return nil }
        guard let policy = multiArray(named: "policy", in: output) ?? firstMultiArray(in: output) else { return nil }
        let value = scalarValue(named: "value", in: output)

        return Inference(
            policyScores: decodePolicy(
                policy: policy,
                boardCoords: boardCoords,
                legalMoves: legalMoves,
                radius: radius
            ),
            value: value
        )
    }

    private func loadModelIfAvailable() {
        guard let compiledURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            model = nil
            return
        }
        model = try? MLModel(contentsOf: compiledURL)
    }

    private func firstMultiArray(in provider: MLFeatureProvider) -> MLMultiArray? {
        if let direct = multiArray(named: "policy", in: provider) {
            return direct
        }
        for name in provider.featureNames {
            if let arr = provider.featureValue(for: name)?.multiArrayValue {
                return arr
            }
        }
        return nil
    }

    private func multiArray(named name: String, in provider: MLFeatureProvider) -> MLMultiArray? {
        provider.featureValue(for: name)?.multiArrayValue
    }

    private func scalarValue(named name: String, in provider: MLFeatureProvider) -> Double {
        if let number = provider.featureValue(for: name)?.doubleValue {
            return number
        }
        if let array = provider.featureValue(for: name)?.multiArrayValue, array.count > 0 {
            return array[0].doubleValue
        }
        return 0
    }

    private func decodePolicy(
        policy: MLMultiArray,
        boardCoords: [AxialCoord],
        legalMoves: [AxialCoord],
        radius: Int
    ) -> [AxialCoord: Double] {
        let count = policy.count
        let legalSet = Set(legalMoves)
        var scores: [AxialCoord: Double] = [:]
        scores.reserveCapacity(legalMoves.count)

        if count == boardCoords.count {
            for (idx, coord) in boardCoords.enumerated() where legalSet.contains(coord) {
                scores[coord] = policy[idx].doubleValue
            }
            return scores
        }

        let gridSize = radius * 2 + 1
        if count == gridSize * gridSize {
            for coord in legalMoves {
                let row = coord.r + radius
                let col = coord.q + radius
                guard row >= 0, row < gridSize, col >= 0, col < gridSize else { continue }
                scores[coord] = policy[row * gridSize + col].doubleValue
            }
            return scores
        }

        for coord in legalMoves {
            scores[coord] = 0
        }
        return scores
    }

    private func flatIndex(batch: Int, channel: Int, row: Int, col: Int, width: Int) -> Int {
        (((batch * 8) + channel) * width + row) * width + col
    }

    private func set(
        _ array: MLMultiArray,
        batch: Int,
        channel: Int,
        row: Int,
        col: Int,
        width: Int,
        value: Float
    ) {
        array[flatIndex(batch: batch, channel: channel, row: row, col: col, width: width)] = NSNumber(value: value)
    }

    private func get(
        _ array: MLMultiArray,
        batch: Int,
        channel: Int,
        row: Int,
        col: Int,
        width: Int
    ) -> Float {
        array[flatIndex(batch: batch, channel: channel, row: row, col: col, width: width)].floatValue
    }
}
