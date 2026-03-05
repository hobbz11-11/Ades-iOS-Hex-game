import CoreML
import Foundation

final class HexplodeNeuralEngine {
    struct Inference {
        let policyScores: [AxialCoord: Double]
        let value: Double
    }

    private let modelName = "HexplodePolicy"
    private var model: MLModel?
    private enum InputLayout {
        case grid(inputName: String, channels: Int, size: Int)
        case graph(inputName: String, nodeCount: Int, featureCount: Int)
    }
    private var inputLayout: InputLayout = .grid(inputName: "input", channels: 8, size: 7)

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
        let boardSet = Set(boardCoords)
        let turnProgress = min(1.0, Float(max(0, turnsPlayed)) / 220.0)

        let provider: MLDictionaryFeatureProvider?
        switch inputLayout {
        case .grid(let inputName, let channels, let size):
            guard channels >= 8 else { return nil }
            guard let input = try? MLMultiArray(
                shape: [1, NSNumber(value: channels), NSNumber(value: size), NSNumber(value: size)],
                dataType: .float32
            ) else {
                return nil
            }
            let legalSet = Set(legalMoves)
            for coord in boardCoords {
                let row = coord.r + radius
                let col = coord.q + radius
                guard row >= 0, row < size, col >= 0, col < size else { continue }
                let owner = owners[coord] ?? .empty
                let level = min(6, max(0, levels[coord] ?? 0))
                let levelNorm = Float(level) / 6.0

                set(input, batch: 0, channel: 0, row: row, col: col, width: size, channels: channels, value: owner == currentPlayer ? 1 : 0)
                set(input, batch: 0, channel: 1, row: row, col: col, width: size, channels: channels, value: owner == currentPlayer.opposite ? 1 : 0)
                set(input, batch: 0, channel: 2, row: row, col: col, width: size, channels: channels, value: owner == .empty ? 1 : 0)
                set(input, batch: 0, channel: 3, row: row, col: col, width: size, channels: channels, value: owner == currentPlayer ? levelNorm : 0)
                set(input, batch: 0, channel: 4, row: row, col: col, width: size, channels: channels, value: owner == currentPlayer.opposite ? levelNorm : 0)
                set(input, batch: 0, channel: 5, row: row, col: col, width: size, channels: channels, value: owner == currentPlayer && level >= 5 ? 1 : 0)
                set(input, batch: 0, channel: 6, row: row, col: col, width: size, channels: channels, value: owner == currentPlayer.opposite && level >= 5 ? 1 : 0)
                set(input, batch: 0, channel: 7, row: row, col: col, width: size, channels: channels, value: legalSet.contains(coord) ? 1 : 0)
                if channels >= 9 {
                    set(input, batch: 0, channel: 8, row: row, col: col, width: size, channels: channels, value: isEdge(coord, radius: radius) ? 1 : 0)
                }
                if channels >= 10 {
                    set(input, batch: 0, channel: 9, row: row, col: col, width: size, channels: channels, value: isCorner(coord, radius: radius) ? 1 : 0)
                }
                if channels >= 11 {
                    set(input, batch: 0, channel: 10, row: row, col: col, width: size, channels: channels, value: neighborDegreeNormalized(coord, boardSet: boardSet))
                }
                if channels >= 12 {
                    set(input, batch: 0, channel: 11, row: row, col: col, width: size, channels: channels, value: turnProgress)
                }
            }
            provider = try? MLDictionaryFeatureProvider(dictionary: [inputName: input])

        case .graph(let inputName, let nodeCount, let featureCount):
            guard nodeCount == boardCoords.count, featureCount >= 8 else { return nil }
            guard let input = try? MLMultiArray(
                shape: [1, NSNumber(value: nodeCount), NSNumber(value: featureCount)],
                dataType: .float32
            ) else {
                return nil
            }
            let legalSet = Set(legalMoves)
            for (node, coord) in boardCoords.enumerated() {
                let owner = owners[coord] ?? .empty
                let level = min(6, max(0, levels[coord] ?? 0))
                let levelNorm = Float(level) / 6.0
                setGraph(input, batch: 0, node: node, feature: 0, nodeCount: nodeCount, featureCount: featureCount, value: owner == currentPlayer ? 1 : 0)
                setGraph(input, batch: 0, node: node, feature: 1, nodeCount: nodeCount, featureCount: featureCount, value: owner == currentPlayer.opposite ? 1 : 0)
                setGraph(input, batch: 0, node: node, feature: 2, nodeCount: nodeCount, featureCount: featureCount, value: owner == .empty ? 1 : 0)
                setGraph(input, batch: 0, node: node, feature: 3, nodeCount: nodeCount, featureCount: featureCount, value: owner == currentPlayer ? levelNorm : 0)
                setGraph(input, batch: 0, node: node, feature: 4, nodeCount: nodeCount, featureCount: featureCount, value: owner == currentPlayer.opposite ? levelNorm : 0)
                setGraph(input, batch: 0, node: node, feature: 5, nodeCount: nodeCount, featureCount: featureCount, value: owner == currentPlayer && level >= 5 ? 1 : 0)
                setGraph(input, batch: 0, node: node, feature: 6, nodeCount: nodeCount, featureCount: featureCount, value: owner == currentPlayer.opposite && level >= 5 ? 1 : 0)
                setGraph(input, batch: 0, node: node, feature: 7, nodeCount: nodeCount, featureCount: featureCount, value: legalSet.contains(coord) ? 1 : 0)
                if featureCount >= 9 {
                    setGraph(input, batch: 0, node: node, feature: 8, nodeCount: nodeCount, featureCount: featureCount, value: isEdge(coord, radius: radius) ? 1 : 0)
                }
                if featureCount >= 10 {
                    setGraph(input, batch: 0, node: node, feature: 9, nodeCount: nodeCount, featureCount: featureCount, value: isCorner(coord, radius: radius) ? 1 : 0)
                }
                if featureCount >= 11 {
                    setGraph(input, batch: 0, node: node, feature: 10, nodeCount: nodeCount, featureCount: featureCount, value: neighborDegreeNormalized(coord, boardSet: boardSet))
                }
                if featureCount >= 12 {
                    setGraph(input, batch: 0, node: node, feature: 11, nodeCount: nodeCount, featureCount: featureCount, value: turnProgress)
                }
            }
            provider = try? MLDictionaryFeatureProvider(dictionary: [inputName: input])
        }

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
        guard let loaded = try? MLModel(contentsOf: compiledURL) else {
            model = nil
            return
        }
        model = loaded
        inputLayout = detectInputLayout(for: loaded)
    }

    private func detectInputLayout(for model: MLModel) -> InputLayout {
        for (name, desc) in model.modelDescription.inputDescriptionsByName {
            guard let constraint = desc.multiArrayConstraint else { continue }
            let shape = constraint.shape.map { $0.intValue }
            if shape.count == 4 {
                let channels = shape.count > 1 ? shape[1] : 8
                let size = shape.count > 3 ? shape[3] : 7
                return .grid(inputName: name, channels: channels, size: size)
            }
            if shape.count == 3 {
                let nodes = shape.count > 1 ? shape[1] : 37
                let features = shape.count > 2 ? shape[2] : 8
                return .graph(inputName: name, nodeCount: nodes, featureCount: features)
            }
        }
        return .grid(inputName: "input", channels: 8, size: 7)
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

    private func flatIndex(batch: Int, channel: Int, row: Int, col: Int, width: Int, channels: Int) -> Int {
        (((batch * channels) + channel) * width + row) * width + col
    }

    private func set(
        _ array: MLMultiArray,
        batch: Int,
        channel: Int,
        row: Int,
        col: Int,
        width: Int,
        channels: Int,
        value: Float
    ) {
        array[flatIndex(batch: batch, channel: channel, row: row, col: col, width: width, channels: channels)] = NSNumber(value: value)
    }

    private func get(
        _ array: MLMultiArray,
        batch: Int,
        channel: Int,
        row: Int,
        col: Int,
        width: Int,
        channels: Int
    ) -> Float {
        array[flatIndex(batch: batch, channel: channel, row: row, col: col, width: width, channels: channels)].floatValue
    }

    private func graphFlatIndex(batch: Int, node: Int, feature: Int, nodeCount: Int, featureCount: Int) -> Int {
        ((batch * nodeCount) + node) * featureCount + feature
    }

    private func setGraph(
        _ array: MLMultiArray,
        batch: Int,
        node: Int,
        feature: Int,
        nodeCount: Int,
        featureCount: Int,
        value: Float
    ) {
        array[graphFlatIndex(batch: batch, node: node, feature: feature, nodeCount: nodeCount, featureCount: featureCount)] = NSNumber(value: value)
    }

    private func isEdge(_ coord: AxialCoord, radius: Int) -> Bool {
        let x = coord.q
        let z = coord.r
        let y = -x - z
        return max(abs(x), max(abs(y), abs(z))) == radius
    }

    private func isCorner(_ coord: AxialCoord, radius: Int) -> Bool {
        return (coord.q == radius && coord.r == 0)
            || (coord.q == 0 && coord.r == radius)
            || (coord.q == -radius && coord.r == radius)
            || (coord.q == -radius && coord.r == 0)
            || (coord.q == 0 && coord.r == -radius)
            || (coord.q == radius && coord.r == -radius)
    }

    private func neighborDegreeNormalized(_ coord: AxialCoord, boardSet: Set<AxialCoord>) -> Float {
        var count = 0
        for dir in HexGrid.directions {
            if boardSet.contains(coord.adding(dir)) {
                count += 1
            }
        }
        return Float(count) / 6.0
    }
}
