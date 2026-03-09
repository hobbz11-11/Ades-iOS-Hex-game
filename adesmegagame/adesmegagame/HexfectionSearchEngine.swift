import Foundation

enum HexfectionSearchMoveType: Int, Sendable {
    case clone
    case leap
}

struct HexfectionSearchMove: Equatable, Sendable {
    let source: AxialCoord
    let destination: AxialCoord
    let type: HexfectionSearchMoveType
}

struct HexfectionSearchResult: Sendable {
    let bestMove: HexfectionSearchMove?
    let score: Int
    let depthReached: Int
    let principalVariation: [HexfectionSearchMove]
}

final class HexfectionSearchEngine {
    private enum TTBound: UInt8 {
        case exact
        case lower
        case upper
    }

    private struct TTEntry {
        let hash: UInt64
        let depth: Int
        let score: Int
        let bound: TTBound
        let bestMove: Int
    }

    private struct SearchState {
        var owners: [UInt8]
        var currentPlayer: UInt8
        var redCount: Int
        var blueCount: Int
        var emptyCount: Int
        var passCount: Int
        var hash: UInt64
    }

    private struct GeneratedMove {
        let encoded: Int
        let sourceIndex: Int
        let destinationIndex: Int
        let type: HexfectionSearchMoveType
        let captureCount: Int
        let supportCount: Int
        let adjacentEmptyCount: Int
        let centrality: Int
        let destinationRing: Int

        var isClone: Bool {
            type == .clone
        }

        var isForcing: Bool {
            captureCount >= 3 || (isClone && captureCount >= 2)
        }
    }

    private struct Undo {
        let sourceIndex: Int
        let destinationIndex: Int
        let previousPassCount: Int
        let previousPlayer: UInt8
        let previousHash: UInt64
        let flippedIndices: [Int]
    }

    private struct MoveSummary {
        let totalMoves: Int
        let cloneMoves: Int
        let leapMoves: Int
        let bestCapture: Int
        let frontierReach: Int
    }

    private let coords: [AxialCoord]
    private let coordToIndex: [AxialCoord: Int]
    private let neighbors: [[Int]]
    private let cloneTargets: [[Int]]
    private let leapTargets: [[Int]]
    private let sourcesWithinTwo: [[Int]]
    private let ringIndex: [Int]
    private let centrality: [Int]
    private let moveSpace: Int
    private let maxDepth: Int
    private let exactEndgameEmptyThreshold = 12
    private let maxPly = 96
    private let mateScore = 1_000_000
    private let endgameBaseScore = 250_000

    private let zobristPieces: [[UInt64]]
    private let zobristPlayers: [UInt64]
    private let zobristPass: [UInt64]

    private var transpositionTable: [UInt64: TTEntry] = [:]
    private var exactCache: [UInt64: Int] = [:]
    private var historyHeuristic: [[Int]]
    private var killerMoves: [[Int]]
    private var deadline: TimeInterval = 0
    private var timedOut = false
    private var nodeBudget = 0
    private var nodesVisited = 0

    init(boardCoords: [AxialCoord], radius: Int) {
        coords = boardCoords

        var mapping: [AxialCoord: Int] = [:]
        mapping.reserveCapacity(boardCoords.count)
        for (index, coord) in boardCoords.enumerated() {
            mapping[coord] = index
        }
        coordToIndex = mapping

        var computedNeighbors: [[Int]] = Array(repeating: [], count: boardCoords.count)
        var computedCloneTargets: [[Int]] = Array(repeating: [], count: boardCoords.count)
        var computedLeapTargets: [[Int]] = Array(repeating: [], count: boardCoords.count)
        var computedSourcesWithinTwo: [[Int]] = Array(repeating: [], count: boardCoords.count)
        var computedRingIndex: [Int] = Array(repeating: 0, count: boardCoords.count)
        var computedCentrality: [Int] = Array(repeating: 0, count: boardCoords.count)

        for (index, coord) in boardCoords.enumerated() {
            let adjacent = HexGrid.directions.compactMap { direction in
                mapping[coord.adding(direction)]
            }
            computedNeighbors[index] = adjacent

            for (targetIndex, targetCoord) in boardCoords.enumerated() where targetIndex != index {
                let distance = HexfectionSearchEngine.hexDistance(coord, targetCoord)
                if distance == 1 {
                    computedCloneTargets[index].append(targetIndex)
                    computedSourcesWithinTwo[targetIndex].append(index)
                } else if distance == 2 {
                    computedLeapTargets[index].append(targetIndex)
                    computedSourcesWithinTwo[targetIndex].append(index)
                }
            }

            let x = coord.q
            let z = coord.r
            let y = -x - z
            let distance = max(abs(x), max(abs(y), abs(z)))
            computedRingIndex[index] = distance
            computedCentrality[index] = max(0, radius - distance)
        }

        neighbors = computedNeighbors
        cloneTargets = computedCloneTargets
        leapTargets = computedLeapTargets
        sourcesWithinTwo = computedSourcesWithinTwo
        ringIndex = computedRingIndex
        centrality = computedCentrality
        moveSpace = max(1, boardCoords.count * boardCoords.count * 2)

        switch boardCoords.count {
        case ..<40:
            maxDepth = 9
        case ..<70:
            maxDepth = 8
        default:
            maxDepth = 7
        }

        var rng = SplitMix64(seed: 0xC0DE_5EED_4845_5846)
        var pieceKeys: [[UInt64]] = Array(repeating: Array(repeating: 0, count: 3), count: boardCoords.count)
        for index in boardCoords.indices {
            pieceKeys[index][1] = rng.next()
            pieceKeys[index][2] = rng.next()
        }
        zobristPieces = pieceKeys
        zobristPlayers = [0, rng.next(), rng.next()]
        zobristPass = [rng.next(), rng.next()]

        historyHeuristic = Array(repeating: Array(repeating: 0, count: moveSpace), count: 3)
        killerMoves = Array(repeating: Array(repeating: -1, count: 2), count: maxDepth + 12)
    }

    func chooseMove(
        owners: [AxialCoord: TileState],
        currentPlayer: TileState,
        timeBudget: TimeInterval
    ) -> HexfectionSearchResult? {
        guard currentPlayer == .red || currentPlayer == .blue else { return nil }

        var state = makeState(owners: owners, currentPlayer: currentPlayer)
        let legalMoves = generateMoves(in: state)
        guard !legalMoves.isEmpty else { return nil }
        if legalMoves.count == 1, let only = decodeMove(legalMoves[0].encoded) {
            return HexfectionSearchResult(bestMove: only, score: 0, depthReached: 0, principalVariation: [only])
        }

        deadline = Date().timeIntervalSinceReferenceDate + timeBudget
        timedOut = false
        nodesVisited = 0
        nodeBudget = defaultNodeBudget()
        transpositionTable.removeAll(keepingCapacity: true)
        exactCache.removeAll(keepingCapacity: true)
        for ownerIndex in historyHeuristic.indices {
            historyHeuristic[ownerIndex] = Array(repeating: 0, count: moveSpace)
        }
        killerMoves = Array(repeating: Array(repeating: -1, count: 2), count: maxDepth + 12)

        if isExactEndgameCandidate(state: state, moveCount: legalMoves.count),
           let exact = solveExactRoot(state: &state) {
            return exact
        }

        var bestMove = legalMoves[0].encoded
        var bestScore = -mateScore
        var depthReached = 0
        var previousBest: Int? = nil
        var lastScore = 0

        for depth in 1...maxDepth {
            if shouldStop() { break }

            let window = aspirationWindow(center: lastScore, depth: depth)
            if let result = searchRoot(state: &state, depth: depth, alpha: window.lower, beta: window.upper, preferredMove: previousBest) {
                bestMove = result.bestMove
                bestScore = result.score
                depthReached = depth
                previousBest = result.bestMove
                lastScore = result.score
            } else if timedOut {
                break
            } else if let widened = searchRoot(state: &state, depth: depth, alpha: -mateScore, beta: mateScore, preferredMove: previousBest) {
                bestMove = widened.bestMove
                bestScore = widened.score
                depthReached = depth
                previousBest = widened.bestMove
                lastScore = widened.score
            } else {
                break
            }
        }

        guard let resolvedMove = decodeMove(bestMove) else { return nil }
        let principalVariation = extractPrincipalVariation(from: state, maxLength: max(1, depthReached))
        return HexfectionSearchResult(
            bestMove: resolvedMove,
            score: bestScore,
            depthReached: depthReached,
            principalVariation: principalVariation
        )
    }

    private func searchRoot(
        state: inout SearchState,
        depth: Int,
        alpha: Int,
        beta: Int,
        preferredMove: Int?
    ) -> (bestMove: Int, score: Int)? {
        var localAlpha = alpha
        let orderedMoves = orderedMoves(in: state, ply: 0, preferredMove: preferredMove)
        guard !orderedMoves.isEmpty else { return nil }

        var bestMove = orderedMoves[0].encoded
        var bestScore = -mateScore

        for (index, move) in orderedMoves.enumerated() {
            if shouldStop() { return nil }

            let undo = makeMove(move, state: &state)
            let childDepth = max(0, depth - 1)
            let score: Int

            if index == 0 {
                score = -negamax(state: &state, depth: childDepth, alpha: -beta, beta: -localAlpha, ply: 1, allowNullWindow: false)
            } else {
                var candidate = -negamax(state: &state, depth: max(0, childDepth - 1), alpha: -localAlpha - 1, beta: -localAlpha, ply: 1, allowNullWindow: true)
                if candidate > localAlpha && candidate < beta {
                    candidate = -negamax(state: &state, depth: childDepth, alpha: -beta, beta: -localAlpha, ply: 1, allowNullWindow: false)
                }
                score = candidate
            }

            unmakeMove(move, undo: undo, state: &state)
            if timedOut { return nil }

            if score > bestScore {
                bestScore = score
                bestMove = move.encoded
            }
            if score > localAlpha {
                localAlpha = score
            }
        }

        let bound: TTBound
        if bestScore <= alpha {
            bound = .upper
        } else if bestScore >= beta {
            bound = .lower
        } else {
            bound = .exact
        }
        storeTT(hash: state.hash, depth: depth, score: bestScore, bound: bound, bestMove: bestMove)
        return (bestMove, bestScore)
    }

    private func negamax(
        state: inout SearchState,
        depth: Int,
        alpha: Int,
        beta: Int,
        ply: Int,
        allowNullWindow: Bool
    ) -> Int {
        nodesVisited += 1
        if shouldStop() {
            timedOut = true
            return evaluate(state: state)
        }
        if ply >= maxPly || nodesVisited >= nodeBudget {
            return evaluate(state: state)
        }

        if let terminal = terminalScoreIfAny(state: state, ply: ply) {
            return terminal
        }

        let legalMoves = generateMoves(in: state)
        if legalMoves.isEmpty {
            if state.passCount == 1 {
                return finalScore(state: state, ply: ply)
            }
            let undo = applyPass(state: &state)
            let score = -negamax(state: &state, depth: max(0, depth - 1), alpha: -beta, beta: -alpha, ply: ply + 1, allowNullWindow: false)
            undoPass(undo, state: &state)
            return score
        }

        if isExactEndgameCandidate(state: state, moveCount: legalMoves.count) {
            return solveToTerminal(state: &state, alpha: alpha, beta: beta, ply: ply, lineHashes: [])
        }

        if depth <= 0 {
            return quiescence(state: &state, alpha: alpha, beta: beta, ply: ply, depth: 0)
        }

        var localAlpha = alpha
        let alphaOriginal = alpha
        let ttMove = probeTT(hash: state.hash, depth: depth, alpha: &localAlpha, beta: beta)
        if localAlpha >= beta {
            return localAlpha
        }

        let ordered = orderedMoves(in: state, ply: ply, preferredMove: ttMove)
        guard !ordered.isEmpty else {
            return evaluate(state: state)
        }

        var bestScore = -mateScore
        var bestMove = ordered[0].encoded

        for (index, move) in ordered.enumerated() {
            let isLateMove = index >= 3
            let canReduce = allowNullWindow && depth >= 3 && isLateMove && !move.isForcing && !isStrongTacticalMove(move, in: state)
            let reduction = canReduce ? (depth >= 6 ? 2 : 1) : 0

            let undo = makeMove(move, state: &state)
            var candidate: Int

            if index == 0 {
                candidate = -negamax(state: &state, depth: depth - 1, alpha: -beta, beta: -localAlpha, ply: ply + 1, allowNullWindow: true)
            } else {
                candidate = -negamax(
                    state: &state,
                    depth: max(0, depth - 1 - reduction),
                    alpha: -localAlpha - 1,
                    beta: -localAlpha,
                    ply: ply + 1,
                    allowNullWindow: true
                )
                if candidate > localAlpha && candidate < beta {
                    candidate = -negamax(state: &state, depth: depth - 1, alpha: -beta, beta: -localAlpha, ply: ply + 1, allowNullWindow: true)
                }
            }

            unmakeMove(move, undo: undo, state: &state)
            if timedOut {
                return evaluate(state: state)
            }

            if candidate > bestScore {
                bestScore = candidate
                bestMove = move.encoded
            }

            if candidate > localAlpha {
                localAlpha = candidate
            }

            if localAlpha >= beta {
                storeKiller(move.encoded, ply: ply)
                historyHeuristic[Int(state.currentPlayer)][move.encoded] += depth * depth
                storeTT(hash: state.hash, depth: depth, score: localAlpha, bound: .lower, bestMove: move.encoded)
                return localAlpha
            }
        }

        let bound: TTBound = bestScore <= alphaOriginal ? .upper : .exact
        storeTT(hash: state.hash, depth: depth, score: bestScore, bound: bound, bestMove: bestMove)
        return bestScore
    }

    private func quiescence(
        state: inout SearchState,
        alpha: Int,
        beta: Int,
        ply: Int,
        depth: Int
    ) -> Int {
        nodesVisited += 1
        if shouldStop() || ply >= maxPly || depth >= 4 {
            return evaluate(state: state)
        }
        if let terminal = terminalScoreIfAny(state: state, ply: ply) {
            return terminal
        }

        var localAlpha = alpha
        let standPat = evaluate(state: state)
        if standPat >= beta {
            return standPat
        }
        if standPat > localAlpha {
            localAlpha = standPat
        }

        let forcingMoves = orderedMoves(in: state, ply: ply, preferredMove: nil).filter { move in
            if move.captureCount >= 2 {
                return true
            }
            if move.isClone && move.captureCount >= 1 && state.emptyCount <= exactEndgameEmptyThreshold + 4 {
                return true
            }
            return false
        }
        guard !forcingMoves.isEmpty else { return standPat }

        for move in forcingMoves {
            let undo = makeMove(move, state: &state)
            let score = -quiescence(state: &state, alpha: -beta, beta: -localAlpha, ply: ply + 1, depth: depth + 1)
            unmakeMove(move, undo: undo, state: &state)

            if score >= beta {
                return score
            }
            if score > localAlpha {
                localAlpha = score
            }
        }
        return localAlpha
    }

    private func solveExactRoot(state: inout SearchState) -> HexfectionSearchResult? {
        let legalMoves = orderedMoves(in: state, ply: 0, preferredMove: nil)
        guard !legalMoves.isEmpty else { return nil }

        var bestMove = legalMoves[0].encoded
        var bestScore = -mateScore

        for move in legalMoves {
            if shouldStop() { return nil }
            let undo = makeMove(move, state: &state)
            let score = -solveToTerminal(state: &state, alpha: -mateScore, beta: mateScore, ply: 1, lineHashes: [state.hash])
            unmakeMove(move, undo: undo, state: &state)

            if score > bestScore {
                bestScore = score
                bestMove = move.encoded
            }
        }

        guard let resolvedMove = decodeMove(bestMove) else { return nil }
        return HexfectionSearchResult(
            bestMove: resolvedMove,
            score: bestScore,
            depthReached: maxDepth,
            principalVariation: [resolvedMove]
        )
    }

    private func solveToTerminal(
        state: inout SearchState,
        alpha: Int,
        beta: Int,
        ply: Int,
        lineHashes: Set<UInt64>
    ) -> Int {
        nodesVisited += 1
        if shouldStop() || ply >= maxPly || nodesVisited >= nodeBudget {
            return evaluate(state: state)
        }
        if let cached = exactCache[state.hash] {
            return cached
        }
        if lineHashes.contains(state.hash) {
            return evaluate(state: state)
        }
        if let terminal = terminalScoreIfAny(state: state, ply: ply) {
            exactCache[state.hash] = terminal
            return terminal
        }

        let legalMoves = orderedMoves(in: state, ply: ply, preferredMove: nil)
        if legalMoves.isEmpty {
            if state.passCount == 1 {
                let final = finalScore(state: state, ply: ply)
                exactCache[state.hash] = final
                return final
            }
            let undo = applyPass(state: &state)
            var nextLine = lineHashes
            nextLine.insert(state.hash)
            let score = -solveToTerminal(state: &state, alpha: -beta, beta: -alpha, ply: ply + 1, lineHashes: nextLine)
            undoPass(undo, state: &state)
            exactCache[state.hash] = score
            return score
        }

        var localAlpha = alpha
        var best = -mateScore
        var nextLine = lineHashes
        nextLine.insert(state.hash)

        for move in legalMoves {
            let undo = makeMove(move, state: &state)
            let score = -solveToTerminal(state: &state, alpha: -beta, beta: -localAlpha, ply: ply + 1, lineHashes: nextLine)
            unmakeMove(move, undo: undo, state: &state)

            if score > best {
                best = score
            }
            if score > localAlpha {
                localAlpha = score
            }
            if localAlpha >= beta {
                exactCache[state.hash] = localAlpha
                return localAlpha
            }
        }

        exactCache[state.hash] = best
        return best
    }

    private func orderedMoves(
        in state: SearchState,
        ply: Int,
        preferredMove: Int?
    ) -> [GeneratedMove] {
        var moves = generateMoves(in: state)
        if moves.isEmpty {
            return moves
        }

        let opponentPieces = opponentCount(in: state)
        let playerIndex = Int(state.currentPlayer)
        let killerOne = killerMoves.indices.contains(ply) ? killerMoves[ply][0] : -1
        let killerTwo = killerMoves.indices.contains(ply) ? killerMoves[ply][1] : -1

        moves.sort { lhs, rhs in
            let lhsScore = orderingScore(
                lhs,
                preferredMove: preferredMove,
                opponentPieces: opponentPieces,
                playerIndex: playerIndex,
                killerOne: killerOne,
                killerTwo: killerTwo
            )
            let rhsScore = orderingScore(
                rhs,
                preferredMove: preferredMove,
                opponentPieces: opponentPieces,
                playerIndex: playerIndex,
                killerOne: killerOne,
                killerTwo: killerTwo
            )
            if lhsScore != rhsScore {
                return lhsScore > rhsScore
            }
            return lhs.encoded < rhs.encoded
        }
        return moves
    }

    private func generateMoves(in state: SearchState) -> [GeneratedMove] {
        let player = state.currentPlayer
        var moves: [GeneratedMove] = []
        moves.reserveCapacity(128)

        for sourceIndex in coords.indices where state.owners[sourceIndex] == player {
            for destinationIndex in cloneTargets[sourceIndex] where state.owners[destinationIndex] == 0 {
                moves.append(makeGeneratedMove(sourceIndex: sourceIndex, destinationIndex: destinationIndex, type: .clone, player: player, state: state))
            }
            for destinationIndex in leapTargets[sourceIndex] where state.owners[destinationIndex] == 0 {
                moves.append(makeGeneratedMove(sourceIndex: sourceIndex, destinationIndex: destinationIndex, type: .leap, player: player, state: state))
            }
        }

        return moves
    }

    private func makeGeneratedMove(
        sourceIndex: Int,
        destinationIndex: Int,
        type: HexfectionSearchMoveType,
        player: UInt8,
        state: SearchState
    ) -> GeneratedMove {
        var captureCount = 0
        var supportCount = 0
        var adjacentEmptyCount = 0
        let opponent = opponent(of: player)

        for neighbor in neighbors[destinationIndex] {
            let owner = state.owners[neighbor]
            if owner == opponent {
                captureCount += 1
            } else if owner == player {
                supportCount += 1
            } else {
                adjacentEmptyCount += 1
            }
        }

        return GeneratedMove(
            encoded: encodeMove(sourceIndex: sourceIndex, destinationIndex: destinationIndex, type: type),
            sourceIndex: sourceIndex,
            destinationIndex: destinationIndex,
            type: type,
            captureCount: captureCount,
            supportCount: supportCount,
            adjacentEmptyCount: adjacentEmptyCount,
            centrality: centrality[destinationIndex],
            destinationRing: ringIndex[destinationIndex]
        )
    }

    private func makeMove(_ move: GeneratedMove, state: inout SearchState) -> Undo {
        let player = state.currentPlayer
        let opponent = opponent(of: player)
        let previousHash = state.hash
        let previousPassCount = state.passCount
        let previousPlayer = state.currentPlayer
        var flipped: [Int] = []

        updatePassHash(from: state.passCount, to: 0, state: &state)
        state.passCount = 0

        if move.type == .clone {
            applyOwnerChange(at: move.destinationIndex, to: player, state: &state)
        } else {
            applyOwnerChange(at: move.sourceIndex, to: 0, state: &state)
            applyOwnerChange(at: move.destinationIndex, to: player, state: &state)
        }

        for neighbor in neighbors[move.destinationIndex] where state.owners[neighbor] == opponent {
            flipped.append(neighbor)
            applyOwnerChange(at: neighbor, to: player, state: &state)
        }

        updatePlayerHash(from: previousPlayer, to: opponent, state: &state)
        state.currentPlayer = opponent

        return Undo(
            sourceIndex: move.sourceIndex,
            destinationIndex: move.destinationIndex,
            previousPassCount: previousPassCount,
            previousPlayer: previousPlayer,
            previousHash: previousHash,
            flippedIndices: flipped
        )
    }

    private func unmakeMove(_ move: GeneratedMove, undo: Undo, state: inout SearchState) {
        let player = undo.previousPlayer
        let opponent = opponent(of: player)

        state.currentPlayer = player
        state.passCount = undo.previousPassCount

        for neighbor in undo.flippedIndices {
            applyOwnerChange(at: neighbor, to: opponent, state: &state)
        }

        if move.type == .clone {
            applyOwnerChange(at: move.destinationIndex, to: 0, state: &state)
        } else {
            applyOwnerChange(at: move.destinationIndex, to: 0, state: &state)
            applyOwnerChange(at: move.sourceIndex, to: player, state: &state)
        }

        state.hash = undo.previousHash
    }

    private func applyPass(state: inout SearchState) -> Undo {
        let previousHash = state.hash
        let previousPassCount = state.passCount
        let previousPlayer = state.currentPlayer
        let nextPlayer = opponent(of: previousPlayer)

        updatePassHash(from: previousPassCount, to: min(1, previousPassCount + 1), state: &state)
        state.passCount = min(1, previousPassCount + 1)
        updatePlayerHash(from: previousPlayer, to: nextPlayer, state: &state)
        state.currentPlayer = nextPlayer

        return Undo(
            sourceIndex: -1,
            destinationIndex: -1,
            previousPassCount: previousPassCount,
            previousPlayer: previousPlayer,
            previousHash: previousHash,
            flippedIndices: []
        )
    }

    private func undoPass(_ undo: Undo, state: inout SearchState) {
        state.currentPlayer = undo.previousPlayer
        state.passCount = undo.previousPassCount
        state.hash = undo.previousHash
    }

    private func evaluate(state: SearchState) -> Int {
        let player = state.currentPlayer
        let opponent = opponent(of: player)
        let ownCount = player == 1 ? state.redCount : state.blueCount
        let opponentCount = opponent == 1 ? state.redCount : state.blueCount

        if ownCount == 0, opponentCount > 0 {
            return -mateScore / 2
        }
        if opponentCount == 0, ownCount > 0 {
            return mateScore / 2
        }

        let totalPieces = ownCount + opponentCount
        let fillPercent = coords.isEmpty ? 0 : (totalPieces * 100) / coords.count

        var ownCentrality = 0
        var opponentCentrality = 0
        var ownSupport = 0
        var opponentSupport = 0
        var ownFrontier = 0
        var opponentFrontier = 0
        var ownSafeMass = 0
        var opponentSafeMass = 0
        var ownInfluence = 0
        var opponentInfluence = 0
        var contestedInfluence = 0

        for index in coords.indices {
            let owner = state.owners[index]
            if owner == player || owner == opponent {
                var friendlyAdjacent = 0
                var emptyAdjacent = 0
                for neighbor in neighbors[index] {
                    let neighborOwner = state.owners[neighbor]
                    if neighborOwner == owner {
                        friendlyAdjacent += 1
                    } else if neighborOwner == 0 {
                        emptyAdjacent += 1
                    }
                }

                if owner == player {
                    ownCentrality += centrality[index]
                    ownSupport += friendlyAdjacent
                    if emptyAdjacent > 0 {
                        ownFrontier += 1
                    } else {
                        ownSafeMass += 1
                    }
                } else {
                    opponentCentrality += centrality[index]
                    opponentSupport += friendlyAdjacent
                    if emptyAdjacent > 0 {
                        opponentFrontier += 1
                    } else {
                        opponentSafeMass += 1
                    }
                }
                continue
            }

            var ownCanReach = false
            var opponentCanReach = false
            for source in sourcesWithinTwo[index] {
                let sourceOwner = state.owners[source]
                if sourceOwner == player {
                    ownCanReach = true
                } else if sourceOwner == opponent {
                    opponentCanReach = true
                }
                if ownCanReach && opponentCanReach {
                    break
                }
            }
            if ownCanReach, opponentCanReach {
                contestedInfluence += 1
            } else if ownCanReach {
                ownInfluence += 1
            } else if opponentCanReach {
                opponentInfluence += 1
            }
        }

        let ownSummary = moveSummary(for: player, in: state)
        let opponentSummary = moveSummary(for: opponent, in: state)

        var score = 0
        score += (ownCount - opponentCount) * (18 + (fillPercent / 6))
        score += (ownSummary.totalMoves - opponentSummary.totalMoves) * 4
        score += (ownSummary.cloneMoves - opponentSummary.cloneMoves) * 2
        score += (ownSummary.bestCapture - opponentSummary.bestCapture) * 20
        score += (ownCentrality - opponentCentrality) * 5
        score += (ownSupport - opponentSupport) * 3
        score += (ownSafeMass - opponentSafeMass) * 9
        score -= (ownFrontier - opponentFrontier) * 5
        score += (ownInfluence - opponentInfluence) * 4
        score += contestedInfluence
        score += (ownSummary.frontierReach - opponentSummary.frontierReach) * 2

        if opponentSummary.totalMoves == 0 {
            score += 450 + (ownCount - opponentCount) * 12
        }
        if ownSummary.totalMoves == 0 {
            score -= 450 + (opponentCount - ownCount) * 12
        }

        return score
    }

    private func moveSummary(for player: UInt8, in state: SearchState) -> MoveSummary {
        var totalMoves = 0
        var cloneMoves = 0
        var leapMoves = 0
        var bestCapture = 0
        var frontierReach = 0

        for sourceIndex in coords.indices where state.owners[sourceIndex] == player {
            for destinationIndex in cloneTargets[sourceIndex] where state.owners[destinationIndex] == 0 {
                totalMoves += 1
                cloneMoves += 1
                bestCapture = max(bestCapture, captureCount(at: destinationIndex, for: player, in: state))
                if ringIndex[destinationIndex] <= 1 {
                    frontierReach += 1
                }
            }
            for destinationIndex in leapTargets[sourceIndex] where state.owners[destinationIndex] == 0 {
                totalMoves += 1
                leapMoves += 1
                bestCapture = max(bestCapture, captureCount(at: destinationIndex, for: player, in: state))
                if ringIndex[destinationIndex] <= 1 {
                    frontierReach += 1
                }
            }
        }

        return MoveSummary(
            totalMoves: totalMoves,
            cloneMoves: cloneMoves,
            leapMoves: leapMoves,
            bestCapture: bestCapture,
            frontierReach: frontierReach
        )
    }

    private func terminalScoreIfAny(state: SearchState, ply: Int) -> Int? {
        let player = state.currentPlayer
        let opponent = opponent(of: player)
        let ownCount = player == 1 ? state.redCount : state.blueCount
        let opponentCount = opponent == 1 ? state.redCount : state.blueCount

        if ownCount == 0, opponentCount > 0 {
            return -mateScore + ply
        }
        if opponentCount == 0, ownCount > 0 {
            return mateScore - ply
        }
        return nil
    }

    private func finalScore(state: SearchState, ply: Int) -> Int {
        let player = state.currentPlayer
        let opponent = opponent(of: player)
        let ownCount = player == 1 ? state.redCount : state.blueCount
        let opponentCount = opponent == 1 ? state.redCount : state.blueCount
        let diff = ownCount - opponentCount

        if diff == 0 {
            return 0
        }
        if diff > 0 {
            return endgameBaseScore + diff * 100 - ply
        }
        return -endgameBaseScore + diff * 100 + ply
    }

    private func isExactEndgameCandidate(state: SearchState, moveCount: Int) -> Bool {
        state.emptyCount <= exactEndgameEmptyThreshold || moveCount <= 4
    }

    private func isStrongTacticalMove(_ move: GeneratedMove, in state: SearchState) -> Bool {
        if move.captureCount >= 3 {
            return true
        }
        let opponentPieces = opponentCount(in: state)
        return move.captureCount >= opponentPieces && opponentPieces > 0
    }

    private func extractPrincipalVariation(from rootState: SearchState, maxLength: Int) -> [HexfectionSearchMove] {
        var state = rootState
        var seen: Set<UInt64> = []
        var line: [HexfectionSearchMove] = []

        for _ in 0..<maxLength {
            guard !seen.contains(state.hash) else { break }
            seen.insert(state.hash)
            guard let entry = transpositionTable[state.hash],
                  let move = decodeMove(entry.bestMove),
                  let generated = generatedMove(from: entry.bestMove, in: state) else {
                break
            }
            line.append(move)
            let undo = makeMove(generated, state: &state)
            _ = undo
        }

        return line
    }

    private func generatedMove(from encoded: Int, in state: SearchState) -> GeneratedMove? {
        let decoded = decodeMoveComponents(encoded)
        guard decoded.sourceIndex >= 0,
              decoded.sourceIndex < coords.count,
              decoded.destinationIndex >= 0,
              decoded.destinationIndex < coords.count else {
            return nil
        }
        guard state.owners[decoded.sourceIndex] == state.currentPlayer,
              state.owners[decoded.destinationIndex] == 0 else {
            return nil
        }
        let legalTargets = decoded.type == .clone ? cloneTargets[decoded.sourceIndex] : leapTargets[decoded.sourceIndex]
        guard legalTargets.contains(decoded.destinationIndex) else {
            return nil
        }
        return makeGeneratedMove(
            sourceIndex: decoded.sourceIndex,
            destinationIndex: decoded.destinationIndex,
            type: decoded.type,
            player: state.currentPlayer,
            state: state
        )
    }

    private func makeState(
        owners: [AxialCoord: TileState],
        currentPlayer: TileState
    ) -> SearchState {
        var ownerCodes: [UInt8] = Array(repeating: 0, count: coords.count)
        var redCount = 0
        var blueCount = 0
        var hash: UInt64 = 0

        for (index, coord) in coords.enumerated() {
            let owner = owners[coord] ?? .empty
            let code = ownerCode(owner)
            ownerCodes[index] = code
            if code == 1 {
                redCount += 1
            } else if code == 2 {
                blueCount += 1
            }
            hash ^= zobristPieces[index][Int(code)]
        }

        let playerCode = ownerCode(currentPlayer)
        hash ^= zobristPlayers[Int(playerCode)]
        hash ^= zobristPass[0]

        return SearchState(
            owners: ownerCodes,
            currentPlayer: playerCode,
            redCount: redCount,
            blueCount: blueCount,
            emptyCount: coords.count - redCount - blueCount,
            passCount: 0,
            hash: hash
        )
    }

    private func orderingScore(
        _ move: GeneratedMove,
        preferredMove: Int?,
        opponentPieces: Int,
        playerIndex: Int,
        killerOne: Int,
        killerTwo: Int
    ) -> Int {
        var score = 0
        if move.encoded == preferredMove {
            score += 5_000_000
        }
        if move.captureCount >= opponentPieces, opponentPieces > 0 {
            score += 2_000_000
        }
        if move.encoded == killerOne {
            score += 800_000
        } else if move.encoded == killerTwo {
            score += 700_000
        }
        score += historyHeuristic[playerIndex][move.encoded] * 4
        score += move.captureCount * 16_000
        score += move.isClone ? 5_000 : 0
        score += move.supportCount * 900
        score += move.centrality * 750
        score += max(0, 3 - move.destinationRing) * 500
        score += move.adjacentEmptyCount * 120
        if !move.isClone && move.captureCount == 0 {
            score -= 2_500
        }
        return score
    }

    private func probeTT(hash: UInt64, depth: Int, alpha: inout Int, beta: Int) -> Int? {
        guard let entry = transpositionTable[hash], entry.hash == hash, entry.depth >= depth else {
            return nil
        }
        switch entry.bound {
        case .exact:
            alpha = entry.score
            return entry.bestMove
        case .lower:
            if entry.score > alpha {
                alpha = entry.score
            }
            return entry.bestMove
        case .upper:
            if entry.score <= alpha {
                alpha = entry.score
            }
            return entry.bestMove
        }
    }

    private func storeTT(hash: UInt64, depth: Int, score: Int, bound: TTBound, bestMove: Int) {
        if let existing = transpositionTable[hash], existing.depth > depth {
            return
        }
        transpositionTable[hash] = TTEntry(hash: hash, depth: depth, score: score, bound: bound, bestMove: bestMove)
    }

    private func storeKiller(_ encoded: Int, ply: Int) {
        guard killerMoves.indices.contains(ply) else { return }
        if killerMoves[ply][0] == encoded {
            return
        }
        killerMoves[ply][1] = killerMoves[ply][0]
        killerMoves[ply][0] = encoded
    }

    private func captureCount(at destinationIndex: Int, for player: UInt8, in state: SearchState) -> Int {
        let opponent = opponent(of: player)
        var count = 0
        for neighbor in neighbors[destinationIndex] where state.owners[neighbor] == opponent {
            count += 1
        }
        return count
    }

    private func opponentCount(in state: SearchState) -> Int {
        let opponent = opponent(of: state.currentPlayer)
        return opponent == 1 ? state.redCount : state.blueCount
    }

    private func applyOwnerChange(at index: Int, to newOwner: UInt8, state: inout SearchState) {
        let oldOwner = state.owners[index]
        guard oldOwner != newOwner else { return }

        state.hash ^= zobristPieces[index][Int(oldOwner)]
        state.hash ^= zobristPieces[index][Int(newOwner)]

        if oldOwner == 1 {
            state.redCount -= 1
        } else if oldOwner == 2 {
            state.blueCount -= 1
        } else {
            state.emptyCount -= 1
        }

        if newOwner == 1 {
            state.redCount += 1
        } else if newOwner == 2 {
            state.blueCount += 1
        } else {
            state.emptyCount += 1
        }

        state.owners[index] = newOwner
    }

    private func updatePlayerHash(from oldPlayer: UInt8, to newPlayer: UInt8, state: inout SearchState) {
        state.hash ^= zobristPlayers[Int(oldPlayer)]
        state.hash ^= zobristPlayers[Int(newPlayer)]
    }

    private func updatePassHash(from oldPass: Int, to newPass: Int, state: inout SearchState) {
        state.hash ^= zobristPass[min(max(0, oldPass), 1)]
        state.hash ^= zobristPass[min(max(0, newPass), 1)]
    }

    private func aspirationWindow(center: Int, depth: Int) -> (lower: Int, upper: Int) {
        let window = max(48, 80 * depth)
        return (center - window, center + window)
    }

    private func encodeMove(sourceIndex: Int, destinationIndex: Int, type: HexfectionSearchMoveType) -> Int {
        ((sourceIndex * coords.count) + destinationIndex) * 2 + type.rawValue
    }

    private func decodeMoveComponents(_ encoded: Int) -> (sourceIndex: Int, destinationIndex: Int, type: HexfectionSearchMoveType) {
        let typeValue = encoded % 2
        let packed = encoded / 2
        let destinationIndex = packed % coords.count
        let sourceIndex = packed / coords.count
        return (sourceIndex, destinationIndex, typeValue == 0 ? .clone : .leap)
    }

    private func decodeMove(_ encoded: Int) -> HexfectionSearchMove? {
        let decoded = decodeMoveComponents(encoded)
        guard coords.indices.contains(decoded.sourceIndex),
              coords.indices.contains(decoded.destinationIndex) else {
            return nil
        }
        return HexfectionSearchMove(
            source: coords[decoded.sourceIndex],
            destination: coords[decoded.destinationIndex],
            type: decoded.type
        )
    }

    private func defaultNodeBudget() -> Int {
        switch coords.count {
        case ..<40:
            return 220_000
        case ..<70:
            return 320_000
        default:
            return 420_000
        }
    }

    private func shouldStop() -> Bool {
        if timedOut {
            return true
        }
        if Date().timeIntervalSinceReferenceDate >= deadline {
            timedOut = true
            return true
        }
        return false
    }

    private func ownerCode(_ owner: TileState) -> UInt8 {
        switch owner {
        case .empty:
            return 0
        case .red:
            return 1
        case .blue:
            return 2
        }
    }

    private func opponent(of player: UInt8) -> UInt8 {
        player == 1 ? 2 : 1
    }

    private static func hexDistance(_ lhs: AxialCoord, _ rhs: AxialCoord) -> Int {
        let dx = lhs.q - rhs.q
        let dz = lhs.r - rhs.r
        let dy = -dx - dz
        return max(abs(dx), max(abs(dy), abs(dz)))
    }
}

private struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
