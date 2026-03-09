import Foundation

struct HexplodeSearchResult {
    let bestMove: AxialCoord?
    let score: Int
    let depthReached: Int
    let principalVariation: [AxialCoord]
}

final class HexplodeSearchEngine {
    private enum TileClass: UInt8 {
        case corner
        case edge
        case interior
    }

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

    private struct MovePreview {
        let move: Int
        let owner: UInt8
        let level: Int
        let tileClass: TileClass
        let centrality: Int
        let friendlyAdjacent: Int
        let enemyAdjacent: Int
        let friendlyMaxLevel: Int
        let enemyMaxLevel: Int
        let friendlyNearCritical: Int
        let enemyNearCritical: Int
        let enemyCritical: Int
        let explodes: Bool
        let touchesEnemyThreat: Bool
        let touchesFriendlyThreat: Bool
        let isolatedEmptyBorder: Bool
        let unsupportedContest: Bool

        var isForcing: Bool {
            explodes || touchesEnemyThreat || touchesFriendlyThreat
        }
    }

    private struct SearchState {
        var owners: [UInt8]
        var levels: [Int]
        var turnsPlayed: Int
        var redTiles: Int
        var blueTiles: Int
        var redBalls: Int
        var blueBalls: Int
        var redHadTile: Bool
        var blueHadTile: Bool
        var hash: UInt64
    }

    private let coords: [AxialCoord]
    private let coordToIndex: [AxialCoord: Int]
    private let neighbors: [[Int]]
    private let tileClasses: [TileClass]
    private let centrality: [Int]
    private let radius: Int
    private let maxDepth: Int

    private var deadline: TimeInterval = 0
    private var timedOut = false
    private var transpositionTable: [UInt64: TTEntry] = [:]
    private var historyHeuristic: [[Int]]
    private var killerMoves: [[Int]]

    init(boardCoords: [AxialCoord], radius: Int) {
        self.coords = boardCoords
        self.radius = radius

        var mapping: [AxialCoord: Int] = [:]
        mapping.reserveCapacity(boardCoords.count)
        for (index, coord) in boardCoords.enumerated() {
            mapping[coord] = index
        }
        coordToIndex = mapping

        var computedNeighbors: [[Int]] = Array(repeating: [], count: boardCoords.count)
        var computedClasses: [TileClass] = Array(repeating: .interior, count: boardCoords.count)
        var computedCentrality: [Int] = Array(repeating: 0, count: boardCoords.count)

        for (index, coord) in boardCoords.enumerated() {
            let adjacent = HexGrid.directions.compactMap { direction -> Int? in
                mapping[coord.adding(direction)]
            }
            computedNeighbors[index] = adjacent
            if adjacent.count <= 3 {
                computedClasses[index] = .corner
            } else if adjacent.count <= 4 {
                computedClasses[index] = .edge
            } else {
                computedClasses[index] = .interior
            }
            let x = coord.q
            let z = coord.r
            let y = -x - z
            let distance = max(abs(x), max(abs(y), abs(z)))
            computedCentrality[index] = max(0, radius - distance)
        }

        neighbors = computedNeighbors
        tileClasses = computedClasses
        centrality = computedCentrality

        historyHeuristic = Array(repeating: Array(repeating: 0, count: boardCoords.count), count: 3)
        killerMoves = Array(repeating: Array(repeating: -1, count: 2), count: 64)

        switch boardCoords.count {
        case ..<20:
            maxDepth = 14
        case ..<40:
            maxDepth = 12
        default:
            maxDepth = 11
        }
    }

    func chooseMove(
        owners: [AxialCoord: TileState],
        levels: [AxialCoord: Int],
        turnsPlayed: Int,
        currentPlayer: TileState,
        redHadTile: Bool,
        blueHadTile: Bool,
        timeBudget: TimeInterval
    ) -> HexplodeSearchResult? {
        guard currentPlayer != .empty else { return nil }

        let state = makeState(
            owners: owners,
            levels: levels,
            turnsPlayed: turnsPlayed,
            redHadTile: redHadTile,
            blueHadTile: blueHadTile
        )
        let player = ownerCode(currentPlayer)
        let legalMoves = legalMoves(for: player, in: state)
        guard !legalMoves.isEmpty else { return nil }
        if legalMoves.count == 1 {
            return HexplodeSearchResult(
                bestMove: coords[legalMoves[0]],
                score: 0,
                depthReached: 0,
                principalVariation: [coords[legalMoves[0]]]
            )
        }

        deadline = Date().timeIntervalSinceReferenceDate + timeBudget
        timedOut = false
        transpositionTable.removeAll(keepingCapacity: true)
        for playerIndex in historyHeuristic.indices {
            historyHeuristic[playerIndex] = Array(repeating: 0, count: coords.count)
        }
        killerMoves = Array(repeating: Array(repeating: -1, count: 2), count: max(64, maxDepth + 8))

        if isExactEndgameCandidate(state: state, legalMoveCount: legalMoves.count),
           let exact = solveExactRoot(state: state, player: player) {
            return exact
        }

        var bestMove = legalMoves[0]
        var bestScore = -mateScore
        var depthReached = 0
        var lastScore = 0
        var principalVariation: [Int] = [bestMove]

        for depth in 1...maxDepth {
            if isTimedOut() { break }
            let aspiration = aspirationWindow(center: lastScore, depth: depth)
            if let result = searchRoot(
                state: state,
                player: player,
                depth: depth,
                alpha: aspiration.lower,
                beta: aspiration.upper
            ) {
                bestMove = result.bestMove
                bestScore = result.score
                depthReached = depth
                lastScore = result.score
                principalVariation = result.principalVariation
            } else {
                if timedOut {
                    break
                }
                if let widened = searchRoot(
                    state: state,
                    player: player,
                    depth: depth,
                    alpha: -mateScore,
                    beta: mateScore
                ) {
                    bestMove = widened.bestMove
                    bestScore = widened.score
                    depthReached = depth
                    lastScore = widened.score
                    principalVariation = widened.principalVariation
                } else {
                    break
                }
            }
        }

        return HexplodeSearchResult(
            bestMove: coords[bestMove],
            score: bestScore,
            depthReached: depthReached,
            principalVariation: principalVariation.map { coords[$0] }
        )
    }

    private func searchRoot(
        state: SearchState,
        player: UInt8,
        depth: Int,
        alpha: Int,
        beta: Int
    ) -> (bestMove: Int, score: Int, principalVariation: [Int])? {
        var localAlpha = alpha
        let ordered = orderedMoves(for: player, in: state, ply: 0)
        guard !ordered.isEmpty else { return nil }

        var bestMove = ordered[0].move
        var bestScore = -mateScore

        for (index, preview) in ordered.enumerated() {
            if isTimedOut() { return nil }
            let next = applyMove(preview.move, for: player, in: state)
            let fullDepth = max(0, depth - 1)
            var score: Int

            if index == 0 {
                score = -negamax(
                    state: next,
                    player: opponent(of: player),
                    depth: fullDepth,
                    alpha: -beta,
                    beta: -localAlpha,
                    ply: 1
                )
            } else {
                score = -negamax(
                    state: next,
                    player: opponent(of: player),
                    depth: max(0, fullDepth - 1),
                    alpha: -localAlpha - 1,
                    beta: -localAlpha,
                    ply: 1
                )
                if timedOut { return nil }
                if score > localAlpha {
                    score = -negamax(
                        state: next,
                        player: opponent(of: player),
                        depth: fullDepth,
                        alpha: -beta,
                        beta: -localAlpha,
                        ply: 1
                    )
                }
            }

            if timedOut { return nil }
            if score > bestScore {
                bestScore = score
                bestMove = preview.move
            }
            if score > localAlpha {
                localAlpha = score
            }
            if localAlpha >= beta {
                recordKillerMove(preview.move, ply: 0)
                break
            }
        }

        if bestScore <= alpha || bestScore >= beta {
            return nil
        }
        let pv = principalVariation(from: state, player: player, maxLength: depth + 2)
        return (bestMove, bestScore, pv.isEmpty ? [bestMove] : pv)
    }

    private func negamax(
        state: SearchState,
        player: UInt8,
        depth: Int,
        alpha: Int,
        beta: Int,
        ply: Int
    ) -> Int {
        if isTimedOut() { return evaluate(state: state, for: player, ply: ply) }
        if ply >= 64 {
            return evaluate(state: state, for: player, ply: ply)
        }
        if let terminal = terminalScore(state: state, for: player, ply: ply) {
            return terminal
        }
        if depth <= 0 {
            return quiescence(state: state, player: player, alpha: alpha, beta: beta, ply: ply, remaining: 2)
        }

        let ttKey = hashKey(for: player, state: state)
        let originalAlpha = alpha
        if let entry = transpositionTable[ttKey], entry.hash == ttKey, entry.depth >= depth {
            switch entry.bound {
            case .exact:
                return entry.score
            case .lower where entry.score >= beta:
                return entry.score
            case .upper where entry.score <= alpha:
                return entry.score
            default:
                break
            }
        }

        let ordered = orderedMoves(for: player, in: state, ply: ply)
        if ordered.isEmpty {
            return evaluate(state: state, for: player, ply: ply)
        }

        var localAlpha = alpha
        var bestScore = -mateScore
        var bestMove = ordered[0].move

        for (index, preview) in ordered.enumerated() {
            if isTimedOut() { return evaluate(state: state, for: player, ply: ply) }
            let next = applyMove(preview.move, for: player, in: state)
            let reducedDepth = reducedSearchDepth(
                fullDepth: depth,
                moveIndex: index,
                preview: preview,
                ply: ply
            )
            let searchDepth = max(0, depth - 1 - reducedDepth)
            let nextPlayer = opponent(of: player)

            var score: Int
            if index == 0 {
                score = -negamax(
                    state: next,
                    player: nextPlayer,
                    depth: searchDepth,
                    alpha: -beta,
                    beta: -localAlpha,
                    ply: ply + 1
                )
            } else {
                score = -negamax(
                    state: next,
                    player: nextPlayer,
                    depth: max(0, searchDepth - 1),
                    alpha: -localAlpha - 1,
                    beta: -localAlpha,
                    ply: ply + 1
                )
                if score > localAlpha {
                    score = -negamax(
                        state: next,
                        player: nextPlayer,
                        depth: searchDepth,
                        alpha: -beta,
                        beta: -localAlpha,
                        ply: ply + 1
                    )
                }
            }

            if score > bestScore {
                bestScore = score
                bestMove = preview.move
            }
            if score > localAlpha {
                localAlpha = score
                if !preview.isForcing {
                    let playerIndex = Int(player)
                    historyHeuristic[playerIndex][preview.move] += depth * depth
                }
            }
            if localAlpha >= beta {
                recordKillerMove(preview.move, ply: ply)
                break
            }
        }

        let bound: TTBound
        if bestScore <= originalAlpha {
            bound = .upper
        } else if bestScore >= beta {
            bound = .lower
        } else {
            bound = .exact
        }
        transpositionTable[ttKey] = TTEntry(
            hash: ttKey,
            depth: depth,
            score: bestScore,
            bound: bound,
            bestMove: bestMove
        )
        return bestScore
    }

    private func quiescence(
        state: SearchState,
        player: UInt8,
        alpha: Int,
        beta: Int,
        ply: Int,
        remaining: Int
    ) -> Int {
        if isTimedOut() { return evaluate(state: state, for: player, ply: ply) }
        if let terminal = terminalScore(state: state, for: player, ply: ply) {
            return terminal
        }

        let standPat = evaluate(state: state, for: player, ply: ply)
        if standPat >= beta {
            return standPat
        }

        var localAlpha = max(alpha, standPat)
        guard remaining > 0 else { return localAlpha }

        let forcing = orderedMoves(for: player, in: state, ply: ply).filter(\.isForcing)
        guard !forcing.isEmpty else { return localAlpha }

        for preview in forcing {
            if isTimedOut() { break }
            let next = applyMove(preview.move, for: player, in: state)
            let score = -quiescence(
                state: next,
                player: opponent(of: player),
                alpha: -beta,
                beta: -localAlpha,
                ply: ply + 1,
                remaining: remaining - 1
            )
            if score >= beta {
                return score
            }
            if score > localAlpha {
                localAlpha = score
            }
        }

        return localAlpha
    }

    private func orderedMoves(for player: UInt8, in state: SearchState, ply: Int) -> [MovePreview] {
        let ttMove = transpositionTable[hashKey(for: player, state: state)]?.bestMove
        let legal = legalMoves(for: player, in: state)
        let playerIndex = Int(player)
        let maxPly = min(ply, killerMoves.count - 1)
        let earlyPhase = isEarlyPhase(state)

        return legal
            .map { makeMovePreview(move: $0, player: player, state: state) }
            .sorted { lhs, rhs in
                let lhsScore = moveOrderingScore(
                    preview: lhs,
                    ttMove: ttMove,
                    playerIndex: playerIndex,
                    ply: maxPly,
                    player: player,
                    earlyPhase: earlyPhase
                )
                let rhsScore = moveOrderingScore(
                    preview: rhs,
                    ttMove: ttMove,
                    playerIndex: playerIndex,
                    ply: maxPly,
                    player: player,
                    earlyPhase: earlyPhase
                )
                if lhsScore != rhsScore {
                    return lhsScore > rhsScore
                }
                let lhsCoord = coords[lhs.move]
                let rhsCoord = coords[rhs.move]
                if lhsCoord.q != rhsCoord.q { return lhsCoord.q < rhsCoord.q }
                return lhsCoord.r < rhsCoord.r
            }
    }

    private func moveOrderingScore(
        preview: MovePreview,
        ttMove: Int?,
        playerIndex: Int,
        ply: Int,
        player: UInt8,
        earlyPhase: Bool
    ) -> Int {
        var score = 0
        if ttMove == preview.move { score += 2_000_000 }
        if killerMoves[ply][0] == preview.move { score += 1_200_000 }
        if killerMoves[ply][1] == preview.move { score += 1_000_000 }
        score += historyHeuristic[playerIndex][preview.move]
        if preview.explodes { score += 400_000 }
        score += preview.enemyCritical * 90_000
        score += preview.enemyNearCritical * 35_000
        score += preview.friendlyNearCritical * 16_000
        score += preview.enemyAdjacent * 5_000
        score += preview.friendlyAdjacent * 2_500
        score += preview.centrality * 400
        score += preview.level * 250
        switch preview.tileClass {
        case .interior:
            score += 600
        case .edge:
            score -= 250
        case .corner:
            score -= 550
        }
        if preview.isolatedEmptyBorder {
            score -= 20_000
        }
        if earlyPhase {
            let afterLevel = preview.level + 1
            if preview.owner == player {
                score += afterLevel * afterLevel * 1_800
                score += max(0, afterLevel - 2) * (11_000 + (preview.centrality * 2_200))
                score += preview.friendlyAdjacent * 5_500
                if afterLevel >= 4 && preview.tileClass == .interior {
                    score += 14_000
                }
                let tempoAfter = afterLevel - preview.enemyMaxLevel
                score += tempoAfter * 8_000
                if preview.enemyAdjacent > 0 && tempoAfter <= 0 {
                    score -= (1 - tempoAfter) * 14_000
                }
                if preview.enemyAdjacent == 0 && preview.level >= 2 {
                    score += 12_000
                }
            } else if preview.owner == emptyCode {
                if preview.friendlyAdjacent == 0 {
                    score -= 20_000
                } else {
                    score += preview.friendlyAdjacent * 3_500
                    score += preview.friendlyMaxLevel * 2_600
                    if preview.friendlyMaxLevel >= 3 {
                        score += 9_000 + (preview.friendlyAdjacent * 1_800)
                    }
                }
                if preview.unsupportedContest {
                    score -= 38_000
                } else if preview.enemyAdjacent > preview.friendlyAdjacent {
                    score -= (preview.enemyAdjacent - preview.friendlyAdjacent) * 7_000
                }
                if preview.enemyAdjacent > 0 && preview.friendlyMaxLevel < preview.enemyMaxLevel {
                    score -= (preview.enemyMaxLevel - preview.friendlyMaxLevel) * 10_000
                }
                if preview.tileClass != .interior && preview.friendlyMaxLevel < 3 && preview.enemyAdjacent == 0 {
                    score -= 7_000
                }
            }
        }
        return score
    }

    private func reducedSearchDepth(fullDepth: Int, moveIndex: Int, preview: MovePreview, ply: Int) -> Int {
        guard fullDepth >= 3 else { return 0 }
        guard moveIndex >= 3 else { return 0 }
        guard !preview.isForcing else { return 0 }
        guard preview.owner == emptyCode else { return 0 }
        guard preview.tileClass != .interior else { return 0 }
        let latePly = ply >= 2 ? 1 : 0
        return min(2, 1 + latePly)
    }

    private func solveExactRoot(state: SearchState, player: UInt8) -> HexplodeSearchResult? {
        let ordered = orderedMoves(for: player, in: state, ply: 0)
        guard !ordered.isEmpty else { return nil }

        var bestMove = ordered[0].move
        var bestScore = -mateScore
        for preview in ordered {
            if isTimedOut() { return nil }
            let next = applyMove(preview.move, for: player, in: state)
            guard let score = solveToTerminal(
                state: next,
                player: opponent(of: player),
                alpha: -mateScore,
                beta: mateScore,
                ply: 1
            ) else {
                return nil
            }
            let value = -score
            if value > bestScore {
                bestScore = value
                bestMove = preview.move
            }
        }

        let pv = principalVariation(from: state, player: player, maxLength: 24)
        return HexplodeSearchResult(
            bestMove: coords[bestMove],
            score: bestScore,
            depthReached: 99,
            principalVariation: (pv.isEmpty ? [bestMove] : pv).map { coords[$0] }
        )
    }

    private func solveToTerminal(
        state: SearchState,
        player: UInt8,
        alpha: Int,
        beta: Int,
        ply: Int
    ) -> Int? {
        if isTimedOut() { return nil }
        if ply >= 48 {
            // Exact solving is only safe in very small late-game trees.
            return evaluate(state: state, for: player, ply: ply)
        }
        if let terminal = terminalScore(state: state, for: player, ply: ply) {
            return terminal
        }

        let legal = orderedMoves(for: player, in: state, ply: ply)
        guard !legal.isEmpty else {
            return evaluate(state: state, for: player, ply: ply)
        }

        var localAlpha = alpha
        var best = -mateScore
        for preview in legal {
            let next = applyMove(preview.move, for: player, in: state)
            guard let child = solveToTerminal(
                state: next,
                player: opponent(of: player),
                alpha: -beta,
                beta: -localAlpha,
                ply: ply + 1
            ) else {
                return nil
            }
            let score = -child
            if score > best {
                best = score
            }
            if score > localAlpha {
                localAlpha = score
            }
            if localAlpha >= beta {
                break
            }
        }
        return best
    }

    private func evaluate(state: SearchState, for player: UInt8, ply: Int) -> Int {
        if let terminal = terminalScore(state: state, for: player, ply: ply) {
            return terminal
        }

        let enemyPlayer = self.opponent(of: player)
        let earlyPhase = isEarlyPhase(state)
        let ownBalls = player == redCode ? state.redBalls : state.blueBalls
        let oppBalls = player == redCode ? state.blueBalls : state.redBalls
        let ownTiles = player == redCode ? state.redTiles : state.blueTiles
        let oppTiles = player == redCode ? state.blueTiles : state.redTiles

        let tileWeight = earlyPhase ? 18 : 90
        let ballWeight = earlyPhase ? 26 : 22
        var score = (ownTiles - oppTiles) * tileWeight
        score += (ownBalls - oppBalls) * ballWeight

        var ownFrontierPressure = 0
        var oppFrontierPressure = 0
        var ownConnectivity = 0
        var oppConnectivity = 0
        var ownBurstPotential = 0
        var oppBurstPotential = 0
        var ownBorderPenalty = 0
        var oppBorderPenalty = 0
        var ownSupportedCritical = 0
        var oppSupportedCritical = 0
        var ownVulnerableCritical = 0
        var oppVulnerableCritical = 0
        var ownConvertibleEnemyCritical = 0
        var oppConvertibleEnemyCritical = 0
        var ownEarlyDevelopment = 0
        var oppEarlyDevelopment = 0
        var ownEarlyRacePenalty = 0
        var oppEarlyRacePenalty = 0
        var ownScatteredLoad = 0
        var oppScatteredLoad = 0
        var ownOpeningBuildups = 0
        var oppOpeningBuildups = 0

        for index in coords.indices {
            let owner = state.owners[index]
            guard owner != emptyCode else { continue }

            let level = state.levels[index]
            let sign = owner == player ? 1 : -1
            let localCentrality = centrality[index]

            var friendlyAdjacent = 0
            var enemyAdjacent = 0
            var friendlyCritical = 0
            var enemyCritical = 0
            var friendlyNearCritical = 0
            var enemyNearCritical = 0
            var friendlyStrength = 0
            var enemyStrength = 0
            var friendlyMaxLevel = 0
            var enemyMaxLevel = 0
            let tileClass = tileClasses[index]

            for neighbor in neighbors[index] {
                let neighborOwner = state.owners[neighbor]
                let neighborLevel = state.levels[neighbor]
                if neighborOwner == owner {
                    friendlyAdjacent += 1
                    friendlyStrength += neighborLevel
                    friendlyMaxLevel = max(friendlyMaxLevel, neighborLevel)
                    if neighborLevel >= 5 {
                        friendlyCritical += 1
                    } else if neighborLevel >= 4 {
                        friendlyNearCritical += 1
                    }
                    if shouldCountConnection(index, neighbor) {
                        if owner == player {
                            ownConnectivity += 1
                        } else {
                            oppConnectivity += 1
                        }
                    }
                } else if neighborOwner == self.opponent(of: owner) {
                    enemyAdjacent += 1
                    enemyStrength += neighborLevel
                    enemyMaxLevel = max(enemyMaxLevel, neighborLevel)
                    if neighborLevel >= 5 {
                        enemyCritical += 1
                    } else if neighborLevel >= 4 {
                        enemyNearCritical += 1
                    }
                }
            }

            score += sign * localCentrality * (level >= 4 ? 14 : 9)

            let supportedCritical = level >= 5 && (friendlyAdjacent + friendlyNearCritical) >= (enemyAdjacent + enemyNearCritical)
            let vulnerableCritical = level >= 5 && (enemyCritical > 0 || enemyAdjacent > friendlyAdjacent || enemyStrength > friendlyStrength)
            let convertibleEnemyCritical = enemyCritical > 0 && level >= 5
            let burstPotential = max(0, (level - 3) * (friendlyAdjacent + (2 * enemyAdjacent) + localCentrality))
            let frontierPressure = max(0, enemyStrength - friendlyStrength) + max(0, enemyAdjacent - friendlyAdjacent)

            switch tileClass {
            case .interior:
                score += sign * level * 10
            case .edge:
                score -= sign * level * 7
                if friendlyAdjacent == 0 && enemyAdjacent > 0 {
                    if owner == player {
                        ownBorderPenalty += level * (1 + enemyAdjacent)
                    } else {
                        oppBorderPenalty += level * (1 + enemyAdjacent)
                    }
                }
            case .corner:
                score -= sign * level * 12
                if friendlyAdjacent == 0 && enemyAdjacent > 0 {
                    if owner == player {
                        ownBorderPenalty += level * (2 + enemyAdjacent)
                    } else {
                        oppBorderPenalty += level * (2 + enemyAdjacent)
                    }
                }
            }

            if owner == player {
                ownBurstPotential += burstPotential
                ownFrontierPressure += frontierPressure
                if supportedCritical { ownSupportedCritical += 1 }
                if vulnerableCritical { ownVulnerableCritical += 1 }
                if convertibleEnemyCritical { ownConvertibleEnemyCritical += enemyCritical }
                if earlyPhase {
                    ownEarlyDevelopment += max(0, level - 1) * (2 + friendlyAdjacent + localCentrality)
                    ownEarlyDevelopment += min(level, 4) * max(0, friendlyMaxLevel - 1)
                    if enemyAdjacent > 0 {
                        let tempoDelta = level - enemyMaxLevel
                        if tempoDelta < 0 {
                            ownEarlyRacePenalty += (abs(tempoDelta) * 6) + (friendlyAdjacent == 0 ? 12 : 0)
                        } else {
                            ownEarlyDevelopment += tempoDelta * 4
                        }
                    }
                    if friendlyAdjacent == 0 && enemyAdjacent > 0 && level <= 2 {
                        ownEarlyRacePenalty += (enemyAdjacent * 8) + level * 4
                    }
                    ownOpeningBuildups += max(0, level - 2) * (8 + localCentrality + (friendlyAdjacent * 2))
                    if level >= 4 {
                        ownOpeningBuildups += 10 + (friendlyAdjacent * 3) + max(0, level - enemyMaxLevel) * 4
                    }
                    if level == 1 {
                        ownScatteredLoad += 10
                        if friendlyAdjacent == 0 {
                            ownScatteredLoad += 22 + localCentrality
                        }
                        if enemyAdjacent > friendlyAdjacent {
                            ownScatteredLoad += 12 + (enemyAdjacent - friendlyAdjacent) * 6
                        }
                        if tileClass != .interior {
                            ownScatteredLoad += 8
                        }
                    } else if level == 2 && friendlyAdjacent == 0 {
                        ownScatteredLoad += 8 + (enemyAdjacent * 6)
                    }
                }
            } else {
                oppBurstPotential += burstPotential
                oppFrontierPressure += frontierPressure
                if supportedCritical { oppSupportedCritical += 1 }
                if vulnerableCritical { oppVulnerableCritical += 1 }
                if convertibleEnemyCritical { oppConvertibleEnemyCritical += enemyCritical }
                if earlyPhase {
                    oppEarlyDevelopment += max(0, level - 1) * (2 + friendlyAdjacent + localCentrality)
                    oppEarlyDevelopment += min(level, 4) * max(0, friendlyMaxLevel - 1)
                    if enemyAdjacent > 0 {
                        let tempoDelta = level - enemyMaxLevel
                        if tempoDelta < 0 {
                            oppEarlyRacePenalty += (abs(tempoDelta) * 6) + (friendlyAdjacent == 0 ? 12 : 0)
                        } else {
                            oppEarlyDevelopment += tempoDelta * 4
                        }
                    }
                    if friendlyAdjacent == 0 && enemyAdjacent > 0 && level <= 2 {
                        oppEarlyRacePenalty += (enemyAdjacent * 8) + level * 4
                    }
                    oppOpeningBuildups += max(0, level - 2) * (8 + localCentrality + (friendlyAdjacent * 2))
                    if level >= 4 {
                        oppOpeningBuildups += 10 + (friendlyAdjacent * 3) + max(0, level - enemyMaxLevel) * 4
                    }
                    if level == 1 {
                        oppScatteredLoad += 10
                        if friendlyAdjacent == 0 {
                            oppScatteredLoad += 22 + localCentrality
                        }
                        if enemyAdjacent > friendlyAdjacent {
                            oppScatteredLoad += 12 + (enemyAdjacent - friendlyAdjacent) * 6
                        }
                        if tileClass != .interior {
                            oppScatteredLoad += 8
                        }
                    } else if level == 2 && friendlyAdjacent == 0 {
                        oppScatteredLoad += 8 + (enemyAdjacent * 6)
                    }
                }
            }
        }

        score += (ownBurstPotential - oppBurstPotential) * 7
        score += (ownConnectivity - oppConnectivity) * 18
        score += (ownSupportedCritical - oppSupportedCritical) * 140
        score -= (ownVulnerableCritical - oppVulnerableCritical) * 155
        score += (ownConvertibleEnemyCritical - oppConvertibleEnemyCritical) * 115
        score -= (ownFrontierPressure - oppFrontierPressure) * 20
        score -= (ownBorderPenalty - oppBorderPenalty) * 26
        if earlyPhase {
            score += (ownEarlyDevelopment - oppEarlyDevelopment) * 24
            score -= (ownEarlyRacePenalty - oppEarlyRacePenalty) * 22
            score += (ownOpeningBuildups - oppOpeningBuildups) * 28
            score -= (ownScatteredLoad - oppScatteredLoad) * 26
        }

        let ownImmediateWins = immediateWinningMovesCount(for: player, in: state, limit: 6)
        let oppImmediateWins = immediateWinningMovesCount(for: enemyPlayer, in: state, limit: 6)
        score += ownImmediateWins * 1_800
        score -= oppImmediateWins * 2_400

        return score
    }

    private func terminalScore(state: SearchState, for player: UInt8, ply: Int) -> Int? {
        if state.redTiles == 0, state.blueTiles == 0, state.redHadTile, state.blueHadTile {
            return 0
        }
        if state.redTiles == 0, state.blueTiles > 0, state.redHadTile {
            return player == blueCode ? mateScore - ply : -mateScore + ply
        }
        if state.blueTiles == 0, state.redTiles > 0, state.blueHadTile {
            return player == redCode ? mateScore - ply : -mateScore + ply
        }
        return nil
    }

    private func isEarlyPhase(_ state: SearchState) -> Bool {
        let totalBalls = state.redBalls + state.blueBalls
        return state.turnsPlayed < max(12, radius * 4)
            || totalBalls < max(radius * 8, coords.count / 2)
    }

    private func legalMoves(for player: UInt8, in state: SearchState) -> [Int] {
        coords.indices.filter { index in
            let owner = state.owners[index]
            return owner == emptyCode || owner == player
        }
    }

    private func makeMovePreview(move: Int, player: UInt8, state: SearchState) -> MovePreview {
        let owner = state.owners[move]
        let level = state.levels[move]
        var friendlyAdjacent = 0
        var enemyAdjacent = 0
        var friendlyMaxLevel = 0
        var enemyMaxLevel = 0
        var friendlyNearCritical = 0
        var enemyNearCritical = 0
        var enemyCritical = 0

        for neighbor in neighbors[move] {
            let neighborOwner = state.owners[neighbor]
            let neighborLevel = state.levels[neighbor]
            if neighborOwner == player {
                friendlyAdjacent += 1
                friendlyMaxLevel = max(friendlyMaxLevel, neighborLevel)
                if neighborLevel >= 4 {
                    friendlyNearCritical += 1
                }
            } else if neighborOwner == opponent(of: player) {
                enemyAdjacent += 1
                enemyMaxLevel = max(enemyMaxLevel, neighborLevel)
                if neighborLevel >= 4 {
                    enemyNearCritical += 1
                }
                if neighborLevel >= 5 {
                    enemyCritical += 1
                }
            }
        }

        let explodes = owner == player && (level + 1) >= 6
        let isolatedEmptyBorder = owner == emptyCode
            && tileClasses[move] != .interior
            && friendlyAdjacent == 0
            && enemyAdjacent == 0
        let unsupportedContest = owner == emptyCode
            && enemyAdjacent > 0
            && friendlyAdjacent == 0

        return MovePreview(
            move: move,
            owner: owner,
            level: level,
            tileClass: tileClasses[move],
            centrality: centrality[move],
            friendlyAdjacent: friendlyAdjacent,
            enemyAdjacent: enemyAdjacent,
            friendlyMaxLevel: friendlyMaxLevel,
            enemyMaxLevel: enemyMaxLevel,
            friendlyNearCritical: friendlyNearCritical,
            enemyNearCritical: enemyNearCritical,
            enemyCritical: enemyCritical,
            explodes: explodes,
            touchesEnemyThreat: enemyNearCritical > 0,
            touchesFriendlyThreat: friendlyNearCritical > 0,
            isolatedEmptyBorder: isolatedEmptyBorder,
            unsupportedContest: unsupportedContest
        )
    }

    private func applyMove(_ move: Int, for player: UInt8, in state: SearchState) -> SearchState {
        var next = state
        let level = next.levels[move]
        setTile(move, owner: player, level: level + 1, in: &next)
        if next.levels[move] >= 6 {
            resolveExplosions(startingAt: move, in: &next)
        }
        next.turnsPlayed += 1
        if next.redTiles > 0 {
            next.redHadTile = true
        }
        if next.blueTiles > 0 {
            next.blueHadTile = true
        }
        return next
    }

    private func resolveExplosions(startingAt start: Int, in state: inout SearchState) {
        var queue = [start]
        var queueIndex = 0
        var queued = Array(repeating: false, count: coords.count)
        queued[start] = true

        while queueIndex < queue.count {
            let current = queue[queueIndex]
            queueIndex += 1
            queued[current] = false

            let owner = state.owners[current]
            let level = state.levels[current]
            guard owner != emptyCode, level >= 6 else { continue }

            let burstCount = level / 6
            let remainder = level % 6
            if remainder > 0 {
                setTile(current, owner: owner, level: remainder, in: &state)
            } else {
                setTile(current, owner: emptyCode, level: 0, in: &state)
            }

            for neighbor in neighbors[current] {
                let nextLevel = state.levels[neighbor] + burstCount
                setTile(neighbor, owner: owner, level: nextLevel, in: &state)
                if nextLevel >= 6, !queued[neighbor] {
                    queue.append(neighbor)
                    queued[neighbor] = true
                }
            }
        }
    }

    private func setTile(_ index: Int, owner: UInt8, level: Int, in state: inout SearchState) {
        let oldOwner = state.owners[index]
        let oldLevel = state.levels[index]
        if oldOwner == owner, oldLevel == level {
            return
        }

        state.hash ^= ownerHash(tile: index, owner: oldOwner)
        state.hash ^= levelHash(tile: index, level: oldLevel)
        adjustCounts(owner: oldOwner, level: oldLevel, direction: -1, state: &state)

        state.owners[index] = owner
        state.levels[index] = level

        adjustCounts(owner: owner, level: level, direction: 1, state: &state)
        state.hash ^= ownerHash(tile: index, owner: owner)
        state.hash ^= levelHash(tile: index, level: level)
    }

    private func adjustCounts(owner: UInt8, level: Int, direction: Int, state: inout SearchState) {
        guard owner != emptyCode else { return }
        if owner == redCode {
            state.redBalls += direction * level
            state.redTiles += direction
        } else {
            state.blueBalls += direction * level
            state.blueTiles += direction
        }
    }

    private func makeState(
        owners: [AxialCoord: TileState],
        levels: [AxialCoord: Int],
        turnsPlayed: Int,
        redHadTile: Bool,
        blueHadTile: Bool
    ) -> SearchState {
        var state = SearchState(
            owners: Array(repeating: emptyCode, count: coords.count),
            levels: Array(repeating: 0, count: coords.count),
            turnsPlayed: turnsPlayed,
            redTiles: 0,
            blueTiles: 0,
            redBalls: 0,
            blueBalls: 0,
            redHadTile: redHadTile,
            blueHadTile: blueHadTile,
            hash: 0
        )

        for (index, coord) in coords.enumerated() {
            let owner = ownerCode(owners[coord] ?? .empty)
            let level = max(0, levels[coord] ?? 0)
            state.owners[index] = owner
            state.levels[index] = level
            adjustCounts(owner: owner, level: level, direction: 1, state: &state)
            state.hash ^= ownerHash(tile: index, owner: owner)
            state.hash ^= levelHash(tile: index, level: level)
        }

        if state.redTiles > 0 {
            state.redHadTile = true
        }
        if state.blueTiles > 0 {
            state.blueHadTile = true
        }
        return state
    }

    private func principalVariation(from state: SearchState, player: UInt8, maxLength: Int) -> [Int] {
        var pv: [Int] = []
        var currentState = state
        var currentPlayer = player

        for _ in 0..<maxLength {
            let key = hashKey(for: currentPlayer, state: currentState)
            guard let entry = transpositionTable[key], entry.hash == key else { break }
            let move = entry.bestMove
            let legal = legalMoves(for: currentPlayer, in: currentState)
            guard legal.contains(move) else { break }
            pv.append(move)
            currentState = applyMove(move, for: currentPlayer, in: currentState)
            currentPlayer = opponent(of: currentPlayer)
            if terminalScore(state: currentState, for: currentPlayer, ply: pv.count) != nil {
                break
            }
        }

        return pv
    }

    private func aspirationWindow(center: Int, depth: Int) -> (lower: Int, upper: Int) {
        if depth <= 2 {
            return (-mateScore, mateScore)
        }
        let window = 180 + (depth * 36)
        return (center - window, center + window)
    }

    private func isExactEndgameCandidate(state: SearchState, legalMoveCount: Int) -> Bool {
        let occupied = state.redTiles + state.blueTiles
        let emptyTiles = max(0, coords.count - occupied)
        let contested = coords.indices.reduce(into: 0) { count, index in
            let owner = state.owners[index]
            guard owner != emptyCode else { return }
            if neighbors[index].contains(where: { state.owners[$0] == opponent(of: owner) }) {
                count += 1
            }
        }
        guard state.turnsPlayed >= max(10, coords.count / 2) else { return false }
        if legalMoveCount <= 3 && contested <= 4 {
            return true
        }
        return emptyTiles <= 4 && legalMoveCount <= 5 && contested <= 6
    }

    private func hasImmediateWinningMove(for player: UInt8, in state: SearchState, limit: Int) -> Bool {
        immediateWinningMovesCount(for: player, in: state, limit: limit) > 0
    }

    private func immediateWinningMovesCount(for player: UInt8, in state: SearchState, limit: Int) -> Int {
        var checked = 0
        var wins = 0
        for move in legalMoves(for: player, in: state) {
            if checked >= limit { break }
            checked += 1
            let next = applyMove(move, for: player, in: state)
            if let terminal = terminalScore(state: next, for: player, ply: 1), terminal > 0 {
                wins += 1
                if wins >= 2 {
                    return wins
                }
            }
        }
        return wins
    }

    private func shouldCountConnection(_ lhs: Int, _ rhs: Int) -> Bool {
        lhs < rhs
    }

    private func recordKillerMove(_ move: Int, ply: Int) {
        guard ply < killerMoves.count else { return }
        if killerMoves[ply][0] == move { return }
        killerMoves[ply][1] = killerMoves[ply][0]
        killerMoves[ply][0] = move
    }

    private func hashKey(for player: UInt8, state: SearchState) -> UInt64 {
        state.hash ^ playerHash(player)
    }

    private func ownerCode(_ state: TileState) -> UInt8 {
        switch state {
        case .empty:
            return emptyCode
        case .red:
            return redCode
        case .blue:
            return blueCode
        }
    }

    private func opponent(of player: UInt8) -> UInt8 {
        player == redCode ? blueCode : redCode
    }

    private func ownerHash(tile: Int, owner: UInt8) -> UInt64 {
        mix64(UInt64(tile &* 0x9E3779B1) ^ UInt64(owner &+ 1) &* 0xBF58476D1CE4E5B9)
    }

    private func levelHash(tile: Int, level: Int) -> UInt64 {
        mix64(UInt64(tile &* 0x94D049BB) ^ UInt64(level &+ 1) &* 0x94D049BB133111EB)
    }

    private func playerHash(_ player: UInt8) -> UInt64 {
        mix64(UInt64(player &+ 17) &* 0x9E3779B97F4A7C15)
    }

    private func mix64(_ value: UInt64) -> UInt64 {
        var x = value &+ 0x9E3779B97F4A7C15
        x = (x ^ (x >> 30)) &* 0xBF58476D1CE4E5B9
        x = (x ^ (x >> 27)) &* 0x94D049BB133111EB
        return x ^ (x >> 31)
    }

    private func isTimedOut() -> Bool {
        if timedOut {
            return true
        }
        if Date().timeIntervalSinceReferenceDate >= deadline {
            timedOut = true
            return true
        }
        return false
    }

    private let emptyCode: UInt8 = 0
    private let redCode: UInt8 = 1
    private let blueCode: UInt8 = 2
    private let mateScore = 1_000_000
}
