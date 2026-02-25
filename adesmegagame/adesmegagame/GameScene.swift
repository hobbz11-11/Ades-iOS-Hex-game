import SceneKit
import UIKit

enum GameMode: String {
    case hexello
    case hexpand
}

enum AIDifficulty {
    case easy
    case hard
}

final class GameSceneController: NSObject {
    let scene: SCNScene
    let boardNode = SCNNode()
    private let tiltNode = SCNNode()
    private let cameraNode = SCNNode()

    private var sideLength = 6
    private var radius: Int {
        sideLength - 1
    }
    private let tileSize: CGFloat = 0.62
    private let tileHeight: CGFloat = 0.12
    private let hexelloTileSpacing: Float = 1.075
    private let hexpandTileSpacing: Float = 1.0

    private var gameMode: GameMode = .hexello
    private var currentYaw: Float = 0
    private let defaultBoardPitch: Float = 0
    private var currentPitch: Float = 0
    private let minPitch: Float = -34 * .pi / 180
    private let maxPitch: Float = 56 * .pi / 180
    private var dragStartYaw: Float = 0
    private var dragStartPitch: Float = 0
    private var angularVelocityYaw: Float = 0
    private var angularVelocityPitch: Float = 0
    private let angularDamping: Float = 0.85
    private let velocityScale: Float = 0.00035
    private var displayLink: CADisplayLink?
    private let initialCameraDistance: Float = 20.0
    private let defaultCameraDistance: Float = 11.0
    private var dragStartDistance: Float = 11.6
    private var cameraDistance: Float = 30.0
    private let minCameraDistance: Float = 3.0
    private let maxCameraDistance: Float = 40.0
    private var dragStartBoardPosition = SCNVector3Zero
    private let boardPanScale: Float = 0.015

    private let popAudio: SCNAudioSource? = {
        guard let source = SCNAudioSource(named: "pop.wav") else { return nil }
        source.load()
        source.volume = 0.8
        source.isPositional = false
        return source
    }()
    private let placeAudio: SCNAudioSource? = {
        guard let source = SCNAudioSource(named: "place.wav") else { return nil }
        source.load()
        source.volume = 0.8
        source.isPositional = false
        return source
    }()
    private let flipAudio: SCNAudioSource? = {
        guard let source = SCNAudioSource(named: "flip.wav") else { return nil }
        source.load()
        source.volume = 0.8
        source.isPositional = false
        return source
    }()

    private var tileNodes: [AxialCoord: HexTileNode] = [:]
    private var tileStates: [AxialCoord: TileState] = [:]
    private var hexpandLevels: [AxialCoord: Int] = [:]
    private var hexpandBallNodes: [AxialCoord: [SCNNode]] = [:]
    private var hexpandPendingExplosions: Set<AxialCoord> = []
    private var hexpandExplosionQueue: [AxialCoord] = []
    private var hexpandExplosionQueueHead = 0
    private var isHexpandExploding = false
    private let hexpandMaxLevel = 6
    private var currentPlayer: TileState = .red
    private var aiDifficulty: AIDifficulty = .hard
    private var aiMoveScheduled = false
    var onTurnUpdate: ((TileState) -> Void)?
    private var isAnimatingMove = false
    private var isGameOver = false
    private var redHadTile = false
    private var blueHadTile = false
    var onPitchUpdate: ((Float) -> Void)?
    var onYawUpdate: ((Float) -> Void)?
    var onMessageUpdate: ((String) -> Void)?
    var onScoreUpdate: ((Int, Int) -> Void)?
    var onGameOver: ((String, String) -> Void)?
    var onActiveUpdate: ((TileState) -> Void)?
    private var aiRedEnabled = false
    private var aiBlueEnabled = false
    private var aiComputationToken = UUID()
    private let aiComputeQueue = DispatchQueue(label: "adesmegagame.hexpand.ai", qos: .userInitiated)
    private var movesMade = 0
    private var startAIOnIntroComplete = false
    private var introCompleted = false
    private let introAnimationDuration: TimeInterval = 2.5
    private var boardCoords: [AxialCoord] = []
    private var hexpandTranspositionTable: [HexpandTTKey: HexpandTTEntry] = [:]

    override init() {
        scene = SCNScene()
        super.init()
        buildScene()
    }

    func configureView(_ view: SCNView) {
        view.scene = scene
        view.backgroundColor = .clear
        view.isPlaying = true
        view.rendersContinuously = true
        view.antialiasingMode = .multisampling4X
        view.autoenablesDefaultLighting = false
        view.scene?.lightingEnvironment.contents = nil
    }

    func handlePan(_ gesture: UIPanGestureRecognizer) {
        let translation = gesture.translation(in: gesture.view)
        switch gesture.state {
        case .began:
            dragStartYaw = currentYaw
            dragStartPitch = currentPitch
            angularVelocityYaw = 0
            angularVelocityPitch = 0
        case .changed:
            let yawDelta = Float(translation.x) * 0.006
            let pitchDelta = Float(translation.y) * 0.006
            currentYaw = dragStartYaw + yawDelta
            currentPitch = clamp(dragStartPitch + pitchDelta, min: minPitch, max: maxPitch)
            updateBoardRotation()
        case .ended, .cancelled:
            let velocity = gesture.velocity(in: gesture.view)
            angularVelocityYaw = Float(velocity.x) * velocityScale
            angularVelocityPitch = Float(velocity.y) * velocityScale
            startInertia()
        default:
            break
        }
    }

    func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        switch gesture.state {
        case .began:
            dragStartDistance = cameraDistance
        case .changed:
            let scaled = dragStartDistance / Float(gesture.scale)
            cameraDistance = clamp(scaled, min: minCameraDistance, max: maxCameraDistance)
            updateCameraPosition()
        default:
            break
        }
    }

    func handleTwoFingerPan(_ gesture: UIPanGestureRecognizer) {
        let translation = gesture.translation(in: gesture.view)
        switch gesture.state {
        case .began:
            dragStartBoardPosition = boardNode.position
        case .changed:
            let dx = Float(translation.x) * boardPanScale
            let dz = Float(translation.y) * boardPanScale
            boardNode.position = SCNVector3(
                dragStartBoardPosition.x + dx,
                dragStartBoardPosition.y,
                dragStartBoardPosition.z + dz
            )
        default:
            break
        }
    }

    func handleTap(_ gesture: UITapGestureRecognizer, in view: SCNView) {
        guard canAcceptHumanInput else { return }

        let location = gesture.location(in: view)
        let hits = view.hitTest(location, options: [
            SCNHitTestOption.searchMode: SCNHitTestSearchMode.closest.rawValue,
            SCNHitTestOption.firstFoundOnly: true
        ])

        if let node = hits.first?.node as? HexTileNode {
            applyMove(at: node.coord)
            return
        }

        if let hitNode = hits.first?.node,
           let q = hitNode.value(forKey: "q") as? Int,
           let r = hitNode.value(forKey: "r") as? Int {
            applyMove(at: AxialCoord(q: q, r: r))
        }
    }

    private var canAcceptHumanInput: Bool {
        guard introCompleted, !isGameOver, !isAnimatingMove, !aiMoveScheduled else { return false }
        return !isAIEnabled(for: currentPlayer)
    }

    private func isAIEnabled(for player: TileState) -> Bool {
        (player == .red && aiRedEnabled) || (player == .blue && aiBlueEnabled)
    }

    private func buildScene() {
        scene.rootNode.addChildNode(tiltNode)
        tiltNode.addChildNode(boardNode)
        buildBoard()
        buildCamera()
        buildLights()
        updateBoardRotation()
        introCompleted = false
        animateInitialZoomIn()
    }

    private func buildBoard() {
        boardNode.position = SCNVector3Zero
        tileNodes.removeAll()
        tileStates.removeAll()
        hexpandLevels.removeAll()
        hexpandBallNodes.removeAll()
        hexpandPendingExplosions.removeAll()
        hexpandExplosionQueue.removeAll()
        hexpandExplosionQueueHead = 0
        aiComputationToken = UUID()
        isHexpandExploding = false
        aiMoveScheduled = false
        movesMade = 0
        isGameOver = false
        redHadTile = false
        blueHadTile = false
        boardNode.childNodes.forEach { $0.removeFromParentNode() }
        let coords = HexGrid.generateHexagon(radius: radius)
        boardCoords = coords
        hexpandTranspositionTable.removeAll(keepingCapacity: true)
        for coord in coords {
            let tile = HexTileNode(coord: coord, size: tileSize, height: tileHeight)
            let position = HexGrid.axialToWorld(
                q: coord.q,
                r: coord.r,
                tileSize: Float(tileSize) * currentTileSpacing()
            )
            tile.position = SCNVector3(position.x, Float(tileHeight * 0.5), position.z)
            tile.setStyle(isHexpand: gameMode == .hexpand)
            tile.setState(.empty)
            boardNode.addChildNode(tile)
            tileNodes[coord] = tile
            tileStates[coord] = .empty
            hexpandLevels[coord] = 0
            hexpandBallNodes[coord] = []
        }
        if gameMode == .hexello {
            setupInitialRing()
        } else {
            onMessageUpdate?("")
            updateScores()
        }
    }

    private func setupInitialRing() {
        let ring = HexGrid.ringCoordinates(radius: 1)
        for (index, coord) in ring.enumerated() {
            let color: TileState = index.isMultiple(of: 2) ? .red : .blue
            setTileState(coord, to: color)
        }
        let centerColor: TileState = Bool.random() ? .red : .blue
        setTileState(AxialCoord(q: 0, r: 0), to: centerColor)
        currentPlayer = centerColor.opposite
        onTurnUpdate?(currentPlayer)
        onActiveUpdate?(currentPlayer)
        onMessageUpdate?("")
        updateScores()
        performAITurnIfNeeded()
    }

    private func applyMove(at coord: AxialCoord) {
        guard !isAnimatingMove, !isGameOver else { return }
        aiMoveScheduled = false
        if gameMode == .hexpand {
            applyHexpandMove(at: coord)
            return
        }
        guard tileStates[coord] == .empty else { return }
        let flipLines = flipLinesForMove(at: coord, player: currentPlayer)
        guard !flipLines.isEmpty else { return }
        isAnimatingMove = true
        tileStates[coord] = currentPlayer
        tileNodes[coord]?.animatePlacement(to: currentPlayer) { [weak self] in
            guard let self else { return }
            self.movesMade += 1
            self.tileNodes[coord]?.setState(self.currentPlayer)
            self.updateScores()
            self.animateFlips(flipLines, to: self.currentPlayer) {
                self.advanceTurn()
            }
        }
        playPlace()
    }

    private func flipsForMove(at coord: AxialCoord, player: TileState) -> [AxialCoord] {
        flipLinesForMove(at: coord, player: player).flatMap { $0.coords }
    }

    private func flipsForMove(at coord: AxialCoord, player: TileState, in state: [AxialCoord: TileState]) -> [AxialCoord] {
        flipLinesForMove(at: coord, player: player, in: state).flatMap { $0.coords }
    }

    private func flipLinesForMove(at coord: AxialCoord, player: TileState, in state: [AxialCoord: TileState]) -> [(direction: AxialCoord, coords: [AxialCoord])] {
        var lines: [(direction: AxialCoord, coords: [AxialCoord])] = []
        for direction in HexGrid.directions {
            var line: [AxialCoord] = []
            var current = coord.adding(direction)
            while let currentState = state[current] {
                if currentState == player.opposite {
                    line.append(current)
                } else if currentState == player {
                    if !line.isEmpty {
                        lines.append((direction: direction, coords: line))
                    }
                    break
                } else {
                    break
                }
                current = current.adding(direction)
            }
        }
        return lines
    }

    private func currentTileSpacing() -> Float {
        gameMode == .hexello ? hexelloTileSpacing : hexpandTileSpacing
    }


    private func applyHexpandMove(at coord: AxialCoord) {
        guard let state = tileStates[coord] else { return }
        guard state == .empty || state == currentPlayer else { return }
        isAnimatingMove = true

        if state == .empty {
            tileStates[coord] = currentPlayer
            hexpandLevels[coord] = 1
            updateHexpandTile(coord: coord, level: 1, owner: currentPlayer)
            updateScores()
            playPlace()
            finishHexpandMove()
            return
        }

        let nextLevel = (hexpandLevels[coord] ?? 0) + 1
        if nextLevel < hexpandMaxLevel {
            hexpandLevels[coord] = nextLevel
            updateHexpandTile(coord: coord, level: nextLevel, owner: currentPlayer)
            updateScores()
            playPlace()
            finishHexpandMove()
            return
        }

        hexpandLevels[coord] = nextLevel
        updateHexpandTile(coord: coord, level: nextLevel, owner: currentPlayer)
        updateScores()
        let explodeDelay: TimeInterval = 0.2
        DispatchQueue.main.asyncAfter(deadline: .now() + explodeDelay) { [weak self] in
            self?.enqueueHexpandExplosion(coord)
        }
    }

    private func finishHexpandMove() {
        movesMade += 1
        advanceTurn()
        isAnimatingMove = false
    }

    private func enqueueHexpandExplosion(_ coord: AxialCoord) {
        guard !hexpandPendingExplosions.contains(coord) else { return }
        hexpandPendingExplosions.insert(coord)
        hexpandExplosionQueue.append(coord)
        processNextHexpandExplosionIfNeeded()
    }

    private func processNextHexpandExplosionIfNeeded() {
        guard !isHexpandExploding else { return }
        guard hexpandExplosionQueueHead < hexpandExplosionQueue.count else { return }
        let coord = hexpandExplosionQueue[hexpandExplosionQueueHead]
        hexpandExplosionQueueHead += 1
        trimHexpandExplosionQueueIfNeeded()
        isHexpandExploding = true
        triggerHexpandExplosion(from: coord)
    }

    private func triggerHexpandExplosion(from coord: AxialCoord) {
        hexpandPendingExplosions.remove(coord)

        guard tileNodes[coord] != nil else {
            isHexpandExploding = false
            processNextHexpandExplosionIfNeeded()
            return
        }

        let burstDuration: TimeInterval = 0.7
        let explosionOwner = tileStates[coord] ?? currentPlayer
        spawnHexpandBurst(from: coord, owner: explosionOwner, duration: burstDuration)
        playPop()

        let neighbors = HexGrid.directions.compactMap { direction -> AxialCoord? in
            let neighbor = coord.adding(direction)
            return tileStates[neighbor] == nil ? nil : neighbor
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + burstDuration) { [weak self] in
            guard let self else { return }
            self.tileStates[coord] = .empty
            self.hexpandLevels[coord] = 0
            self.updateHexpandTile(coord: coord, level: 0, owner: .empty, duration: 0.4)

            for neighbor in neighbors {
                let level = (self.hexpandLevels[neighbor] ?? 0) + 1
                self.tileStates[neighbor] = self.currentPlayer
                self.hexpandLevels[neighbor] = min(level, self.hexpandMaxLevel)
                self.updateHexpandTile(coord: neighbor, level: min(level, self.hexpandMaxLevel), owner: self.currentPlayer, duration: 0.3)

                if level >= self.hexpandMaxLevel {
                    self.enqueueHexpandExplosion(neighbor)
                }
            }

            self.updateScores()
            self.playPlace()
            self.isHexpandExploding = false
            if !self.hasPendingHexpandExplosions {
                if !self.checkForHexpandWinIfNeeded() {
                    self.finishHexpandMove()
                } else {
                    self.isAnimatingMove = false
                }
            }
            self.processNextHexpandExplosionIfNeeded()
        }
    }

    private func updateHexpandTile(coord: AxialCoord, level: Int, owner: TileState, duration: TimeInterval = 0.2) {
        guard let node = tileNodes[coord] else { return }
        node.setState(owner)
        node.animateState(to: owner, duration: duration)
        updateHexpandBalls(on: node, coord: coord, count: level, owner: owner, duration: duration)
    }

    private func updateHexpandBalls(
        on node: HexTileNode,
        coord: AxialCoord,
        count: Int,
        owner: TileState,
        duration: TimeInterval
    ) {
        let clampedCount = max(0, min(hexpandMaxLevel, count))
        let existing = hexpandBallNodes[coord] ?? []
        for ball in existing {
            ball.removeAllActions()
            ball.removeFromParentNode()
        }
        hexpandBallNodes[coord] = []
        guard clampedCount > 0 else { return }
        guard owner != .empty else { return }

        let ballRadius = hexpandBallRadius()
        let rotation = clampedCount > 1 ? Float.random(in: 0...(Float.pi)) : 0
        let offsets = hexpandBallOffsets(count: clampedCount, ballRadius: ballRadius, rotation: rotation)
        let localTop = SCNVector3(0, Float(tileHeight) * 0.5, 0)
        let worldTop = node.convertPosition(localTop, to: boardNode)
        let baseY = worldTop.y + Float(ballRadius) + 0.08
        let center = boardNode.convertPosition(node.position, from: node.parent ?? boardNode)

        let ballMaterial = makeHexpandBallMaterial(from: node, owner: owner)
        var balls: [SCNNode] = []
        for offset in offsets {
            let sphere = SCNSphere(radius: ballRadius)
            sphere.segmentCount = 20
            let ball = SCNNode(geometry: sphere)
            ball.geometry?.materials = [ballMaterial.copy() as? SCNMaterial ?? ballMaterial]
            let position = SCNVector3(center.x + offset.x, baseY, center.z + offset.z)
            ball.position = position
            ball.scale = SCNVector3(0.01, 0.01, 0.01)
            ball.name = "hexpandBall"
            ball.setValue(coord.q, forKey: "q")
            ball.setValue(coord.r, forKey: "r")
            boardNode.addChildNode(ball)
            balls.append(ball)

            let grow = SCNAction.scale(to: 1.0, duration: duration)
            grow.timingMode = .easeOut
            ball.runAction(grow)
        }
        hexpandBallNodes[coord] = balls
    }

    private func hexpandBallOffsets(count: Int, ballRadius: CGFloat, rotation: Float) -> [SCNVector3] {
        if count == 1 {
            return [SCNVector3(0, 0, 0)]
        }
        let n = Float(count)
        let desired = Float(ballRadius) * 1.02
        let ringRadius = desired / sin(.pi / n)
        return (0..<count).map { index in
            let angle = Float(index) * (2 * .pi / n) - (.pi / 2) + rotation
            return SCNVector3(cos(angle) * ringRadius, 0, sin(angle) * ringRadius)
        }
    }

    private func makeHexpandBallMaterial(from tile: HexTileNode, owner: TileState) -> SCNMaterial {
        let material = (tile.geometry?.firstMaterial?.copy() as? SCNMaterial) ?? SCNMaterial()
        material.emission.contents = UIColor.black
        material.emission.intensity = 0.2
        material.metalness.contents = 0.35
        material.roughness.contents = 0.35
        let color = HexTileNode.hexpandColor(for: owner)
        material.diffuse.contents = HexTileNode.darker(by: 0.18, color: color)
        return material
    }

    private func hexpandBallRadius() -> CGFloat {
        tileSize * 0.25
    }


    private func spawnHexpandBurst(from coord: AxialCoord, owner: TileState, duration: TimeInterval) {
        guard let originNode = tileNodes[coord] else { return }
        let origin = originNode.position
        let burstRadius = hexpandBallRadius()
        let localTop = SCNVector3(0, Float(tileHeight) * 0.5, 0)
        let worldTop = originNode.convertPosition(localTop, to: boardNode)
        let burstStartY = worldTop.y + Float(burstRadius) + 0.08

        if let existing = hexpandBallNodes[coord] {
            for ball in existing {
                ball.removeAllActions()
                ball.removeFromParentNode()
            }
            hexpandBallNodes[coord] = []
        }

        for direction in HexGrid.directions {
            let neighbor = coord.adding(direction)
            guard let neighborNode = tileNodes[neighbor] else { continue }
            let sphere = SCNSphere(radius: burstRadius)
            sphere.segmentCount = 20
            let shard = SCNNode(geometry: sphere)
            let shardMaterial = makeHexpandBallMaterial(from: originNode, owner: owner)
            shard.geometry?.materials = [shardMaterial.copy() as? SCNMaterial ?? shardMaterial]
            shard.position = SCNVector3(origin.x, burstStartY, origin.z)
            boardNode.addChildNode(shard)

            let target = SCNVector3(neighborNode.position.x, neighborNode.position.y + 0.18, neighborNode.position.z)
            let mid = SCNVector3(
                (origin.x + target.x) * 0.5,
                max(origin.y, target.y) + 0.96,
                (origin.z + target.z) * 0.5
            )
            let lift = SCNAction.move(to: mid, duration: duration * 0.45)
            lift.timingMode = .easeOut
            let drop = SCNAction.move(to: target, duration: duration * 0.55)
            drop.timingMode = .easeIn
            let fade = SCNAction.fadeOut(duration: duration)
            let scale = SCNAction.scale(to: 0.35, duration: duration)
            let spin = SCNAction.rotateBy(x: CGFloat.pi * 0.5, y: CGFloat.pi * 0.5, z: 0, duration: duration)
            let group = SCNAction.group([fade, scale, spin])
            let remove = SCNAction.removeFromParentNode()
            shard.runAction(.sequence([lift, drop, group, remove]))
        }
    }


    private func flipLinesForMove(at coord: AxialCoord, player: TileState) -> [(direction: AxialCoord, coords: [AxialCoord])] {
        flipLinesForMove(at: coord, player: player, in: tileStates)
    }

    private func setTileState(_ coord: AxialCoord, to state: TileState) {
        tileStates[coord] = state
        tileNodes[coord]?.setState(state)
        updateScores()
    }

    private func updateScores() {
        var redCount = 0
        var blueCount = 0
        for state in tileStates.values {
            switch state {
            case .red:
                redCount += 1
            case .blue:
                blueCount += 1
            case .empty:
                break
            }
        }
        if redCount > 0 {
            redHadTile = true
        }
        if blueCount > 0 {
            blueHadTile = true
        }
        onScoreUpdate?(redCount, blueCount)
    }

    private func animateFlips(
        _ lines: [(direction: AxialCoord, coords: [AxialCoord])],
        to state: TileState,
        completion: @escaping () -> Void
    ) {
        let sequence = lines.flatMap { line in
            line.coords.map { coord in (coord: coord, axis: flipAxis(for: line.direction)) }
        }
        guard !sequence.isEmpty else {
            completion()
            return
        }
        let duration: TimeInterval = 0.33
        var completedFlips = 0
        for (index, entry) in sequence.enumerated() {
            let delay = duration * Double(index)
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
                guard let self else { return }
                self.tileStates[entry.coord] = state
                self.tileNodes[entry.coord]?.animateFlip(to: state, axis: entry.axis, duration: duration) {
                    completedFlips += 1
                    if completedFlips == sequence.count {
                        completion()
                    }
                }
                self.updateScores()
                self.playFlip()
            }
        }
    }

    private func flipAxis(for direction: AxialCoord) -> SCNVector3 {
        let world = HexGrid.axialToWorld(
            q: direction.q,
            r: direction.r,
            tileSize: Float(tileSize) * currentTileSpacing()
        )
        let dx = world.x
        let dz = world.z
        let length = sqrt(dx * dx + dz * dz)
        if length == 0 {
            return SCNVector3(0, 0, 1)
        }
        let nx = dx / length
        let nz = dz / length
        return SCNVector3(-nz, 0, nx)
    }

    func setAIModes(redAI: Bool, blueAI: Bool) {
        aiRedEnabled = redAI
        aiBlueEnabled = blueAI
        performAITurnIfNeeded()
    }

    func setBoardSize(_ sideLength: Int) {
        self.sideLength = max(2, sideLength)
    }

    func setGameMode(_ mode: GameMode) {
        gameMode = mode
    }

    func setAIDifficulty(_ difficulty: AIDifficulty) {
        aiDifficulty = difficulty
    }

    func startGameOverAnimation() {
        angularVelocityYaw = 0
        angularVelocityPitch = 0
        cameraNode.removeAction(forKey: "gameOverZoom")
        boardNode.removeAction(forKey: "gameOverSpin")

        let startDistance = cameraDistance
        let startPitch = currentPitch
        let targetPitch = defaultBoardPitch
        let zoomAction = SCNAction.customAction(duration: 1.0) { [weak self] _, elapsed in
            guard let self else { return }
            let t = Float(min(1, elapsed))
            let eased = t * t * (3 - 2 * t)
            let targetDistance: Float = 15.0
            self.cameraDistance = startDistance + (targetDistance - startDistance) * eased
            self.currentPitch = startPitch + (targetPitch - startPitch) * eased
            self.updateCameraPosition()
            self.updateBoardRotation()
        }
        let spin = SCNAction.repeatForever(
            SCNAction.rotateBy(x: 0, y: .pi / 3, z: 0, duration: 2.0)
        )
        let startSpin = SCNAction.run { [weak self] _ in
            self?.boardNode.runAction(spin, forKey: "gameOverSpin")
        }
        cameraNode.runAction(.sequence([zoomAction, startSpin]), forKey: "gameOverZoom")
    }

    func stopGameOverAnimation() {
        cameraNode.removeAction(forKey: "gameOverZoom")
        boardNode.removeAction(forKey: "gameOverSpin")
    }

    func startNewGame() {
        currentPlayer = .red
        onTurnUpdate?(currentPlayer)
        onMessageUpdate?("")
        startAIOnIntroComplete = true
        introCompleted = false
        buildBoard()
        animateInitialZoomIn()
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.onTurnUpdate?(self.currentPlayer)
            self.onActiveUpdate?(self.currentPlayer)
            self.onMessageUpdate?("")
            self.updateScores()
        }
    }

    func currentPlayerState() -> TileState {
        currentPlayer
    }

    private func advanceTurn() {
        currentPlayer = currentPlayer.opposite
        onTurnUpdate?(currentPlayer)
        onActiveUpdate?(currentPlayer)
        if gameMode == .hexello {
            checkForNoMoves()
        } else {
            onMessageUpdate?("")
        }
        isAnimatingMove = false
        performAITurnIfNeeded()
    }

    private func checkForNoMoves() {
        if !hasLegalMove(for: currentPlayer) {
            let blockedPlayer = currentPlayer
            onMessageUpdate?(blockedPlayer == .red ? "No Red Move Possible" : "No Blue Move Possible")
            currentPlayer = currentPlayer.opposite
            onTurnUpdate?(currentPlayer)
            onActiveUpdate?(currentPlayer)
            if !hasLegalMove(for: currentPlayer) {
                onMessageUpdate?("No Moves Possible")
                let (_, winner) = gameOverMessage()
                onGameOver?("No Moves Possible", winner)
                isGameOver = true
            }
        } else {
            onMessageUpdate?("")
        }
    }

    private func checkForHexpandWinIfNeeded() -> Bool {
        guard gameMode == .hexpand else { return false }
        guard !isGameOver else { return false }
        let redCount = tileStates.values.filter { $0 == .red }.count
        let blueCount = tileStates.values.filter { $0 == .blue }.count
        if redCount == 0, blueCount > 0, redHadTile {
            let (message, winner) = gameOverMessage()
            onMessageUpdate?(message)
            onGameOver?(message, winner)
            isGameOver = true
            aiMoveScheduled = false
            return true
        }
        if blueCount == 0, redCount > 0, blueHadTile {
            let (message, winner) = gameOverMessage()
            onMessageUpdate?(message)
            onGameOver?(message, winner)
            isGameOver = true
            aiMoveScheduled = false
            return true
        }
        return false
    }

    private func gameOverMessage() -> (String, String) {
        let redCount = tileStates.values.filter { $0 == .red }.count
        let blueCount = tileStates.values.filter { $0 == .blue }.count
        if redCount == blueCount {
            return ("Game over", "Draw")
        }
        let winner = redCount > blueCount ? "Red wins" : "Blue wins"
        let message = "Game over: \(redCount)–\(blueCount)"
        return (message, winner)
    }

    private func hasLegalMove(for player: TileState) -> Bool {
        !legalMoves(for: player).isEmpty
    }

    private func performAITurnIfNeeded() {
        let aiEnabled = (currentPlayer == .red && aiRedEnabled) || (currentPlayer == .blue && aiBlueEnabled)
        guard aiEnabled, !isAnimatingMove, introCompleted, !aiMoveScheduled, !isGameOver else { return }

        if gameMode == .hexello {
            if let move = bestMove(for: currentPlayer) {
                scheduleAIMove(move)
            }
            return
        }

        scheduleHexpandAIMove()
    }

    private func bestMove(for player: TileState) -> AxialCoord? {
        let state = tileStates
        let moves = legalMoves(for: player, in: state)
        guard !moves.isEmpty else { return nil }

        if aiDifficulty == .easy {
            let scored = moves.map { (coord: $0, flips: flipsForMove(at: $0, player: player, in: state).count) }
            let bestFlips = scored.map(\.flips).max() ?? 0
            let nearBest = scored.filter { $0.flips >= max(1, bestFlips - 1) }.map(\.coord)
            return nearBest.randomElement() ?? scored.max { $0.flips < $1.flips }?.coord
        }

        let depth = hexelloSearchDepth(in: state)
        var bestMoves: [AxialCoord] = []
        var bestScore = Int.min
        var alpha = Int.min
        let beta = Int.max

        for move in orderedHexelloMoves(for: player, in: state) {
            let next = simulateHexelloMove(at: move, player: player, in: state)
            var branchAlpha = alpha
            var branchBeta = beta
            let score = minimaxHexello(
                state: next,
                player: player.opposite,
                maximizingFor: player,
                depth: depth - 1,
                alpha: &branchAlpha,
                beta: &branchBeta
            )
            if score > bestScore {
                bestScore = score
                bestMoves = [move]
            } else if score == bestScore {
                bestMoves.append(move)
            }
            alpha = max(alpha, bestScore)
        }

        return bestMoves.randomElement()
    }

    private func legalMoves(for player: TileState) -> [AxialCoord] {
        legalMoves(for: player, in: tileStates)
    }

    private func legalMoves(for player: TileState, in state: [AxialCoord: TileState]) -> [AxialCoord] {
        state.compactMap { coord, tileState in
            guard tileState == .empty else { return nil }
            return flipsForMove(at: coord, player: player, in: state).isEmpty ? nil : coord
        }
    }

    private func simulateHexelloMove(
        at coord: AxialCoord,
        player: TileState,
        in state: [AxialCoord: TileState]
    ) -> [AxialCoord: TileState] {
        var next = state
        next[coord] = player
        for flipped in flipsForMove(at: coord, player: player, in: state) {
            next[flipped] = player
        }
        return next
    }

    private func minimaxHexello(
        state: [AxialCoord: TileState],
        player: TileState,
        maximizingFor: TileState,
        depth: Int,
        alpha: inout Int,
        beta: inout Int
    ) -> Int {
        let moves = orderedHexelloMoves(for: player, in: state)
        let opponentHasMove = !legalMoves(for: player.opposite, in: state).isEmpty
        if moves.isEmpty, !opponentHasMove {
            return terminalHexelloScore(state: state, for: maximizingFor, depth: depth)
        }
        if depth == 0 {
            return evaluateHexello(state: state, for: maximizingFor)
        }
        if moves.isEmpty {
            return minimaxHexello(
                state: state,
                player: player.opposite,
                maximizingFor: maximizingFor,
                depth: depth - 1,
                alpha: &alpha,
                beta: &beta
            )
        }

        if player == maximizingFor {
            var best = Int.min
            var localAlpha = alpha
            let localBeta = beta
            for move in moves {
                let next = simulateHexelloMove(at: move, player: player, in: state)
                var childAlpha = localAlpha
                var childBeta = localBeta
                let score = minimaxHexello(
                    state: next,
                    player: player.opposite,
                    maximizingFor: maximizingFor,
                    depth: depth - 1,
                    alpha: &childAlpha,
                    beta: &childBeta
                )
                best = max(best, score)
                localAlpha = max(localAlpha, best)
                if localAlpha >= localBeta {
                    break
                }
            }
            alpha = localAlpha
            return best
        }

        var best = Int.max
        let localAlpha = alpha
        var localBeta = beta
        for move in moves {
            let next = simulateHexelloMove(at: move, player: player, in: state)
            var childAlpha = localAlpha
            var childBeta = localBeta
            let score = minimaxHexello(
                state: next,
                player: player.opposite,
                maximizingFor: maximizingFor,
                depth: depth - 1,
                alpha: &childAlpha,
                beta: &childBeta
            )
            best = min(best, score)
            localBeta = min(localBeta, best)
            if localAlpha >= localBeta {
                break
            }
        }
        beta = localBeta
        return best
    }

    private func evaluateHexello(state: [AxialCoord: TileState], for player: TileState) -> Int {
        let opponent = player.opposite
        let corners = cornerCoordinates()
        let emptyCount = state.values.filter { $0 == .empty }.count
        let playerMobility = legalMoves(for: player, in: state).count
        let opponentMobility = legalMoves(for: opponent, in: state).count

        var playerCount = 0
        var opponentCount = 0
        var playerCorners = 0
        var opponentCorners = 0
        var playerEdges = 0
        var opponentEdges = 0

        for (coord, owner) in state {
            if owner == player {
                playerCount += 1
                if corners.contains(coord) {
                    playerCorners += 1
                } else if isEdge(coord) {
                    playerEdges += 1
                }
            } else if owner == opponent {
                opponentCount += 1
                if corners.contains(coord) {
                    opponentCorners += 1
                } else if isEdge(coord) {
                    opponentEdges += 1
                }
            }
        }

        var playerRisk = 0
        var opponentRisk = 0
        for corner in corners where state[corner] == .empty {
            for neighbor in neighbors(of: corner, in: state) {
                if state[neighbor] == player {
                    playerRisk += 1
                } else if state[neighbor] == opponent {
                    opponentRisk += 1
                }
            }
        }

        let discDiff = playerCount - opponentCount
        let mobilityDiff = playerMobility - opponentMobility
        let cornerDiff = playerCorners - opponentCorners
        let edgeDiff = playerEdges - opponentEdges
        let riskDiff = opponentRisk - playerRisk

        let discWeight: Int
        let mobilityWeight: Int
        let cornerWeight: Int
        let edgeWeight: Int
        let riskWeight: Int

        if emptyCount > 40 {
            discWeight = 1
            mobilityWeight = 18
            cornerWeight = 120
            edgeWeight = 4
            riskWeight = 24
        } else if emptyCount > 18 {
            discWeight = 2
            mobilityWeight = 14
            cornerWeight = 140
            edgeWeight = 5
            riskWeight = 20
        } else {
            discWeight = 6
            mobilityWeight = 10
            cornerWeight = 160
            edgeWeight = 6
            riskWeight = 16
        }

        return discDiff * discWeight
            + mobilityDiff * mobilityWeight
            + cornerDiff * cornerWeight
            + edgeDiff * edgeWeight
            + riskDiff * riskWeight
    }

    private func orderedHexelloMoves(for player: TileState, in state: [AxialCoord: TileState]) -> [AxialCoord] {
        legalMoves(for: player, in: state).sorted { lhs, rhs in
            let lhsScore = hexelloMoveOrderingScore(lhs, player: player, in: state)
            let rhsScore = hexelloMoveOrderingScore(rhs, player: player, in: state)
            if lhsScore != rhsScore {
                return lhsScore > rhsScore
            }
            if lhs.q != rhs.q {
                return lhs.q < rhs.q
            }
            return lhs.r < rhs.r
        }
    }

    private func hexelloMoveOrderingScore(
        _ coord: AxialCoord,
        player: TileState,
        in state: [AxialCoord: TileState]
    ) -> Int {
        var score = flipsForMove(at: coord, player: player, in: state).count
        if isCorner(coord) {
            score += 1000
        } else if isEdge(coord) {
            score += 80
        }
        if isAdjacentToEmptyCorner(coord, in: state) {
            score -= 160
        }
        return score
    }

    private func terminalHexelloScore(state: [AxialCoord: TileState], for player: TileState, depth: Int) -> Int {
        var playerCount = 0
        var opponentCount = 0
        for owner in state.values {
            if owner == player {
                playerCount += 1
            } else if owner == player.opposite {
                opponentCount += 1
            }
        }
        if playerCount == opponentCount {
            return 0
        }
        return playerCount > opponentCount ? (10000 + depth) : (-10000 - depth)
    }

    private func hexelloSearchDepth(in state: [AxialCoord: TileState]) -> Int {
        let emptyCount = state.values.filter { $0 == .empty }.count
        if emptyCount <= 14 {
            return 5
        }
        if emptyCount <= 28 {
            return sideLength >= 6 ? 4 : 5
        }
        return sideLength >= 6 ? 3 : 4
    }

    private func cornerCoordinates() -> [AxialCoord] {
        let r = radius
        return [
            AxialCoord(q: r, r: 0),
            AxialCoord(q: 0, r: r),
            AxialCoord(q: -r, r: r),
            AxialCoord(q: -r, r: 0),
            AxialCoord(q: 0, r: -r),
            AxialCoord(q: r, r: -r)
        ]
    }

    private func neighbors(of coord: AxialCoord, in state: [AxialCoord: TileState]) -> [AxialCoord] {
        HexGrid.directions.compactMap { direction in
            let neighbor = coord.adding(direction)
            return state[neighbor] == nil ? nil : neighbor
        }
    }

    private func isAdjacentToEmptyCorner(_ coord: AxialCoord, in state: [AxialCoord: TileState]) -> Bool {
        for corner in cornerCoordinates() where state[corner] == .empty {
            if neighbors(of: corner, in: state).contains(coord) {
                return true
            }
        }
        return false
    }

    // MARK: - Hexpand AI
    private struct HexpandState {
        var owners: [AxialCoord: TileState]
        var levels: [AxialCoord: Int]
        var turnsPlayed: Int
    }

    private struct HexpandSummary {
        var aiScore = 0
        var humanScore = 0
        var aiLevel5 = 0
        var humanLevel5 = 0
        var aiPieces = 0
        var humanPieces = 0
    }

    private struct HexpandTTKey: Hashable {
        let stateHash: UInt64
        let playerCode: UInt8
        let maximizingCode: UInt8
    }

    private struct HexpandTTEntry {
        let depth: Int
        let score: Int
        let bound: HexpandTTBound
        let bestMove: AxialCoord?
    }

    private enum HexpandTTBound {
        case exact
        case lower
        case upper
    }

    private func bestHexpandMove(for player: TileState, depth: Int) -> AxialCoord? {
        bestHexpandMove(for: player, depth: depth, in: currentHexpandState())
    }

    private func bestHexpandEasyMove(for player: TileState, in state: HexpandState) -> AxialCoord? {
        let moves = orderedHexpandMoves(for: player, in: state)
        guard !moves.isEmpty else { return nil }
        let scored = moves
            .map { (coord: $0, score: scoreHexpandEasyMove($0, player: player, in: state)) }
            .sorted { lhs, rhs in
                if lhs.score != rhs.score { return lhs.score > rhs.score }
                if lhs.coord.q != rhs.coord.q { return lhs.coord.q < rhs.coord.q }
                return lhs.coord.r < rhs.coord.r
            }

        guard scored.count > 1 else { return scored.first?.coord }

        // Easy mode still makes suboptimal choices, but less often.
        let roll = Int.random(in: 0..<100)
        if roll < 15 {
            return scored.randomElement()?.coord
        }
        if roll < 45 {
            let upperBucketCount = max(2, (scored.count * 2) / 3)
            return scored[..<upperBucketCount].randomElement()?.coord
        }
        let topCount = min(2, scored.count)
        return scored[..<topCount].randomElement()?.coord
    }

    private func scoreHexpandEasyMove(_ move: AxialCoord, player: TileState, in state: HexpandState) -> Int {
        let ownerBefore = state.owners[move] ?? .empty
        let levelBefore = state.levels[move] ?? 0
        let next = simulateHexpandMove(at: move, player: player, in: state)
        let summary = summarizeHexpand(state: next, for: player)
        var score = Int.random(in: -2...2)
        score += summary.aiScore - summary.humanScore
        score += summary.aiLevel5 - summary.humanLevel5

        if ownerBefore == player {
            score += 2 + levelBefore
        } else {
            score += 1
        }

        if levelBefore >= 5 {
            score += 4
        }

        let opponent = player.opposite
        let opponentCriticalCount = hexpandLegalMoves(for: opponent, in: next).reduce(0) { partial, coord in
            guard (next.owners[coord] ?? .empty) == opponent else { return partial }
            return partial + (((next.levels[coord] ?? 0) >= 5) ? 1 : 0)
        }
        score -= opponentCriticalCount * 2

        let ownCriticalCount = hexpandLegalMoves(for: player, in: next).reduce(0) { partial, coord in
            guard (next.owners[coord] ?? .empty) == player else { return partial }
            return partial + (((next.levels[coord] ?? 0) >= 5) ? 1 : 0)
        }
        score += ownCriticalCount

        if isEdge(move) { score -= 1 }
        if isCorner(move) { score -= 1 }

        return score
    }

    private func bestHexpandMove(for player: TileState, depth: Int, in state: HexpandState) -> AxialCoord? {
        hexpandTranspositionTable.removeAll(keepingCapacity: true)

        let moves = orderedHexpandMoves(for: player, in: state)
        guard !moves.isEmpty else { return nil }
        var bestMoves: [AxialCoord] = []
        var bestScore = Int.min
        var alpha = Int.min
        let beta = Int.max

        for move in moves {
            let next = simulateHexpandMove(at: move, player: player, in: state)
            var branchAlpha = alpha
            var branchBeta = beta
            let score = minimax(
                state: next,
                player: player.opposite,
                depth: depth - 1,
                maximizingFor: player,
                alpha: &branchAlpha,
                beta: &branchBeta
            )
            if score > bestScore {
                bestScore = score
                bestMoves = [move]
            } else if score == bestScore {
                bestMoves.append(move)
            }
            alpha = max(alpha, bestScore)
        }

        return bestMoves.randomElement()
    }

    private func minimax(
        state: HexpandState,
        player: TileState,
        depth: Int,
        maximizingFor: TileState,
        alpha: inout Int,
        beta: inout Int
    ) -> Int {
        if let terminal = terminalHexpandScore(state: state, for: maximizingFor, depth: depth) {
            return terminal
        }
        if depth == 0 {
            return evaluateHexpand(state: state, for: maximizingFor)
        }

        let cacheKey = HexpandTTKey(
            stateHash: hashHexpandState(state),
            playerCode: tileStateCode(player),
            maximizingCode: tileStateCode(maximizingFor)
        )
        var preferredMove: AxialCoord?
        if let entry = hexpandTranspositionTable[cacheKey] {
            preferredMove = entry.bestMove
            if entry.depth >= depth {
                switch entry.bound {
                case .exact:
                    return entry.score
                case .lower:
                    alpha = max(alpha, entry.score)
                case .upper:
                    beta = min(beta, entry.score)
                }
                if alpha >= beta {
                    return entry.score
                }
            }
        }
        let alphaStart = alpha
        let betaStart = beta

        let moves = orderedHexpandMoves(for: player, in: state, preferredMove: preferredMove)
        if moves.isEmpty {
            return evaluateHexpand(state: state, for: maximizingFor)
        }

        let maximizingTurn = player == maximizingFor
        var best = maximizingTurn ? Int.min : Int.max
        var bestMove: AxialCoord?
        var localAlpha = alpha
        var localBeta = beta
        for move in moves {
            let next = simulateHexpandMove(at: move, player: player, in: state)
            var childAlpha = localAlpha
            var childBeta = localBeta
            let score = minimax(
                state: next,
                player: player.opposite,
                depth: depth - 1,
                maximizingFor: maximizingFor,
                alpha: &childAlpha,
                beta: &childBeta
            )
            if maximizingTurn {
                if score > best {
                    best = score
                    bestMove = move
                }
                localAlpha = max(localAlpha, best)
            } else {
                if score < best {
                    best = score
                    bestMove = move
                }
                localBeta = min(localBeta, best)
            }
            if localAlpha >= localBeta {
                break
            }
        }
        alpha = localAlpha
        beta = localBeta
        let bound: HexpandTTBound
        if best <= alphaStart {
            bound = .upper
        } else if best >= betaStart {
            bound = .lower
        } else {
            bound = .exact
        }
        hexpandTranspositionTable[cacheKey] = HexpandTTEntry(
            depth: depth,
            score: best,
            bound: bound,
            bestMove: bestMove
        )
        return best
    }

    private func evaluateHexpand(state: HexpandState, for player: TileState) -> Int {
        let summary = summarizeHexpand(state: state, for: player)
        return (summary.aiScore - summary.humanScore) + 2 * (summary.aiLevel5 - summary.humanLevel5)
    }

    private func currentHexpandState() -> HexpandState {
        HexpandState(owners: tileStates, levels: hexpandLevels, turnsPlayed: movesMade)
    }

    private func hexpandLegalMoves(for player: TileState) -> [AxialCoord] {
        hexpandLegalMoves(for: player, in: currentHexpandState())
    }

    private func hexpandLegalMoves(for player: TileState, in state: HexpandState) -> [AxialCoord] {
        let coords = boardCoords.isEmpty ? Array(state.owners.keys) : boardCoords
        return coords.filter { coord in
            let owner = state.owners[coord] ?? .empty
            return owner == .empty || owner == player
        }
    }

    private func simulateHexpandMove(at coord: AxialCoord, player: TileState, in state: HexpandState) -> HexpandState {
        var next = state
        next.turnsPlayed += 1
        next.owners[coord] = player
        next.levels[coord] = (next.levels[coord] ?? 0) + 1
        if (next.levels[coord] ?? 0) > 5 {
            triggerHexpandExplosionSim(startingAt: coord, in: &next)
        }
        return next
    }

    private func triggerHexpandExplosionSim(startingAt coord: AxialCoord, in state: inout HexpandState) {
        var queue: [AxialCoord] = [coord]
        var queueHead = 0
        var queued: Set<AxialCoord> = [coord]

        while queueHead < queue.count {
            let current = queue[queueHead]
            queueHead += 1
            queued.remove(current)
            let currentOwner = state.owners[current] ?? .empty
            let currentLevel = state.levels[current] ?? 0
            if currentOwner == .empty || currentLevel <= 5 { continue }
            let explodingPlayer = currentOwner

            state.owners[current] = .empty
            state.levels[current] = 0

            for direction in HexGrid.directions {
                let neighbor = current.adding(direction)
                guard state.owners[neighbor] != nil else { continue }
                state.owners[neighbor] = explodingPlayer
                state.levels[neighbor] = (state.levels[neighbor] ?? 0) + 1
                if (state.levels[neighbor] ?? 0) > 5, !queued.contains(neighbor) {
                    queue.append(neighbor)
                    queued.insert(neighbor)
                }
            }
        }
    }

    private func orderedHexpandMoves(
        for player: TileState,
        in state: HexpandState,
        preferredMove: AxialCoord? = nil
    ) -> [AxialCoord] {
        hexpandLegalMoves(for: player, in: state).sorted { lhs, rhs in
            let lhsScore = hexpandMoveOrderingScore(lhs, player: player, in: state, preferredMove: preferredMove)
            let rhsScore = hexpandMoveOrderingScore(rhs, player: player, in: state, preferredMove: preferredMove)
            if lhsScore != rhsScore {
                return lhsScore > rhsScore
            }
            if lhs.q != rhs.q {
                return lhs.q < rhs.q
            }
            return lhs.r < rhs.r
        }
    }

    private func hexpandMoveOrderingScore(
        _ coord: AxialCoord,
        player: TileState,
        in state: HexpandState,
        preferredMove: AxialCoord?
    ) -> Int {
        var score = 0
        if coord == preferredMove {
            score += 10_000
        }

        let owner = state.owners[coord] ?? .empty
        let level = state.levels[coord] ?? 0
        if owner == player {
            score += 120 + level * 18
            if level >= 5 {
                score += 350
            }
        } else {
            score += 40
        }

        if owner == player, level + 1 > 5 {
            score += 500
        }

        let x = coord.q
        let z = coord.r
        let y = -x - z
        let distanceFromCenter = max(abs(x), max(abs(y), abs(z)))
        score += max(0, radius - distanceFromCenter)
        return score
    }

    private func tileStateCode(_ state: TileState) -> UInt8 {
        switch state {
        case .empty:
            return 0
        case .red:
            return 1
        case .blue:
            return 2
        }
    }

    private func hashHexpandState(_ state: HexpandState) -> UInt64 {
        let coords = boardCoords.isEmpty ? Array(state.owners.keys) : boardCoords
        var hash: UInt64 = 14_695_981_039_346_656_037
        for coord in coords {
            let owner = state.owners[coord] ?? .empty
            let level = state.levels[coord] ?? 0
            let encoded = UInt64(tileStateCode(owner) | UInt8((max(0, min(7, level))) << 2))
            hash ^= encoded &* 1_099_511_628_211
            hash = hash &* 1_099_511_628_211
        }
        hash ^= UInt64(state.turnsPlayed & 0xFFFF)
        return hash
    }

    private func summarizeHexpand(state: HexpandState, for player: TileState) -> HexpandSummary {
        var summary = HexpandSummary()
        let opponent = player.opposite
        for (coord, owner) in state.owners {
            let level = state.levels[coord] ?? 0
            if owner == player {
                summary.aiScore += level
                summary.aiPieces += 1
                if level == 5 {
                    summary.aiLevel5 += 1
                }
            } else if owner == opponent {
                summary.humanScore += level
                summary.humanPieces += 1
                if level == 5 {
                    summary.humanLevel5 += 1
                }
            }
        }
        return summary
    }

    private func terminalHexpandScore(state: HexpandState, for player: TileState, depth: Int) -> Int? {
        let summary = summarizeHexpand(state: state, for: player)
        if summary.aiScore >= 75 || summary.humanScore >= 75 {
            if summary.aiScore > summary.humanScore {
                return 10000 + depth
            }
            if summary.humanScore > summary.aiScore {
                return -10000 - depth
            }
            return 0
        }

        if state.turnsPlayed >= 2 {
            if summary.aiPieces > 0 && summary.humanPieces == 0 {
                return 10000 + depth
            }
            if summary.humanPieces > 0 && summary.aiPieces == 0 {
                return -10000 - depth
            }
        }
        return nil
    }

    private func isEdge(_ coord: AxialCoord) -> Bool {
        let x = coord.q
        let z = coord.r
        let y = -x - z
        let r = radius
        return max(abs(x), max(abs(y), abs(z))) == r
    }

    private func isCorner(_ coord: AxialCoord) -> Bool {
        let r = radius
        return (coord.q == r && coord.r == 0)
            || (coord.q == 0 && coord.r == r)
            || (coord.q == -r && coord.r == r)
            || (coord.q == -r && coord.r == 0)
            || (coord.q == 0 && coord.r == -r)
            || (coord.q == r && coord.r == -r)
    }

    private func buildCamera() {
        let camera = SCNCamera()
        camera.fieldOfView = 55
        camera.zNear = 0.1
        camera.zFar = 100
        camera.wantsHDR = false
        camera.wantsExposureAdaptation = false
        cameraNode.camera = camera
        updateCameraPosition()
        cameraNode.look(at: SCNVector3Zero)
        scene.rootNode.addChildNode(cameraNode)
    }

    private func buildLights() {
        let ambient = SCNLight()
        ambient.type = .ambient
        ambient.intensity = 350
        ambient.color = UIColor(white: 0.5, alpha: 1.0)
        let ambientNode = SCNNode()
        ambientNode.light = ambient
        scene.rootNode.addChildNode(ambientNode)

        let keyLight = SCNLight()
        keyLight.type = .directional
        keyLight.intensity = 900
        keyLight.castsShadow = true
        keyLight.shadowRadius = 10
        let keyNode = SCNNode()
        keyNode.light = keyLight
        keyNode.eulerAngles = SCNVector3(-0.9, 0.6, 0)
        scene.rootNode.addChildNode(keyNode)

        let fillLight = SCNLight()
        fillLight.type = .omni
        fillLight.intensity = 350
        fillLight.color = UIColor(white: 0.8, alpha: 1.0)
        let fillNode = SCNNode()
        fillNode.light = fillLight
        fillNode.position = SCNVector3(-4, 6, -4)
        scene.rootNode.addChildNode(fillNode)
    }

    private func updateBoardRotation() {
        tiltNode.eulerAngles.x = currentPitch
        boardNode.eulerAngles.y = currentYaw
        onPitchUpdate?(currentPitch)
        onYawUpdate?(currentYaw)
    }

    private func updateCameraPosition() {
        // Camera direction normalized from the original (0, 6.5, 9.5).
        let dir = SCNVector3(0, 0.57, 0.82)
        cameraNode.position = SCNVector3(
            dir.x * cameraDistance,
            dir.y * cameraDistance,
            dir.z * cameraDistance
        )
        cameraNode.look(at: SCNVector3Zero)
    }

    private func animateInitialZoomIn() {
        let action = SCNAction.customAction(duration: introAnimationDuration) { [weak self] _, elapsed in
            guard let self else { return }
            let t = Float(min(1, elapsed / introAnimationDuration))
            let eased = t * t * (3 - 2 * t)
            self.cameraDistance = self.initialCameraDistance + (self.defaultCameraDistance - self.initialCameraDistance) * eased
            self.currentYaw = eased * (Float.pi / 3)
            self.updateCameraPosition()
            self.updateBoardRotation()
        }
        let finish = SCNAction.run { [weak self] _ in
            guard let self else { return }
            self.introCompleted = true
            if self.startAIOnIntroComplete {
                self.startAIOnIntroComplete = false
                DispatchQueue.main.async { [weak self] in
                    self?.performAITurnIfNeeded()
                }
            }
        }
        cameraNode.runAction(.sequence([action, finish]), forKey: "initialZoom")
    }

    private func startInertia() {
        guard displayLink == nil else { return }
        let link = CADisplayLink(target: self, selector: #selector(handleInertiaTick))
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    @objc private func handleInertiaTick() {
        angularVelocityYaw *= angularDamping
        angularVelocityPitch *= angularDamping

        if abs(angularVelocityYaw) < 0.0005 && abs(angularVelocityPitch) < 0.0005 {
            displayLink?.invalidate()
            displayLink = nil
            return
        }

        currentYaw += angularVelocityYaw
        currentPitch = clamp(currentPitch + angularVelocityPitch, min: minPitch, max: maxPitch)
        updateBoardRotation()
    }

    private func playPop() {
        guard let popAudio else { return }
        boardNode.runAction(SCNAction.playAudio(popAudio, waitForCompletion: false))
    }

    private func playFlip() {
        guard let flipAudio else { return }
        boardNode.runAction(SCNAction.playAudio(flipAudio, waitForCompletion: false))
    }

    private func playPlace() {
        guard let placeAudio else { return }
        boardNode.runAction(SCNAction.playAudio(placeAudio, waitForCompletion: false))
    }

    private func clamp(_ value: Float, min: Float, max: Float) -> Float {
        Swift.max(min, Swift.min(value, max))
    }

    private var hasPendingHexpandExplosions: Bool {
        hexpandExplosionQueueHead < hexpandExplosionQueue.count
    }

    private func trimHexpandExplosionQueueIfNeeded() {
        guard hexpandExplosionQueueHead > 32 else { return }
        guard hexpandExplosionQueueHead * 2 >= hexpandExplosionQueue.count else { return }
        hexpandExplosionQueue.removeFirst(hexpandExplosionQueueHead)
        hexpandExplosionQueueHead = 0
    }

    private func scheduleAIMove(_ move: AxialCoord) {
        let delay: TimeInterval = movesMade == 0 ? 1.0 : 0.4
        aiMoveScheduled = true
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            self?.applyMove(at: move)
        }
    }

    private func scheduleHexpandAIMove() {
        let delay: TimeInterval = movesMade == 0 ? 1.0 : 0.4
        let player = currentPlayer
        let useEasyAI: Bool
        switch aiDifficulty {
        case .easy:
            useEasyAI = true
        case .hard:
            useEasyAI = false
        }
        let depth = hexpandSearchDepth()
        let stateSnapshot = currentHexpandState()
        let token = aiComputationToken
        aiMoveScheduled = true

        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            self.aiComputeQueue.async { [weak self] in
                guard let self else { return }
                let move: AxialCoord?
                if useEasyAI {
                    move = self.bestHexpandEasyMove(for: player, in: stateSnapshot)
                } else {
                    move = self.bestHexpandMove(for: player, depth: depth, in: stateSnapshot)
                }
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    defer { self.aiMoveScheduled = false }
                    guard self.aiComputationToken == token else { return }
                    guard self.currentPlayer == player else { return }
                    let aiEnabled = (player == .red && self.aiRedEnabled) || (player == .blue && self.aiBlueEnabled)
                    guard aiEnabled, !self.isAnimatingMove, self.introCompleted else { return }
                    guard let move else { return }
                    self.applyMove(at: move)
                }
            }
        }
    }

    private func hexpandSearchDepth() -> Int {
        guard aiDifficulty == .hard else {
            return 2
        }

        let emptyCount = tileStates.values.reduce(into: 0) { partial, state in
            if state == .empty {
                partial += 1
            }
        }

        if sideLength >= 4 {
            return emptyCount <= 14 ? 6 : 5
        }
        return 6
    }
}
