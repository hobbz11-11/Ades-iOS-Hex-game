import SceneKit
import UIKit

enum GameMode: String {
    case hexello
    case hexpand
    case hexfection
}

enum AIDifficulty {
    case easy
    case medium
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
    private let hexfectionTileSpacing: Float = 1.0

    private var gameMode: GameMode = .hexello
    private var currentYaw: Float = 0
    private let defaultBoardPitch: Float = 0
    private var currentPitch: Float = 0
    private let minPitch: Float = -34 * .pi / 180
    private let maxPitch: Float = 55 * .pi / 180
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
    private let jumpAudio: SCNAudioSource? = {
        guard let source = SCNAudioSource(named: "jump.wav") else { return nil }
        source.load()
        source.volume = 0.85
        source.isPositional = false
        return source
    }()
    private let splitAudio: SCNAudioSource? = {
        guard let source = SCNAudioSource(named: "split.wav") else { return nil }
        source.load()
        source.volume = 0.85
        source.isPositional = false
        return source
    }()
    private let raiseAudio: SCNAudioSource? = {
        guard let source = SCNAudioSource(named: "raise.wav") else { return nil }
        source.load()
        source.volume = 0.8
        source.isPositional = false
        return source
    }()
    private let takeoverAudio: SCNAudioSource? = {
        guard let source = SCNAudioSource(named: "pop3.wav") else { return nil }
        source.load()
        source.volume = 0.85
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
    private let hexelloMoveHintBorderBaseName = "hexelloMoveHintBorder"
    private lazy var hexelloMoveHintBorderGeometry: SCNGeometry = makeHexelloMoveHintBorderGeometry()
    private let hexelloMoveHighlightScale: Float = 0.94
    private let hexelloMoveHighlightDuration: TimeInterval = 0.12
    private let hexelloMoveTileTintBlend: CGFloat = 0.22
    private let hexelloMoveBorderThicknessScale: Float = 0.1
    private let hexelloMoveBorderTopInset: Float = 0.004
    private var hexfectionPieceNodes: [AxialCoord: SCNNode] = [:]
    private var hexfectionTransformedTiles: Set<AxialCoord> = []
    private var hexfectionTintedTiles: Set<AxialCoord> = []
    private var hexfectionSelectedSource: AxialCoord?
    private var hexfectionCloneTargets: Set<AxialCoord> = []
    private var hexfectionLeapTargets: Set<AxialCoord> = []
    private let hexfectionPieceBaseRadiusFactor: CGFloat = 0.13
    private let hexfectionPieceDiameterScale: Float = 5.0
    private let hexfectionPieceHeightScale: Float = 1.5
    private let hexfectionTileHighlightScale: Float = 0.92
    private let hexfectionTileLiftAmount: Float = 0.0
    private let hexfectionTileHighlightDuration: TimeInterval = 0.12
    private let hexfectionCloneTileTintBlend: CGFloat = 0.5
    private let hexfectionLeapTileTintBlend: CGFloat = 0.25
    private let hexfectionMoveAnimationDuration: TimeInterval = 1.0
    private let hexfectionCaptureAnimationDuration: TimeInterval = 1.0
    private let hexfectionCaptureWaveDuration: TimeInterval = 1.0
    private let hexfectionCaptureWaveLeadDelay: TimeInterval = 0.0
    private let hexfectionCaptureWaveOuterRadiusFactor: CGFloat = 0.30
    private let hexfectionCaptureWaveInnerRadiusFactor: CGFloat = 0.08
    private let hexfectionCaptureWaveHeight: CGFloat = 0.06
    private let hexfectionCaptureWaveThickness: CGFloat = 0.12
    private let hexfectionCaptureWaveLayerCount: Int = 5
    private let hexfectionCaptureFadeDuration: TimeInterval = 0.3
    private let hexfectionLeapArcHeight: Float = 0.8
    private var isHexpandExploding = false
    private let hexpandMaxLevel = 6
    private let hexpandPrimedLevel = 5
    private let hexpandPrimedMinScale: CGFloat = 0.8
    private let hexpandPrimedMaxScale: CGFloat = 1.0
    private let hexpandPrimedSpacingDelta: CGFloat = 0.1
    private let hexpandPrimedPulsePeriod: TimeInterval = 1.0
    private var currentPlayer: TileState = .red
    private let nodeCoordTagPrefix = "|coord:"
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
    var onTurnCountUpdate: ((Int) -> Void)?
    var onGameOver: ((String, String) -> Void)?
    var onActiveUpdate: ((TileState) -> Void)?
    var onAIThinkingUpdate: ((TileState) -> Void)?
    private var aiRedEnabled = false
    private var aiBlueEnabled = false
    private var aiComputationToken = UUID()
    private let aiComputeQueue = DispatchQueue(label: "adesmegagame.hexpand.ai", qos: .userInitiated)
    private let hexplodeNeuralEngine = HexplodeNeuralEngine()
    private var movesMade = 0
    private var startAIOnIntroComplete = false
    private var introCompleted = false
    private let introAnimationDuration: TimeInterval = 2.5
    private var boardCoords: [AxialCoord] = []

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

    private func setAIScheduling(_ scheduled: Bool, player: TileState = .empty) {
        aiMoveScheduled = scheduled
        onAIThinkingUpdate?(scheduled ? player : .empty)
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

        if let coord = coordFromNode(hits.first?.node) {
            applyMove(at: coord)
        }
    }

    private func coordFromNode(_ startNode: SCNNode?) -> AxialCoord? {
        var current = startNode
        while let node = current {
            if let coord = coordFromNodeName(node.name) {
                return coord
            }
            if let tile = node as? HexTileNode {
                return tile.coord
            }
            current = node.parent
        }
        return nil
    }

    private func coordFromNodeName(_ name: String?) -> AxialCoord? {
        guard let name else { return nil }
        guard let range = name.range(of: nodeCoordTagPrefix) else { return nil }
        let payload = name[range.upperBound...]
        let parts = payload.split(separator: ",", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2,
              let q = Int(parts[0]),
              let r = Int(parts[1]) else {
            return nil
        }
        return AxialCoord(q: q, r: r)
    }

    private func setNodeCoordMetadata(_ node: SCNNode, coord: AxialCoord) {
        let baseName: String
        if let name = node.name, let range = name.range(of: nodeCoordTagPrefix) {
            baseName = String(name[..<range.lowerBound])
        } else {
            baseName = node.name ?? ""
        }
        node.name = "\(baseName)\(nodeCoordTagPrefix)\(coord.q),\(coord.r)"
    }

    private var canAcceptHumanInput: Bool {
        guard introCompleted, !isGameOver, !isAnimatingMove, !aiMoveScheduled else { return false }
        return !isAIEnabled(for: currentPlayer)
    }

    private func isAIEnabled(for player: TileState) -> Bool {
        return (player == .red && aiRedEnabled) || (player == .blue && aiBlueEnabled)
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
        hexfectionPieceNodes.removeAll()
        clearHexfectionSelectionState()
        aiComputationToken = UUID()
        isHexpandExploding = false
        setAIScheduling(false)
        movesMade = 0
        onTurnCountUpdate?(movesMade)
        isGameOver = false
        redHadTile = false
        blueHadTile = false
        boardNode.childNodes.forEach { $0.removeFromParentNode() }
        let coords = HexGrid.generateHexagon(radius: radius)
        boardCoords = coords
        for coord in coords {
            let tile = HexTileNode(coord: coord, size: tileSize, height: tileHeight)
            let position = HexGrid.axialToWorld(
                q: coord.q,
                r: coord.r,
                tileSize: Float(tileSize) * currentTileSpacing()
            )
            tile.position = SCNVector3(position.x, Float(tileHeight * 0.5), position.z)
            tile.setStyle(for: gameMode)
            tile.setState(.empty)
            boardNode.addChildNode(tile)
            tileNodes[coord] = tile
            tileStates[coord] = .empty
            hexpandLevels[coord] = 0
            hexpandBallNodes[coord] = []
        }
        if gameMode == .hexello {
            setupInitialRing()
        } else if gameMode == .hexfection {
            setupInitialHexfection()
        } else {
            onMessageUpdate?("")
            updateScores()
        }
        refreshHexelloMoveHighlights()
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

    private func setupInitialHexfection() {
        let corners = cornerCoordinates()
        for (index, coord) in corners.enumerated() {
            let owner: TileState = index.isMultiple(of: 2) ? .red : .blue
            tileStates[coord] = owner
            tileNodes[coord]?.setState(owner)
            updateHexfectionPiece(at: coord, owner: owner, animated: false)
        }
        onTurnUpdate?(currentPlayer)
        onActiveUpdate?(currentPlayer)
        onMessageUpdate?("Select one of your tiles")
        updateScores()
    }

    private func applyMove(at coord: AxialCoord) {
        guard !isAnimatingMove, !isGameOver else { return }
        setAIScheduling(false)
        if gameMode == .hexpand {
            applyHexpandMove(at: coord)
            return
        }
        if gameMode == .hexfection {
            applyHexfectionMove(at: coord)
            return
        }
        guard tileStates[coord] == .empty else { return }
        let flipLines = flipLinesForMove(at: coord, player: currentPlayer)
        guard !flipLines.isEmpty else { return }
        clearHexelloMoveHighlights(animated: false)
        isAnimatingMove = true
        tileStates[coord] = currentPlayer
        tileNodes[coord]?.animatePlacement(to: currentPlayer) { [weak self] in
            guard let self else { return }
            self.movesMade += 1
            self.onTurnCountUpdate?(self.movesMade)
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
        switch gameMode {
        case .hexello:
            return hexelloTileSpacing
        case .hexpand:
            return hexpandTileSpacing
        case .hexfection:
            return hexfectionTileSpacing
        }
    }

    private enum HexfectionMoveType {
        case clone
        case leap
    }

    private struct HexfectionAIMove: Equatable {
        let source: AxialCoord
        let destination: AxialCoord
        let type: HexfectionMoveType
    }

    private func applyHexfectionMove(at coord: AxialCoord) {
        guard let destinationOwner = tileStates[coord] else { return }

        if let source = hexfectionSelectedSource {
            if coord == source {
                clearHexfectionSelectionState()
                onMessageUpdate?("Select one of your tiles")
                return
            }

            if destinationOwner == currentPlayer {
                setHexfectionSelectionSource(coord)
                onMessageUpdate?("Select empty adjacent to clone or empty 2-away to leap")
                return
            }

            guard let moveType = hexfectionMoveType(for: coord) else {
                clearHexfectionSelectionState()
                onMessageUpdate?("Select one of your tiles")
                return
            }
            startHexfectionMove(from: source, to: coord, type: moveType, player: currentPlayer)
            return
        }

        guard destinationOwner == currentPlayer else {
            clearHexfectionSelectionState()
            return
        }
        setHexfectionSelectionSource(coord)
        onMessageUpdate?("Select empty adjacent to clone or empty 2-away to leap")
    }

    private func setHexfectionSelectionSource(_ source: AxialCoord) {
        guard tileStates[source] == currentPlayer else { return }
        hexfectionSelectedSource = source
        let targets = hexfectionTargets(from: source)
        hexfectionCloneTargets = targets.clone
        hexfectionLeapTargets = targets.leap
        refreshHexfectionTileHighlights()
        playRaise()
    }

    private func refreshHexelloMoveHighlights() {
        guard gameMode == .hexello else {
            clearHexelloMoveHighlights(animated: false)
            return
        }
        guard !isGameOver, !isAnimatingMove, canAcceptHumanInput else {
            clearHexelloMoveHighlights()
            return
        }

        let moves = Set(legalMoves(for: currentPlayer))
        clearHexelloMoveHighlights()
        guard !moves.isEmpty else { return }

        let baseColor = HexTileNode.hexelloTileBaseColor()
        let playerColor = HexTileNode.hexelloColor(for: currentPlayer)
        let tintColor = interpolateColor(from: baseColor, to: playerColor, t: hexelloMoveTileTintBlend)

        for coord in moves {
            guard tileStates[coord] == .empty, let tile = tileNodes[coord] else { continue }
            let borderNode = makeHexelloMoveBorderNode(for: coord, matching: tile)
            borderNode.opacity = 0
            let borderFadeIn = SCNAction.fadeIn(duration: hexelloMoveHighlightDuration)
            borderFadeIn.timingMode = .easeInEaseOut
            borderNode.runAction(borderFadeIn, forKey: "hexelloMoveHintBorderFade")
            boardNode.addChildNode(borderNode)
            tile.removeAction(forKey: "hexelloMoveHintScale")
            let scale = SCNAction.scale(to: CGFloat(hexelloMoveHighlightScale), duration: hexelloMoveHighlightDuration)
            scale.timingMode = .easeInEaseOut
            tile.runAction(scale, forKey: "hexelloMoveHintScale")
            tile.animateTintColor(to: tintColor, duration: hexelloMoveHighlightDuration)
        }
    }

    private func clearHexelloMoveHighlights(animated: Bool = true) {
        let duration = animated ? hexelloMoveHighlightDuration : 0
        let borderNodes = boardNode.childNodes.filter { node in
            guard let name = node.name else { return false }
            return name.hasPrefix(hexelloMoveHintBorderBaseName)
        }
        for borderNode in borderNodes {
            borderNode.removeAction(forKey: "hexelloMoveHintBorderFade")
            if duration > 0 {
                let fadeOut = SCNAction.fadeOut(duration: duration)
                fadeOut.timingMode = .easeInEaseOut
                let remove = SCNAction.removeFromParentNode()
                borderNode.runAction(.sequence([fadeOut, remove]), forKey: "hexelloMoveHintBorderFade")
            } else {
                borderNode.removeFromParentNode()
            }
        }

        for (coord, tile) in tileNodes {
            tile.removeAction(forKey: "hexelloMoveHintScale")
            if duration > 0 {
                let scale = SCNAction.scale(to: 1.0, duration: duration)
                scale.timingMode = .easeInEaseOut
                tile.runAction(scale, forKey: "hexelloMoveHintScale")
                if let state = tileStates[coord] {
                    tile.animateState(to: state, duration: duration)
                }
            } else {
                tile.scale = SCNVector3(1, 1, 1)
                if let state = tileStates[coord] {
                    tile.setState(state)
                }
            }
        }
    }

    private func makeHexelloMoveBorderNode(for coord: AxialCoord, matching tile: HexTileNode) -> SCNNode {
        let node = SCNNode(geometry: hexelloMoveHintBorderGeometry.copy() as? SCNGeometry ?? hexelloMoveHintBorderGeometry)
        let borderHalfHeight = Float(tileHeight * CGFloat(hexelloMoveBorderThicknessScale) * 0.5)
        let verticalOffset = Float(tileHeight * 0.5) - hexelloMoveBorderTopInset - borderHalfHeight
        node.position = SCNVector3(tile.position.x, tile.position.y + verticalOffset, tile.position.z)
        node.eulerAngles.x = -.pi / 2
        node.scale = SCNVector3(1, 1, hexelloMoveBorderThicknessScale)
        node.castsShadow = false
        node.renderingOrder = -1
        node.name = hexelloMoveHintBorderBaseName
        setNodeCoordMetadata(node, coord: coord)
        return node
    }

    private func makeHexelloMoveHintBorderGeometry() -> SCNGeometry {
        let path = UIBezierPath()
        for index in 0..<6 {
            let angle = CGFloat(index) * .pi / 3 - .pi / 2
            let point = CGPoint(x: cos(angle) * tileSize, y: sin(angle) * tileSize)
            if index == 0 {
                path.move(to: point)
            } else {
                path.addLine(to: point)
            }
        }
        path.close()

        let shape = SCNShape(path: path, extrusionDepth: tileHeight)
        shape.chamferRadius = tileHeight * 0.6

        let material = SCNMaterial()
        material.diffuse.contents = UIColor(white: 0.94, alpha: 1.0)
        material.emission.contents = UIColor.black
        material.emission.intensity = 0
        material.lightingModel = .physicallyBased
        material.metalness.contents = 0
        material.roughness.contents = 1.0
        material.specular.contents = UIColor(white: 0.08, alpha: 1.0)
        material.isDoubleSided = false
        shape.materials = [material]
        return shape
    }

    private func clearHexfectionSelectionState(resetPieceTransforms: Bool = true) {
        hexfectionSelectedSource = nil
        hexfectionCloneTargets.removeAll()
        hexfectionLeapTargets.removeAll()
        clearHexfectionTileTransforms(resetPieceTransforms: resetPieceTransforms)
    }

    private func startHexfectionMove(
        from source: AxialCoord,
        to destination: AxialCoord,
        type: HexfectionMoveType,
        player: TileState
    ) {
        guard tileStates[source] == player, tileStates[destination] == .empty else {
            clearHexfectionSelectionState()
            return
        }
        isAnimatingMove = true
        clearHexfectionSelectionState(resetPieceTransforms: false)
        executeHexfectionMove(from: source, to: destination, type: type, player: player)
    }

    private func executeHexfectionMove(
        from source: AxialCoord,
        to destination: AxialCoord,
        type: HexfectionMoveType,
        player: TileState
    ) {
        let finalizeMove: () -> Void = { [weak self] in
            guard let self else { return }
            let capturedAny = self.captureHexfectionAdjacents(around: destination, owner: player)
            self.movesMade += 1
            self.onTurnCountUpdate?(self.movesMade)
            self.updateScores()
            if capturedAny {
                DispatchQueue.main.asyncAfter(deadline: .now() + self.hexfectionCaptureAnimationDuration) { [weak self] in
                    self?.advanceTurn()
                }
            } else {
                self.advanceTurn()
            }
        }

        settleHexfectionSourceCounter(at: source, duration: 0.33) { [weak self] in
            guard let self else { return }
            switch type {
            case .clone:
                self.performHexfectionClone(from: source, to: destination, owner: player, completion: finalizeMove)
            case .leap:
                self.performHexfectionLeap(from: source, to: destination, owner: player, completion: finalizeMove)
            }
        }
    }

    private func hexfectionMoveType(for destination: AxialCoord) -> HexfectionMoveType? {
        if hexfectionCloneTargets.contains(destination) {
            return .clone
        }
        if hexfectionLeapTargets.contains(destination) {
            return .leap
        }
        return nil
    }

    private func hexfectionTargets(from source: AxialCoord) -> (clone: Set<AxialCoord>, leap: Set<AxialCoord>) {
        var cloneTargets: Set<AxialCoord> = []
        var leapTargets: Set<AxialCoord> = []

        for (coord, owner) in tileStates where owner == .empty {
            let distance = hexDistance(source, coord)
            if distance == 1 {
                cloneTargets.insert(coord)
            } else if distance == 2 {
                leapTargets.insert(coord)
            }
        }

        return (cloneTargets, leapTargets)
    }

    private func performHexfectionClone(
        from source: AxialCoord,
        to destination: AxialCoord,
        owner: TileState,
        completion: @escaping () -> Void
    ) {
        guard let sourceTile = tileNodes[source], let destinationTile = tileNodes[destination] else {
            completion()
            return
        }

        tileStates[destination] = owner
        tileNodes[destination]?.setState(owner)
        if let destinationPiece = hexfectionPieceNodes[destination] {
            destinationPiece.removeAllActions()
            destinationPiece.removeFromParentNode()
            hexfectionPieceNodes.removeValue(forKey: destination)
        }

        let sourcePiece: SCNNode
        if let existing = hexfectionPieceNodes[source] {
            sourcePiece = existing
            sourcePiece.removeAllActions()
        } else {
            sourcePiece = makeHexfectionPieceNode(owner: owner)
            sourcePiece.position = SCNVector3(sourceTile.position.x, hexfectionPieceBaseY, sourceTile.position.z)
            setNodeCoordMetadata(sourcePiece, coord: source)
            boardNode.addChildNode(sourcePiece)
            hexfectionPieceNodes[source] = sourcePiece
        }

        let traveler = makeHexfectionPieceNode(owner: owner)
        traveler.position = SCNVector3(sourceTile.position.x, hexfectionPieceBaseY, sourceTile.position.z)
        traveler.eulerAngles = sourcePiece.eulerAngles
        traveler.name = "hexfectionPieceTravel"
        boardNode.addChildNode(traveler)

        let originCenter = SCNVector3(sourceTile.position.x, hexfectionPieceBaseY, sourceTile.position.z)
        let destinationCenter = SCNVector3(destinationTile.position.x, hexfectionPieceBaseY, destinationTile.position.z)

        let dx = destinationCenter.x - originCenter.x
        let dz = destinationCenter.z - originCenter.z
        let length = max(0.0001, sqrt((dx * dx) + (dz * dz)))
        let dirX = dx / length
        let dirZ = dz / length
        let splitOffset = Float(tileSize) * 0.24
        let sourceSplit = SCNVector3(originCenter.x - (dirX * splitOffset), originCenter.y, originCenter.z - (dirZ * splitOffset))
        let travelerSplit = SCNVector3(originCenter.x + (dirX * splitOffset), originCenter.y, originCenter.z + (dirZ * splitOffset))

        let splitDuration = hexfectionMoveAnimationDuration * 0.28
        let travelDuration = hexfectionMoveAnimationDuration * 0.44
        let growDuration = hexfectionMoveAnimationDuration * 0.28

        let sourceFactor = max(0.01, sourcePiece.scale.x / hexfectionPieceDiameterScale)
        let travelerFactor = max(0.01, traveler.scale.x / hexfectionPieceDiameterScale)
        let halfFactor: Float = 0.5

        playSplit()

        let sourceSplitMove = SCNAction.move(to: sourceSplit, duration: splitDuration)
        sourceSplitMove.timingMode = .easeInEaseOut
        let travelerSplitMove = SCNAction.move(to: travelerSplit, duration: splitDuration)
        travelerSplitMove.timingMode = .easeInEaseOut
        let sourceSplitScale = squashedPieceScaleAction(from: sourceFactor, to: halfFactor, duration: splitDuration)
        let travelerSplitScale = squashedPieceScaleAction(from: travelerFactor, to: halfFactor, duration: splitDuration)

        let sourceReturnMove = SCNAction.move(to: originCenter, duration: travelDuration)
        sourceReturnMove.timingMode = .easeInEaseOut
        let travelerTravelMove = SCNAction.move(to: destinationCenter, duration: travelDuration)
        travelerTravelMove.timingMode = .easeInEaseOut

        let sourceGrow = squashedPieceScaleAction(from: halfFactor, to: 1.0, duration: growDuration)
        let travelerGrow = squashedPieceScaleAction(from: halfFactor, to: 1.0, duration: growDuration)

        sourcePiece.runAction(
            .sequence([
                .group([sourceSplitMove, sourceSplitScale]),
                sourceReturnMove,
                sourceGrow
            ]),
            forKey: "hexfectionCloneSource"
        )

        let finish = SCNAction.run { [weak self] _ in
            guard let self else { return }
            sourcePiece.position = originCenter
            self.setNodeCoordMetadata(sourcePiece, coord: source)
            traveler.position = destinationCenter
            self.setNodeCoordMetadata(traveler, coord: destination)
            Task { @MainActor in
                self.hexfectionPieceNodes[destination] = traveler
                completion()
            }
        }
        traveler.runAction(
            .sequence([
                .group([travelerSplitMove, travelerSplitScale]),
                travelerTravelMove,
                travelerGrow,
                finish
            ]),
            forKey: "hexfectionCloneTraveler"
        )
    }

    private func performHexfectionLeap(
        from source: AxialCoord,
        to destination: AxialCoord,
        owner: TileState,
        completion: @escaping () -> Void
    ) {
        guard let sourceTile = tileNodes[source], let destinationTile = tileNodes[destination] else {
            completion()
            return
        }

        tileStates[source] = .empty
        tileNodes[source]?.setState(.empty)
        tileStates[destination] = owner
        tileNodes[destination]?.setState(owner)
        updateHexfectionPiece(at: destination, owner: .empty, animated: false)

        let movingPiece: SCNNode
        if let sourcePiece = hexfectionPieceNodes[source] {
            movingPiece = sourcePiece
            movingPiece.removeAllActions()
            hexfectionPieceNodes.removeValue(forKey: source)
        } else {
            movingPiece = makeHexfectionPieceNode(owner: owner)
            movingPiece.position = SCNVector3(sourceTile.position.x, hexfectionPieceBaseY, sourceTile.position.z)
            boardNode.addChildNode(movingPiece)
        }

        let start = movingPiece.position
        let end = SCNVector3(destinationTile.position.x, hexfectionPieceBaseY, destinationTile.position.z)
        let duration = hexfectionMoveAnimationDuration
        let arcHeight = hexfectionLeapArcHeight
        let arc = SCNAction.customAction(duration: duration) { node, elapsed in
            let t = max(0, min(1, Float(elapsed / duration)))
            let eased = t * t * (3 - 2 * t)
            let x = start.x + ((end.x - start.x) * eased)
            let z = start.z + ((end.z - start.z) * eased)
            let y = start.y + ((end.y - start.y) * eased) + (arcHeight * 4 * eased * (1 - eased))
            node.position = SCNVector3(x, y, z)
        }
        let startOrientation = movingPiece.orientation
        let travelDirection = normalizedVector(SCNVector3(end.x - start.x, 0, end.z - start.z))
        let up = SCNVector3(0, 1, 0)
        let flipAxis = normalizedVector(crossProduct(travelDirection, up))
        let effectiveFlipAxis = (abs(flipAxis.x) < 0.0001 && abs(flipAxis.z) < 0.0001) ? SCNVector3(1, 0, 0) : flipAxis
        let flip = SCNAction.customAction(duration: duration) { node, elapsed in
            let t = max(0, min(1, Float(elapsed / duration)))
            let eased = t * t * (3 - 2 * t)
            let rotation = self.quaternion(axis: effectiveFlipAxis, angle: -.pi * eased)
            node.orientation = self.multiplyQuaternions(rotation, startOrientation)
        }
        let finish = SCNAction.run { [weak self] _ in
            guard let self else { return }
            movingPiece.position = end
            self.setNodeCoordMetadata(movingPiece, coord: destination)
            Task { @MainActor in
                self.hexfectionPieceNodes[destination] = movingPiece
                completion()
            }
        }
        playJump()
        movingPiece.runAction(.sequence([.group([arc, flip]), finish]), forKey: "hexfectionLeapTravel")
    }

    private func captureHexfectionAdjacents(around coord: AxialCoord, owner: TileState) -> Bool {
        let opponent = owner.opposite
        var capturedNeighbors: [AxialCoord] = []
        for direction in HexGrid.directions {
            let neighbor = coord.adding(direction)
            guard tileStates[neighbor] == opponent else { continue }
            capturedNeighbors.append(neighbor)
            tileStates[neighbor] = owner
        }
        guard !capturedNeighbors.isEmpty else { return false }

        let transitionDelays = animateHexfectionCaptureWave(from: coord, to: capturedNeighbors, owner: owner)
        for target in capturedNeighbors {
            let delay = transitionDelays[target] ?? 0
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
                guard let self else { return }
                self.tileNodes[target]?.animateState(to: owner, duration: self.hexfectionCaptureFadeDuration)
                self.animateHexfectionPieceOwnershipChange(at: target, to: owner)
            }
        }
        return true
    }

    private func refreshHexfectionTileHighlights() {
        clearHexfectionTileTransforms()
        guard let source = hexfectionSelectedSource else { return }

        applyHexfectionTileTransform(
            at: source,
            scale: hexfectionTileHighlightScale,
            lift: 0,
            pieceLift: hexfectionSourcePieceLift
        )
        let playerColor = HexTileNode.hexfectionPieceColor(for: currentPlayer)
        let baseColor = HexTileNode.hexfectionTileBaseColor()
        let cloneColor = interpolateColor(from: baseColor, to: playerColor, t: hexfectionCloneTileTintBlend)
        let leapColor = interpolateColor(from: baseColor, to: playerColor, t: hexfectionLeapTileTintBlend)
        for coord in hexfectionCloneTargets {
            applyHexfectionTileTransform(at: coord, scale: hexfectionTileHighlightScale, lift: hexfectionTileLiftAmount)
            applyHexfectionTileTint(at: coord, color: cloneColor)
        }
        for coord in hexfectionLeapTargets {
            applyHexfectionTileTransform(at: coord, scale: hexfectionTileHighlightScale, lift: hexfectionTileLiftAmount)
            applyHexfectionTileTint(at: coord, color: leapColor)
        }
    }

    private func clearHexfectionTileTransforms(resetPieceTransforms: Bool = true) {
        for coord in hexfectionTransformedTiles {
            applyHexfectionTileTransform(
                at: coord,
                scale: 1.0,
                lift: 0,
                track: false,
                adjustPiece: resetPieceTransforms
            )
        }
        hexfectionTransformedTiles.removeAll()
        for coord in hexfectionTintedTiles {
            guard let state = tileStates[coord] else { continue }
            tileNodes[coord]?.animateState(to: state, duration: hexfectionTileHighlightDuration)
        }
        hexfectionTintedTiles.removeAll()
    }

    private func applyHexfectionTileTint(at coord: AxialCoord, color: UIColor) {
        guard let tile = tileNodes[coord] else { return }
        tile.animateTintColor(to: color, duration: hexfectionTileHighlightDuration)
        hexfectionTintedTiles.insert(coord)
    }

    private func applyHexfectionTileTransform(
        at coord: AxialCoord,
        scale: Float,
        lift: Float,
        pieceLift: Float? = nil,
        track: Bool = true,
        adjustPiece: Bool = true
    ) {
        guard let tile = tileNodes[coord] else { return }
        let targetY = Float(tileHeight * 0.5) + lift
        let targetPosition = SCNVector3(tile.position.x, targetY, tile.position.z)

        tile.removeAction(forKey: "hexfectionTileTransform")
        let move = SCNAction.move(to: targetPosition, duration: hexfectionTileHighlightDuration)
        move.timingMode = .easeInEaseOut
        let scaleAction = SCNAction.scale(to: CGFloat(scale), duration: hexfectionTileHighlightDuration)
        scaleAction.timingMode = .easeInEaseOut
        tile.runAction(.group([move, scaleAction]), forKey: "hexfectionTileTransform")
        if adjustPiece, let piece = hexfectionPieceNodes[coord] {
            let resolvedPieceLift = pieceLift ?? lift
            let pieceMove = SCNAction.move(
                to: SCNVector3(piece.position.x, hexfectionPieceBaseY + resolvedPieceLift, piece.position.z),
                duration: hexfectionTileHighlightDuration
            )
            pieceMove.timingMode = .easeInEaseOut
            piece.runAction(pieceMove, forKey: "hexfectionPieceTransform")
        }

        if track {
            hexfectionTransformedTiles.insert(coord)
        } else {
            hexfectionTransformedTiles.remove(coord)
        }
    }

    private func settleHexfectionSourceCounter(
        at coord: AxialCoord,
        duration: TimeInterval,
        completion: @escaping () -> Void
    ) {
        guard let piece = hexfectionPieceNodes[coord] else {
            DispatchQueue.main.asyncAfter(deadline: .now() + duration) {
                completion()
            }
            return
        }

        piece.removeAction(forKey: "hexfectionPieceTransform")
        piece.removeAction(forKey: "hexfectionSourceSettle")
        let move = SCNAction.move(
            to: SCNVector3(piece.position.x, hexfectionPieceBaseY, piece.position.z),
            duration: duration
        )
        move.timingMode = .easeInEaseOut
        let finish = SCNAction.run { _ in }
        piece.runAction(.sequence([move, finish]), forKey: "hexfectionSourceSettle")
        DispatchQueue.main.asyncAfter(deadline: .now() + duration) {
            completion()
        }
    }

    private var hexfectionPieceBaseY: Float {
        let radius = tileSize * hexfectionPieceBaseRadiusFactor
        return Float(tileHeight) + (Float(radius) * hexfectionPieceHeightScale) + 0.02
    }

    private var hexfectionSourcePieceLift: Float {
        let pieceHeight = Float(tileSize * hexfectionPieceBaseRadiusFactor * 2) * hexfectionPieceHeightScale
        return pieceHeight * 1.5
    }

    private func hexDistance(_ lhs: AxialCoord, _ rhs: AxialCoord) -> Int {
        let dx = lhs.q - rhs.q
        let dz = lhs.r - rhs.r
        let dy = -dx - dz
        return max(abs(dx), max(abs(dy), abs(dz)))
    }

    private func makeHexfectionPieceNode(owner: TileState) -> SCNNode {
        let radius = tileSize * hexfectionPieceBaseRadiusFactor
        let sphere = SCNSphere(radius: radius)
        sphere.segmentCount = 22
        let material = SCNMaterial()
        material.lightingModel = .physicallyBased
        material.diffuse.contents = HexTileNode.hexfectionPieceColor(for: owner)
        material.metalness.contents = 0.18
        material.roughness.contents = 0.27
        material.specular.contents = UIColor(white: 1.0, alpha: 1.0)
        sphere.materials = [material]

        let piece = SCNNode(geometry: sphere)
        piece.scale = SCNVector3(hexfectionPieceDiameterScale, hexfectionPieceHeightScale, hexfectionPieceDiameterScale)
        piece.name = "hexfectionPiece"
        return piece
    }

    private func animateHexfectionPieceOwnershipChange(at coord: AxialCoord, to owner: TileState) {
        guard let piece = hexfectionPieceNodes[coord] else {
            updateHexfectionPiece(at: coord, owner: owner, animated: false)
            return
        }

        piece.removeAction(forKey: "hexfectionCapture")
        let targetColor = HexTileNode.hexfectionPieceColor(for: owner)
        let startColor = resolvedColor(from: piece.geometry?.firstMaterial?.diffuse.contents)
            ?? targetColor
        let fade = SCNAction.customAction(duration: hexfectionCaptureFadeDuration) { node, elapsed in
            guard let material = node.geometry?.firstMaterial else { return }
            let t = max(0, min(1, CGFloat(elapsed / self.hexfectionCaptureFadeDuration)))
            let eased = t * t * (3 - 2 * t)
            material.diffuse.contents = self.interpolateColor(from: startColor, to: targetColor, t: eased)
        }
        let finish = SCNAction.run { [weak piece] _ in
            piece?.geometry?.firstMaterial?.diffuse.contents = targetColor
        }
        piece.runAction(.sequence([fade, finish]), forKey: "hexfectionCapture")
    }

    private func animateHexfectionCaptureWave(from source: AxialCoord, to targets: [AxialCoord], owner: TileState) -> [AxialCoord: TimeInterval] {
        guard !targets.isEmpty else { return [:] }
        guard let sourceTile = tileNodes[source] else { return [:] }

        let sourcePosition: SCNVector3
        if let sourcePiece = hexfectionPieceNodes[source] {
            sourcePosition = sourcePiece.presentation.position
        } else {
            sourcePosition = SCNVector3(sourceTile.position.x, hexfectionPieceBaseY, sourceTile.position.z)
        }

        var targetDistances: [AxialCoord: Float] = [:]
        var maxDistance: Float = 0.0001
        for target in targets {
            guard let targetTile = tileNodes[target] else { continue }
            let targetPosition = SCNVector3(targetTile.position.x, hexfectionPieceBaseY, targetTile.position.z)
            let dx = targetPosition.x - sourcePosition.x
            let dz = targetPosition.z - sourcePosition.z
            let distance = sqrt((dx * dx) + (dz * dz))
            targetDistances[target] = distance
            maxDistance = max(maxDistance, distance)
        }

        let sourceColor = resolvedColor(from: hexfectionPieceNodes[source]?.geometry?.firstMaterial?.diffuse.contents)
            ?? HexTileNode.hexfectionPieceColor(for: owner)
        let outerRadius = max(0.04, tileSize * hexfectionCaptureWaveOuterRadiusFactor)
        let waveDiameter = outerRadius * 2
        let ringContainer = SCNNode()
        ringContainer.name = "hexfectionCaptureWaveContainer"
        ringContainer.position = SCNVector3(sourcePosition.x, sourcePosition.y, sourcePosition.z)
        ringContainer.scale = SCNVector3(0.22, 1.0, 0.22)
        ringContainer.opacity = 0

        let layerCount = max(1, hexfectionCaptureWaveLayerCount)
        let thickness = max(0.01, hexfectionCaptureWaveThickness)
        let layerSpacing = thickness / CGFloat(layerCount)
        let layerStart = -thickness * 0.5
        let waveTexture = makeHexfectionCaptureWaveTexture(color: sourceColor)
        for index in 0..<layerCount {
            let ring = SCNPlane(width: waveDiameter, height: waveDiameter)
            let material = SCNMaterial()
            material.lightingModel = .constant
            material.diffuse.contents = waveTexture
            material.emission.contents = waveTexture
            material.isDoubleSided = true
            material.blendMode = .add
            material.readsFromDepthBuffer = false
            material.writesToDepthBuffer = false
            ring.materials = [material]

            let layerNode = SCNNode(geometry: ring)
            layerNode.name = "hexfectionCaptureWaveLayer"
            layerNode.eulerAngles.x = -.pi / 2
            let yOffset = layerStart + (layerSpacing * (CGFloat(index) + 0.5))
            layerNode.position = SCNVector3(0, Float(yOffset), 0)
            ringContainer.addChildNode(layerNode)
        }
        boardNode.addChildNode(ringContainer)

        let endScale = max(1.35, ((CGFloat(maxDistance) / outerRadius) + 0.75) * 1.38)
        let waveExpand = SCNAction.customAction(duration: hexfectionCaptureWaveDuration) { node, elapsed in
            let t = max(0, min(1, CGFloat(elapsed / self.hexfectionCaptureWaveDuration)))
            let eased = t * t * (3 - 2 * t)
            let scale = 0.22 + ((endScale - 0.22) * eased)
            node.scale = SCNVector3(Float(scale), 1.0, Float(scale))
            let fade = max(0.0, 1.0 - (eased * eased))
            node.opacity = CGFloat(0.95 * fade)
        }
        let playTakeover = SCNAction.run { [weak self] _ in
            self?.playTakeover()
        }
        ringContainer.runAction(
            .sequence([.wait(duration: hexfectionCaptureWaveLeadDelay), playTakeover, waveExpand, .removeFromParentNode()]),
            forKey: "hexfectionCaptureWave"
        )

        var delays: [AxialCoord: TimeInterval] = [:]
        let transitionWindowStart = max(0.0, hexfectionCaptureWaveDuration - 0.6)
        let transitionWindowRange = max(0.0, 0.6 - hexfectionCaptureFadeDuration)
        for (target, distance) in targetDistances {
            let progress = maxDistance > 0.0001 ? (distance / maxDistance) : 0.5
            delays[target] = hexfectionCaptureWaveLeadDelay + transitionWindowStart + (transitionWindowRange * TimeInterval(progress))
        }
        return delays
    }

    private func makeHexfectionCaptureWaveTexture(color: UIColor) -> UIImage {
        let size: CGFloat = 256
        let innerFraction = max(0.2, min(0.85, hexfectionCaptureWaveInnerRadiusFactor / hexfectionCaptureWaveOuterRadiusFactor))
        let innerClearEnd = innerFraction * 0.9
        let rampMid = min(0.9, innerFraction + ((1.0 - innerFraction) * 0.55))
        let brightEdge = min(0.98, rampMid + 0.2)
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: size, height: size))
        return renderer.image { context in
            let cg = context.cgContext
            let center = CGPoint(x: size * 0.5, y: size * 0.5)
            let radius = size * 0.5
            let colors = [
                color.withAlphaComponent(0.0).cgColor,
                color.withAlphaComponent(0.0).cgColor,
                color.withAlphaComponent(0.42).cgColor,
                color.withAlphaComponent(1.0).cgColor,
                color.withAlphaComponent(0.0).cgColor
            ] as CFArray
            let locations: [CGFloat] = [0.0, innerClearEnd, rampMid, brightEdge, 1.0]
            guard let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: colors, locations: locations) else {
                return
            }
            cg.drawRadialGradient(
                gradient,
                startCenter: center,
                startRadius: 0,
                endCenter: center,
                endRadius: radius,
                options: [.drawsAfterEndLocation]
            )
        }
    }

    private func squashedPieceScaleAction(from start: Float, to end: Float, duration: TimeInterval) -> SCNAction {
        let clampedStart = max(0.01, start)
        let clampedEnd = max(0.01, end)
        return SCNAction.customAction(duration: duration) { node, elapsed in
            let t = max(0, min(1, Float(elapsed / duration)))
            let eased = t * t * (3 - 2 * t)
            let factor = clampedStart + ((clampedEnd - clampedStart) * eased)
            node.scale = SCNVector3(
                self.hexfectionPieceDiameterScale * factor,
                self.hexfectionPieceHeightScale * factor,
                self.hexfectionPieceDiameterScale * factor
            )
        }
    }

    private func resolvedColor(from contents: Any?) -> UIColor? {
        if let color = contents as? UIColor {
            return color
        }
        return nil
    }

    private func crossProduct(_ lhs: SCNVector3, _ rhs: SCNVector3) -> SCNVector3 {
        SCNVector3(
            (lhs.y * rhs.z) - (lhs.z * rhs.y),
            (lhs.z * rhs.x) - (lhs.x * rhs.z),
            (lhs.x * rhs.y) - (lhs.y * rhs.x)
        )
    }

    private func normalizedVector(_ vector: SCNVector3) -> SCNVector3 {
        let length = sqrt((vector.x * vector.x) + (vector.y * vector.y) + (vector.z * vector.z))
        guard length > 0.0001 else { return SCNVector3Zero }
        return SCNVector3(vector.x / length, vector.y / length, vector.z / length)
    }

    private func quaternion(axis: SCNVector3, angle: Float) -> SCNQuaternion {
        let normalizedAxis = normalizedVector(axis)
        let half = angle * 0.5
        let s = sin(half)
        return SCNQuaternion(
            normalizedAxis.x * s,
            normalizedAxis.y * s,
            normalizedAxis.z * s,
            cos(half)
        )
    }

    private func multiplyQuaternions(_ lhs: SCNQuaternion, _ rhs: SCNQuaternion) -> SCNQuaternion {
        SCNQuaternion(
            (lhs.w * rhs.x) + (lhs.x * rhs.w) + (lhs.y * rhs.z) - (lhs.z * rhs.y),
            (lhs.w * rhs.y) - (lhs.x * rhs.z) + (lhs.y * rhs.w) + (lhs.z * rhs.x),
            (lhs.w * rhs.z) + (lhs.x * rhs.y) - (lhs.y * rhs.x) + (lhs.z * rhs.w),
            (lhs.w * rhs.w) - (lhs.x * rhs.x) - (lhs.y * rhs.y) - (lhs.z * rhs.z)
        )
    }

    private func interpolateColor(from: UIColor, to: UIColor, t: CGFloat) -> UIColor {
        var fr: CGFloat = 0
        var fg: CGFloat = 0
        var fb: CGFloat = 0
        var fa: CGFloat = 0
        var tr: CGFloat = 0
        var tg: CGFloat = 0
        var tb: CGFloat = 0
        var ta: CGFloat = 0
        from.getRed(&fr, green: &fg, blue: &fb, alpha: &fa)
        to.getRed(&tr, green: &tg, blue: &tb, alpha: &ta)
        return UIColor(
            red: fr + (tr - fr) * t,
            green: fg + (tg - fg) * t,
            blue: fb + (tb - fb) * t,
            alpha: fa + (ta - fa) * t
        )
    }

    private func updateHexfectionPiece(at coord: AxialCoord, owner: TileState, animated: Bool) {
        if let existing = hexfectionPieceNodes[coord] {
            existing.removeAllActions()
            existing.removeFromParentNode()
            hexfectionPieceNodes.removeValue(forKey: coord)
        }
        guard owner != .empty else { return }
        guard let tile = tileNodes[coord] else { return }

        let piece = makeHexfectionPieceNode(owner: owner)
        piece.position = SCNVector3(
            tile.position.x,
            hexfectionPieceBaseY,
            tile.position.z
        )
        setNodeCoordMetadata(piece, coord: coord)
        boardNode.addChildNode(piece)
        hexfectionPieceNodes[coord] = piece

        guard animated else { return }
        piece.opacity = 0
        let fade = SCNAction.fadeIn(duration: 0.16)
        fade.timingMode = .easeOut
        piece.runAction(fade)
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
        onTurnCountUpdate?(movesMade)
        if checkForHexpandWinIfNeeded() {
            isAnimatingMove = false
            return
        }
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
        let levelBefore = hexpandLevels[coord] ?? 0
        let burstCount = max(0, levelBefore / hexpandMaxLevel)
        let remainder = max(0, levelBefore % hexpandMaxLevel)
        guard burstCount > 0, explosionOwner != .empty else {
            isHexpandExploding = false
            processNextHexpandExplosionIfNeeded()
            return
        }
        spawnHexpandBurst(from: coord, owner: explosionOwner, duration: burstDuration)
        playPop()

        let neighbors = HexGrid.directions.compactMap { direction -> AxialCoord? in
            let neighbor = coord.adding(direction)
            return tileStates[neighbor] == nil ? nil : neighbor
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + burstDuration) { [weak self] in
            guard let self else { return }
            if remainder > 0 {
                self.tileStates[coord] = explosionOwner
                self.hexpandLevels[coord] = remainder
                self.updateHexpandTile(coord: coord, level: remainder, owner: explosionOwner, duration: 0.4)
            } else {
                self.tileStates[coord] = .empty
                self.hexpandLevels[coord] = 0
                self.updateHexpandTile(coord: coord, level: 0, owner: .empty, duration: 0.4)
            }

            for neighbor in neighbors {
                let level = (self.hexpandLevels[neighbor] ?? 0) + burstCount
                self.tileStates[neighbor] = explosionOwner
                self.hexpandLevels[neighbor] = level
                self.updateHexpandTile(coord: neighbor, level: level, owner: explosionOwner, duration: 0.3)

                if level >= self.hexpandMaxLevel {
                    self.enqueueHexpandExplosion(neighbor)
                }
            }

            self.updateScores()
            self.playPlace()
            self.isHexpandExploding = false
            if !self.hasPendingHexpandExplosions {
                self.finishHexpandMove()
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
        let isPrimed = clampedCount == hexpandPrimedLevel
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
            setNodeCoordMetadata(ball, coord: coord)
            boardNode.addChildNode(ball)
            balls.append(ball)

            let grow = SCNAction.scale(to: 1.0, duration: duration)
            grow.timingMode = .easeOut
            ball.runAction(grow)
            applyHexpandPrimedPulseIfNeeded(
                on: ball,
                isPrimed: isPrimed,
                after: duration,
                center: center,
                baseOffset: offset,
                baseY: baseY
            )
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
        material.emission.intensity = 0.0
        material.metalness.contents = 0.35
        material.roughness.contents = 0.35
        let color = HexTileNode.hexpandColor(for: owner)
        material.diffuse.contents = HexTileNode.darker(by: 0.18, color: color)
        return material
    }

    private func hexpandBallRadius() -> CGFloat {
        tileSize * 0.25
    }

    private func applyHexpandPrimedPulseIfNeeded(
        on ball: SCNNode,
        isPrimed: Bool,
        after appearDuration: TimeInterval,
        center: SCNVector3,
        baseOffset: SCNVector3,
        baseY: Float
    ) {
        ball.removeAction(forKey: "hexpandPrimedPulse")
        guard ball.geometry?.firstMaterial != nil else { return }
        guard isPrimed else { return }

        let wait = SCNAction.wait(duration: max(0, appearDuration))
        let pulse = makeHexpandPrimedPulseAction(center: center, baseOffset: baseOffset, baseY: baseY)
        ball.runAction(.sequence([wait, pulse]), forKey: "hexpandPrimedPulse")
    }

    private func makeHexpandPrimedPulseAction(
        center: SCNVector3,
        baseOffset: SCNVector3,
        baseY: Float
    ) -> SCNAction {
        SCNAction.repeatForever(
            SCNAction.customAction(duration: hexpandPrimedPulsePeriod) { [weak self] node, elapsed in
                guard let self else { return }
                let phase = CGFloat(elapsed / CGFloat(self.hexpandPrimedPulsePeriod))
                let angle = phase * 2 * .pi

                // Scale breathes between 0.8 and 1.0.
                let scaleWave = 0.5 + (0.5 * cos(angle))
                let scale = self.hexpandPrimedMinScale + ((self.hexpandPrimedMaxScale - self.hexpandPrimedMinScale) * scaleWave)
                node.scale = SCNVector3(Float(scale), Float(scale), Float(scale))

                // Ball spacing breathes by +/-10% around base offset.
                let spacingFactor = 1.0 + (self.hexpandPrimedSpacingDelta * sin(angle))
                node.position = SCNVector3(
                    center.x + (baseOffset.x * Float(spacingFactor)),
                    baseY,
                    center.z + (baseOffset.z * Float(spacingFactor))
                )
            }
        )
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
        let (redTiles, blueTiles) = tileOwnershipTotals()
        let (redScore, blueScore): (Int, Int)
        if gameMode == .hexpand {
            (redScore, blueScore) = hexpandBallTotals()
        } else {
            (redScore, blueScore) = (redTiles, blueTiles)
        }

        if redTiles > 0 {
            redHadTile = true
        }
        if blueTiles > 0 {
            blueHadTile = true
        }
        onScoreUpdate?(redScore, blueScore)
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
        refreshHexelloMoveHighlights()
    }

    func setBoardSize(_ sideLength: Int) {
        self.sideLength = max(2, sideLength)
    }

    func setGameMode(_ mode: GameMode) {
        if gameMode != mode {
            clearHexfectionSelectionState()
            if gameMode == .hexello {
                clearHexelloMoveHighlights(animated: false)
            }
        }
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

    func prepareForMenuReturn() {
        aiComputationToken = UUID()
        setAIScheduling(false)
        startAIOnIntroComplete = false
        introCompleted = false
        isAnimatingMove = false
        isHexpandExploding = false
        angularVelocityYaw = 0
        angularVelocityPitch = 0
        displayLink?.invalidate()
        displayLink = nil
        cameraNode.removeAction(forKey: "initialZoom")
        cameraNode.removeAction(forKey: "gameOverZoom")
        tiltNode.removeAllActions()
        boardNode.removeAction(forKey: "gameOverSpin")
        boardNode.removeAllActions()
        for node in boardNode.childNodes {
            node.removeAllActions()
        }
    }

    func startNewGame() {
        prepareForMenuReturn()
        currentPlayer = Bool.random() ? .red : .blue
        onTurnUpdate?(currentPlayer)
        onMessageUpdate?("")
        onAIThinkingUpdate?(.empty)
        startAIOnIntroComplete = true
        introCompleted = false
        buildBoard()
        animateInitialZoomIn()
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.onTurnUpdate?(self.currentPlayer)
            self.onActiveUpdate?(self.currentPlayer)
            self.onMessageUpdate?(self.gameMode == .hexfection ? "Select one of your tiles" : "")
            self.updateScores()
            self.refreshHexelloMoveHighlights()
        }
    }

    func currentPlayerState() -> TileState {
        currentPlayer
    }

    private func advanceTurn() {
        if gameMode == .hexfection {
            clearHexfectionSelectionState()
            if checkForHexfectionEliminationWinIfNeeded() {
                refreshHexelloMoveHighlights()
                return
            }
        }
        currentPlayer = currentPlayer.opposite
        onTurnUpdate?(currentPlayer)
        onActiveUpdate?(currentPlayer)
        if gameMode == .hexello {
            checkForNoMoves()
        } else if gameMode == .hexfection {
            checkForNoHexfectionMoves()
        } else {
            onMessageUpdate?("")
        }
        isAnimatingMove = false
        refreshHexelloMoveHighlights()
        performAITurnIfNeeded()
    }

    private func checkForHexfectionEliminationWinIfNeeded() -> Bool {
        guard gameMode == .hexfection else { return false }
        guard !isGameOver else { return false }

        let (redCount, blueCount) = tileOwnershipTotals()
        if redCount == 0, blueCount > 0 {
            return finalizeHexfectionElimination(eliminated: .red, winner: .blue)
        }
        if blueCount == 0, redCount > 0 {
            return finalizeHexfectionElimination(eliminated: .blue, winner: .red)
        }
        return false
    }

    private func finalizeHexfectionElimination(eliminated: TileState, winner: TileState) -> Bool {
        let eliminatedMessage = eliminated == .red ? "Red has been eliminated" : "Blue has been eliminated"
        let winnerMessage = winner == .red ? "Red wins" : "Blue wins"
        onMessageUpdate?(eliminatedMessage)
        onGameOver?(eliminatedMessage, winnerMessage)
        isGameOver = true
        isAnimatingMove = false
        setAIScheduling(false)
        clearHexfectionSelectionState()
        return true
    }

    private func checkForNoMoves() {
        if !hasLegalMove(for: currentPlayer) {
            let blockedPlayer = currentPlayer
            onMessageUpdate?(blockedPlayer == .red ? "No Red Move Possible" : "No Blue Move Possible")
            currentPlayer = currentPlayer.opposite
            onTurnUpdate?(currentPlayer)
            onActiveUpdate?(currentPlayer)
            if !hasLegalMove(for: currentPlayer) {
                let hasEmptyTiles = tileStates.values.contains(.empty)
                let (message, winner) = gameOverMessage()
                if hasEmptyTiles {
                    onMessageUpdate?("No Moves Possible")
                    onGameOver?("No Moves Possible", winner)
                } else {
                    onMessageUpdate?(message)
                    onGameOver?(message, winner)
                }
                isGameOver = true
                clearHexelloMoveHighlights()
            }
        } else {
            onMessageUpdate?("")
        }
    }

    private func checkForNoHexfectionMoves() {
        if !hasLegalHexfectionMove(for: currentPlayer) {
            let blockedPlayer = currentPlayer
            onMessageUpdate?(blockedPlayer == .red ? "No Red Move Possible" : "No Blue Move Possible")
            currentPlayer = currentPlayer.opposite
            onTurnUpdate?(currentPlayer)
            onActiveUpdate?(currentPlayer)
            if !hasLegalHexfectionMove(for: currentPlayer) {
                let hasEmptyTiles = tileStates.values.contains(.empty)
                let (message, winner) = gameOverMessage()
                if hasEmptyTiles {
                    onMessageUpdate?("No Moves Possible")
                    onGameOver?("No Moves Possible", winner)
                } else {
                    onMessageUpdate?(message)
                    onGameOver?(message, winner)
                }
                isGameOver = true
                clearHexfectionSelectionState()
                setAIScheduling(false)
            }
        } else {
            onMessageUpdate?("Select one of your tiles")
        }
    }

    private func checkForHexpandWinIfNeeded() -> Bool {
        guard gameMode == .hexpand else { return false }
        guard !isGameOver else { return false }

        let (redCount, blueCount) = tileOwnershipTotals()
        if redCount == 0, blueCount == 0, redHadTile, blueHadTile {
            return finalizeHexpandGame()
        }
        if redCount == 0, blueCount > 0, redHadTile {
            return finalizeHexpandGame()
        }
        if blueCount == 0, redCount > 0, blueHadTile {
            return finalizeHexpandGame()
        }
        return false
    }

    private func finalizeHexpandGame() -> Bool {
        let (message, winner) = gameOverMessage()
        onMessageUpdate?(message)
        onGameOver?(message, winner)
        isGameOver = true
        setAIScheduling(false)
        return true
    }

    private func gameOverMessage() -> (String, String) {
        if gameMode == .hexpand {
            let (redCount, blueCount) = tileOwnershipTotals()
            if redCount == 0, blueCount > 0 { return ("Blue eliminated Red", "Blue wins") }
            if blueCount == 0, redCount > 0 { return ("Red eliminated Blue", "Red wins") }
            return ("Game over", "Draw")
        }

        let (redCount, blueCount) = tileOwnershipTotals()
        if redCount == blueCount { return ("Game over", "Draw") }
        let winner = redCount > blueCount ? "Red wins" : "Blue wins"
        return ("Game over: \(redCount)–\(blueCount)", winner)
    }

    private func tileOwnershipTotals() -> (red: Int, blue: Int) {
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
        return (redCount, blueCount)
    }

    private func hexpandBallTotals() -> (red: Int, blue: Int) {
        var redTotal = 0
        var blueTotal = 0
        for (coord, owner) in tileStates {
            let level = hexpandLevels[coord] ?? 0
            switch owner {
            case .red:
                redTotal += level
            case .blue:
                blueTotal += level
            case .empty:
                break
            }
        }
        return (redTotal, blueTotal)
    }

    private func hasLegalMove(for player: TileState) -> Bool {
        !legalMoves(for: player).isEmpty
    }

    private func hasLegalHexfectionMove(for player: TileState) -> Bool {
        hasLegalHexfectionMove(for: player, in: tileStates)
    }

    private func hasLegalHexfectionMove(for player: TileState, in state: [AxialCoord: TileState]) -> Bool {
        !hexfectionLegalMoves(for: player, in: state).isEmpty
    }

    private func performAITurnIfNeeded() {
        let aiEnabled = (currentPlayer == .red && aiRedEnabled) || (currentPlayer == .blue && aiBlueEnabled)
        guard aiEnabled, !isAnimatingMove, introCompleted, !aiMoveScheduled, !isGameOver else { return }

        if gameMode == .hexello {
            scheduleHexelloAIMove()
            return
        }

        if gameMode == .hexfection {
            scheduleHexfectionAIMove()
            return
        }

        scheduleHexpandAIMove()
    }

    private func hexfectionLegalMoves(for player: TileState, in state: [AxialCoord: TileState]) -> [HexfectionAIMove] {
        let coords = boardCoords.isEmpty ? Array(state.keys) : boardCoords
        var moves: [HexfectionAIMove] = []
        for source in coords {
            guard state[source] == player else { continue }
            for destination in coords {
                guard destination != source else { continue }
                guard let owner = state[destination], owner == .empty else { continue }
                let distance = hexDistance(source, destination)
                if distance == 1 {
                    moves.append(HexfectionAIMove(source: source, destination: destination, type: .clone))
                } else if distance == 2 {
                    moves.append(HexfectionAIMove(source: source, destination: destination, type: .leap))
                }
            }
        }
        return moves
    }

    private func simulateHexfectionMove(
        _ move: HexfectionAIMove,
        player: TileState,
        in state: [AxialCoord: TileState]
    ) -> [AxialCoord: TileState] {
        var next = state
        switch move.type {
        case .clone:
            next[move.destination] = player
        case .leap:
            next[move.source] = .empty
            next[move.destination] = player
        }
        let opponent = player.opposite
        for direction in HexGrid.directions {
            let neighbor = move.destination.adding(direction)
            guard next[neighbor] == opponent else { continue }
            next[neighbor] = player
        }
        return next
    }

    private func evaluateHexfectionState(_ state: [AxialCoord: TileState], for player: TileState) -> Int {
        var playerCount = 0
        var opponentCount = 0
        for owner in state.values {
            if owner == player {
                playerCount += 1
            } else if owner == player.opposite {
                opponentCount += 1
            }
        }
        let mobility = hexfectionLegalMoves(for: player, in: state).count
        let opponentMobility = hexfectionLegalMoves(for: player.opposite, in: state).count
        return ((playerCount - opponentCount) * 12) + ((mobility - opponentMobility) * 3)
    }

    private func bestHexfectionMove(
        for player: TileState,
        in state: [AxialCoord: TileState],
        difficulty: AIDifficulty? = nil
    ) -> HexfectionAIMove? {
        let effectiveDifficulty = difficulty ?? aiDifficulty
        let legalMoves = hexfectionLegalMoves(for: player, in: state)
        guard !legalMoves.isEmpty else { return nil }

        let engine = HexfectionSearchEngine(boardCoords: boardCoords, radius: radius)
        let bestMove = engine.chooseMove(
            owners: state,
            currentPlayer: player,
            timeBudget: hexfectionHardSearchBudget()
        )?.bestMove.flatMap(hexfectionAIMove(from:))
            ?? bestHexfectionFallbackMove(for: player, in: state)

        guard let bestMove else { return nil }

        let randomOverrideDenominator: Int?
        switch effectiveDifficulty {
        case .easy:
            randomOverrideDenominator = 3
        case .medium:
            randomOverrideDenominator = 5
        case .hard:
            randomOverrideDenominator = nil
        }

        guard let denominator = randomOverrideDenominator else {
            return bestMove
        }
        guard Int.random(in: 1...denominator) == 1 else {
            return bestMove
        }

        let alternatives = legalMoves.filter { $0 != bestMove }
        return alternatives.randomElement() ?? bestMove
    }

    private func bestHexfectionFallbackMove(
        for player: TileState,
        in state: [AxialCoord: TileState]
    ) -> HexfectionAIMove? {
        let moves = hexfectionLegalMoves(for: player, in: state)
        guard !moves.isEmpty else { return nil }
        let depth = 2
        var bestMove = moves[0]
        var bestScore = Int.min
        var alpha = Int.min
        let beta = Int.max
        for move in moves {
            let next = simulateHexfectionMove(move, player: player, in: state)
            var branchAlpha = alpha
            var branchBeta = beta
            let score = minimaxHexfection(
                state: next,
                player: player.opposite,
                maximizingFor: player,
                depth: depth - 1,
                alpha: &branchAlpha,
                beta: &branchBeta
            )
            if score > bestScore {
                bestScore = score
                bestMove = move
            }
            alpha = max(alpha, bestScore)
        }
        return bestMove
    }

    private func hexfectionAIMove(from move: HexfectionSearchMove) -> HexfectionAIMove? {
        let type: HexfectionMoveType
        switch move.type {
        case .clone:
            type = .clone
        case .leap:
            type = .leap
        }
        return HexfectionAIMove(source: move.source, destination: move.destination, type: type)
    }

    private func minimaxHexfection(
        state: [AxialCoord: TileState],
        player: TileState,
        maximizingFor: TileState,
        depth: Int,
        alpha: inout Int,
        beta: inout Int
    ) -> Int {
        let moves = hexfectionLegalMoves(for: player, in: state)
        let opponentHasMove = hasLegalHexfectionMove(for: player.opposite, in: state)
        if moves.isEmpty, !opponentHasMove {
            return terminalHexfectionScore(state: state, for: maximizingFor, depth: depth)
        }
        if depth == 0 {
            return evaluateHexfectionState(state, for: maximizingFor)
        }
        if moves.isEmpty {
            return minimaxHexfection(
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
                let next = simulateHexfectionMove(move, player: player, in: state)
                var childAlpha = localAlpha
                var childBeta = localBeta
                let score = minimaxHexfection(
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
            let next = simulateHexfectionMove(move, player: player, in: state)
            var childAlpha = localAlpha
            var childBeta = localBeta
            let score = minimaxHexfection(
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

    private func terminalHexfectionScore(state: [AxialCoord: TileState], for player: TileState, depth: Int) -> Int {
        let value = evaluateHexfectionState(state, for: player)
        if value == 0 { return 0 }
        return value > 0 ? (10_000 + depth) : (-10_000 - depth)
    }

    private func bestMove(
        for player: TileState,
        in state: [AxialCoord: TileState]? = nil,
        difficulty: AIDifficulty? = nil,
        sideLength: Int? = nil
    ) -> AxialCoord? {
        let state = state ?? tileStates
        let effectiveDifficulty = difficulty ?? aiDifficulty
        let effectiveSideLength = sideLength ?? self.sideLength
        let effectiveRadius = max(1, effectiveSideLength - 1)
        let moves = legalMoves(for: player, in: state)
        guard !moves.isEmpty else { return nil }

        let depth = hexelloSearchDepth(in: state, sideLength: effectiveSideLength)
        var bestMoves: [AxialCoord] = []
        var bestScore = Int.min
        var alpha = Int.min
        let beta = Int.max

        for move in orderedHexelloMoves(for: player, in: state, radius: effectiveRadius) {
            let next = simulateHexelloMove(at: move, player: player, in: state)
            var branchAlpha = alpha
            var branchBeta = beta
            let score = minimaxHexello(
                state: next,
                player: player.opposite,
                maximizingFor: player,
                depth: depth - 1,
                radius: effectiveRadius,
                sideLength: effectiveSideLength,
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
        guard let bestMove = bestMoves.randomElement() else { return nil }

        // Easy/medium keep hard evaluation, but occasionally replace with a random legal alternative.
        let randomOverrideDenominator: Int?
        switch effectiveDifficulty {
        case .easy:
            randomOverrideDenominator = 3
        case .medium:
            randomOverrideDenominator = 5
        case .hard:
            randomOverrideDenominator = nil
        }

        guard let denominator = randomOverrideDenominator else {
            return bestMove
        }
        guard Int.random(in: 1...denominator) == 1 else {
            return bestMove
        }

        let alternatives = moves.filter { $0 != bestMove }
        return alternatives.randomElement() ?? bestMove
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
        radius: Int,
        sideLength: Int,
        alpha: inout Int,
        beta: inout Int
    ) -> Int {
        let moves = orderedHexelloMoves(for: player, in: state, radius: radius)
        let opponentHasMove = !legalMoves(for: player.opposite, in: state).isEmpty
        if moves.isEmpty, !opponentHasMove {
            return terminalHexelloScore(state: state, for: maximizingFor, depth: depth)
        }
        if depth == 0 {
            return evaluateHexello(state: state, for: maximizingFor, radius: radius)
        }
        if moves.isEmpty {
            return minimaxHexello(
                state: state,
                player: player.opposite,
                maximizingFor: maximizingFor,
                depth: depth - 1,
                radius: radius,
                sideLength: sideLength,
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
                    radius: radius,
                    sideLength: sideLength,
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
                radius: radius,
                sideLength: sideLength,
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

    private func evaluateHexello(state: [AxialCoord: TileState], for player: TileState, radius: Int) -> Int {
        let opponent = player.opposite
        let corners = cornerCoordinates(radius: radius)
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
                } else if isEdge(coord, radius: radius) {
                    playerEdges += 1
                }
            } else if owner == opponent {
                opponentCount += 1
                if corners.contains(coord) {
                    opponentCorners += 1
                } else if isEdge(coord, radius: radius) {
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

    private func orderedHexelloMoves(
        for player: TileState,
        in state: [AxialCoord: TileState],
        radius: Int
    ) -> [AxialCoord] {
        legalMoves(for: player, in: state).sorted { lhs, rhs in
            let lhsScore = hexelloMoveOrderingScore(lhs, player: player, in: state, radius: radius)
            let rhsScore = hexelloMoveOrderingScore(rhs, player: player, in: state, radius: radius)
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
        in state: [AxialCoord: TileState],
        radius: Int
    ) -> Int {
        var score = flipsForMove(at: coord, player: player, in: state).count
        if isCorner(coord, radius: radius) {
            score += 1000
        } else if isEdge(coord, radius: radius) {
            score += 80
        }
        if isAdjacentToEmptyCorner(coord, in: state, radius: radius) {
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

    private func hexelloSearchDepth(in state: [AxialCoord: TileState], sideLength: Int) -> Int {
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
        cornerCoordinates(radius: radius)
    }

    private func cornerCoordinates(radius: Int) -> [AxialCoord] {
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

    private func isAdjacentToEmptyCorner(_ coord: AxialCoord, in state: [AxialCoord: TileState], radius: Int) -> Bool {
        for corner in cornerCoordinates(radius: radius) where state[corner] == .empty {
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
        var redHadTile: Bool
        var blueHadTile: Bool
    }

    private struct HexpandSummary {
        var aiScore = 0
        var humanScore = 0
        var aiLevel5 = 0
        var humanLevel5 = 0
        var aiPieces = 0
        var humanPieces = 0
    }

    private struct HexpandPolicySummary {
        var score = 0
        var pieces = 0
        var critical = 0
        var edgeLoad = 0
        var centerControl = 0
    }

    private struct HexpandTacticalSummary {
        var nearCritical = 0
        var critical = 0
        var criticalChains = 0
        var exposedCriticalPairs = 0
    }

    private struct HexpandStrategicSummary {
        var burstPotential = 0
        var supportedCriticals = 0
        var vulnerableCriticals = 0
        var frontierPressure = 0
        var unsupportedFrontier = 0
        var edgeTrapLoad = 0
        var cornerTrapLoad = 0
    }

    private struct HexpandEvolvedWeights {
        let bias: Double
        let scoreDiff: Double
        let pieceDiff: Double
        let criticalDiff: Double
        let edgeDiff: Double
        let centerDiff: Double
        let swing: Double
        let immediateWin: Double
        let opponentImmediateWin: Double
    }

    private struct HexpandTacticalWeights {
        let nearCriticalGain: Double
        let criticalGain: Double
        let chainGain: Double
        let exposedCriticalPenalty: Double
        let localPressurePenalty: Double
        let safeBurstBonus: Double
        let riskyBurstPenalty: Double
    }

    private struct HexpandStrategicWeights {
        let bias: Double
        let scoreDiff: Double
        let pieceDiff: Double
        let criticalDiff: Double
        let centerDiff: Double
        let edgeDiff: Double
        let burstPotentialDiff: Double
        let supportedCriticalDiff: Double
        let vulnerableCriticalPenalty: Double
        let frontierPressurePenalty: Double
        let unsupportedFrontierPenalty: Double
        let edgeTrapPenalty: Double
        let cornerTrapPenalty: Double
        let chainDiff: Double
        let exposedCriticalPenalty: Double
        let immediateWin: Double
        let opponentImmediateWin: Double
    }

    private struct HexpandTacticalCacheKey: Hashable {
        let boardHash: Int
        let playerValue: Int
        let depth: Int
    }

    private struct HexpandTacticalCacheEntry {
        let depth: Int
        let score: Double
    }

    private let largeHardHexpandWeights = HexpandEvolvedWeights(
        bias: 0.383594,
        scoreDiff: 1.306195,
        pieceDiff: 4.303645,
        criticalDiff: 4.725590,
        edgeDiff: -2.498602,
        centerDiff: 0.425609,
        swing: 1.614324,
        immediateWin: 11935.626926,
        opponentImmediateWin: 2558.127751
    )

    private let hardHexpandTacticalWeights = HexpandTacticalWeights(
        nearCriticalGain: 2.2,
        criticalGain: 3.8,
        chainGain: 5.6,
        exposedCriticalPenalty: 6.7,
        localPressurePenalty: 1.9,
        safeBurstBonus: 8.5,
        riskyBurstPenalty: 9.5
    )
    private let hardHexpandStrategicWeights = HexpandStrategicWeights(
        bias: 0.0,
        scoreDiff: 1.45,
        pieceDiff: 5.8,
        criticalDiff: 6.9,
        centerDiff: 0.55,
        edgeDiff: -2.35,
        burstPotentialDiff: 2.25,
        supportedCriticalDiff: 8.5,
        vulnerableCriticalPenalty: 10.5,
        frontierPressurePenalty: 3.1,
        unsupportedFrontierPenalty: 5.0,
        edgeTrapPenalty: 2.6,
        cornerTrapPenalty: 3.5,
        chainDiff: 6.4,
        exposedCriticalPenalty: 8.0,
        immediateWin: 12000.0,
        opponentImmediateWin: 3400.0
    )
    private let hardHexpandEvolvedTacticalBlend = 1.0
    private let hardHexpandNeuralTacticalBlend = 0.7
    private let hardHexpandNeuralThreatPenalty = 5.0
    private let mediumHexpandStrategicReplyWeight = 0.48
    private let hardHexpandStrategicReplyWeight = 0.72

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

        let roll = Int.random(in: 0..<100)
        if roll < 20 {
            return scored.randomElement()?.coord
        }
        let topCount = min(4, max(2, scored.count / 2))
        return scored[..<topCount].randomElement()?.coord
    }

    private func scoreHexpandEasyMove(_ move: AxialCoord, player: TileState, in state: HexpandState) -> Int {
        let ownerBefore = state.owners[move] ?? .empty
        let levelBefore = state.levels[move] ?? 0
        let next = simulateHexpandMove(at: move, player: player, in: state)
        let summary = summarizeHexpand(state: next, for: player)
        var score = (summary.aiScore - summary.humanScore)
            + (summary.aiLevel5 - summary.humanLevel5)
            + Int.random(in: -2...2)
        if ownerBefore == player {
            score += 1 + levelBefore
        }
        return score
    }

    private func bestHexpandMove(for player: TileState, depth: Int, in state: HexpandState) -> AxialCoord? {
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

    private func bestHexpandTacticalMove(for player: TileState, depth: Int, in state: HexpandState) -> AxialCoord? {
        let legalMoves = orderedHexpandTacticalMoves(for: player, in: state)
        guard !legalMoves.isEmpty else { return nil }

        let maxDepth = max(1, depth)
        let deadline = Date().timeIntervalSinceReferenceDate + tacticalSearchBudgetSeconds(in: state)
        let rootLimit = tacticalRootMoveLimit(in: state)
        var movePriority = legalMoves
        if movePriority.count > rootLimit {
            movePriority = Array(movePriority.prefix(rootLimit))
        }
        var bestMove = movePriority.first

        for currentDepth in 1...maxDepth {
            if Date().timeIntervalSinceReferenceDate >= deadline { break }
            var alpha = -Double.greatestFiniteMagnitude
            let beta = Double.greatestFiniteMagnitude
            var localBestMove: AxialCoord?
            var localBestScore = -Double.greatestFiniteMagnitude
            var scoredMoves: [(move: AxialCoord, score: Double)] = []
            var timedOut = false
            var cache: [HexpandTacticalCacheKey: HexpandTacticalCacheEntry] = [:]

            for move in movePriority {
                if Date().timeIntervalSinceReferenceDate >= deadline {
                    timedOut = true
                    break
                }
                let next = simulateHexpandMove(at: move, player: player, in: state)
                let score = tacticalSearchHexpand(
                    state: next,
                    player: player.opposite,
                    maximizingFor: player,
                    depth: currentDepth - 1,
                    alpha: alpha,
                    beta: beta,
                    deadline: deadline,
                    timedOut: &timedOut,
                    cache: &cache
                )
                if timedOut { break }
                scoredMoves.append((move: move, score: score))
                if score > localBestScore {
                    localBestScore = score
                    localBestMove = move
                }
                alpha = max(alpha, localBestScore)
            }

            if !timedOut, let localBestMove {
                bestMove = localBestMove
                movePriority = scoredMoves
                    .sorted { lhs, rhs in
                        if lhs.score != rhs.score { return lhs.score > rhs.score }
                        if lhs.move.q != rhs.move.q { return lhs.move.q < rhs.move.q }
                        return lhs.move.r < rhs.move.r
                    }
                    .map(\.move)
            } else {
                break
            }
        }

        return bestMove
    }

    private func tacticalSearchHexpand(
        state: HexpandState,
        player: TileState,
        maximizingFor: TileState,
        depth: Int,
        alpha: Double,
        beta: Double,
        deadline: TimeInterval,
        timedOut: inout Bool,
        cache: inout [HexpandTacticalCacheKey: HexpandTacticalCacheEntry]
    ) -> Double {
        if timedOut || Date().timeIntervalSinceReferenceDate >= deadline {
            timedOut = true
            return evaluateHexpandTactical(state: state, for: maximizingFor)
        }

        if let terminal = terminalHexpandScore(state: state, for: maximizingFor, depth: depth) {
            return Double(terminal)
        }
        if depth <= 0 {
            return evaluateHexpandTactical(state: state, for: maximizingFor)
        }

        let key = HexpandTacticalCacheKey(
            boardHash: hashHexpandState(state),
            playerValue: tileStateCode(player),
            depth: depth
        )
        if let cached = cache[key], cached.depth >= depth {
            return cached.score
        }

        var moves = orderedHexpandTacticalMoves(for: player, in: state)
        if moves.isEmpty {
            let score = evaluateHexpandTactical(state: state, for: maximizingFor)
            cache[key] = HexpandTacticalCacheEntry(depth: depth, score: score)
            return score
        }
        let limit = tacticalBranchLimit(depth: depth, totalMoves: moves.count, in: state)
        if moves.count > limit {
            moves = Array(moves.prefix(limit))
        }

        let maximizingTurn = player == maximizingFor
        var localAlpha = alpha
        var localBeta = beta
        var best = maximizingTurn ? -Double.greatestFiniteMagnitude : Double.greatestFiniteMagnitude

        for move in moves {
            if Date().timeIntervalSinceReferenceDate >= deadline {
                timedOut = true
                break
            }
            let next = simulateHexpandMove(at: move, player: player, in: state)
            let score = tacticalSearchHexpand(
                state: next,
                player: player.opposite,
                maximizingFor: maximizingFor,
                depth: depth - 1,
                alpha: localAlpha,
                beta: localBeta,
                deadline: deadline,
                timedOut: &timedOut,
                cache: &cache
            )

            if maximizingTurn {
                best = max(best, score)
                localAlpha = max(localAlpha, best)
            } else {
                best = min(best, score)
                localBeta = min(localBeta, best)
            }
            if localAlpha >= localBeta || timedOut {
                break
            }
        }

        if !timedOut {
            cache[key] = HexpandTacticalCacheEntry(depth: depth, score: best)
            return best
        }
        return evaluateHexpandTactical(state: state, for: maximizingFor)
    }

    private func orderedHexpandTacticalMoves(for player: TileState, in state: HexpandState) -> [AxialCoord] {
        let moves = orderedHexpandMoves(for: player, in: state)
        guard !moves.isEmpty else { return [] }
        let playerBefore = summarizeHexpandPolicy(state: state, for: player)
        let opponentBefore = summarizeHexpandPolicy(state: state, for: player.opposite)
        let playerTacticalBefore = summarizeHexpandTactical(state: state, for: player)

        return moves.sorted { lhs, rhs in
            let lhsScore = tacticalOrderingScore(
                move: lhs,
                player: player,
                in: state,
                playerBefore: playerBefore,
                opponentBefore: opponentBefore,
                playerTacticalBefore: playerTacticalBefore
            )
            let rhsScore = tacticalOrderingScore(
                move: rhs,
                player: player,
                in: state,
                playerBefore: playerBefore,
                opponentBefore: opponentBefore,
                playerTacticalBefore: playerTacticalBefore
            )
            if lhsScore != rhsScore { return lhsScore > rhsScore }
            if lhs.q != rhs.q { return lhs.q < rhs.q }
            return lhs.r < rhs.r
        }
    }

    private func tacticalOrderingScore(
        move: AxialCoord,
        player: TileState,
        in state: HexpandState,
        playerBefore: HexpandPolicySummary,
        opponentBefore: HexpandPolicySummary,
        playerTacticalBefore: HexpandTacticalSummary
    ) -> Double {
        let next = simulateHexpandMove(at: move, player: player, in: state)
        if isWinningHexpandState(next, for: player) {
            return 1_000_000
        }
        let evolvedScore = scoreHexpandEvolvedMove(
            move,
            player: player,
            in: state,
            playerBefore: playerBefore,
            opponentBefore: opponentBefore,
            playerTacticalBefore: playerTacticalBefore
        )
        let opponentImmediate = countImmediateHexpandWins(for: player.opposite, in: next, maxChecks: 10)
        var score = evolvedScore - (Double(opponentImmediate) * 250.0)
        let ownerBefore = state.owners[move] ?? .empty
        if ownerBefore == .empty {
            let support = emptyMoveSupportScore(at: move, player: player, in: state)
            let earlyWeight = isEarlyHexpandPhase(state) ? 22.0 : 8.0
            score += support * earlyWeight
            if support < 0 {
                score += support * (isEarlyHexpandPhase(state) ? 28.0 : 10.0)
            }
        }
        return score
    }

    private func evaluateHexpandTactical(state: HexpandState, for player: TileState) -> Double {
        let opponent = player.opposite
        let playerPolicy = summarizeHexpandPolicy(state: state, for: player)
        let opponentPolicy = summarizeHexpandPolicy(state: state, for: opponent)
        let playerTactical = summarizeHexpandTactical(state: state, for: player)
        let opponentTactical = summarizeHexpandTactical(state: state, for: opponent)

        let scoreDiff = playerPolicy.score - opponentPolicy.score
        let pieceDiff = playerPolicy.pieces - opponentPolicy.pieces
        let criticalDiff = playerPolicy.critical - opponentPolicy.critical
        let edgeDiff = playerPolicy.edgeLoad - opponentPolicy.edgeLoad
        let centerDiff = playerPolicy.centerControl - opponentPolicy.centerControl
        let nearCriticalDiff = playerTactical.nearCritical - opponentTactical.nearCritical
        let chainDiff = playerTactical.criticalChains - opponentTactical.criticalChains
        let exposedDiff = playerTactical.exposedCriticalPairs - opponentTactical.exposedCriticalPairs
        let immediateWins = countImmediateHexpandWins(for: player, in: state, maxChecks: 10)
        let opponentImmediateWins = countImmediateHexpandWins(for: opponent, in: state, maxChecks: 10)

        var score = largeHardHexpandWeights.bias
        score += largeHardHexpandWeights.scoreDiff * Double(scoreDiff)
        score += largeHardHexpandWeights.pieceDiff * Double(pieceDiff)
        score += largeHardHexpandWeights.criticalDiff * Double(criticalDiff)
        score += largeHardHexpandWeights.edgeDiff * Double(edgeDiff)
        score += largeHardHexpandWeights.centerDiff * Double(centerDiff)
        score += hardHexpandTacticalWeights.nearCriticalGain * Double(nearCriticalDiff)
        score += hardHexpandTacticalWeights.chainGain * Double(chainDiff)
        score -= hardHexpandTacticalWeights.exposedCriticalPenalty * Double(exposedDiff)
        score += Double(immediateWins) * 120.0
        score -= Double(opponentImmediateWins) * 180.0
        return score
    }

    private func tacticalBranchLimit(depth: Int, totalMoves: Int, in state: HexpandState) -> Int {
        let early = isEarlyHexpandPhase(state)
        if sideLength >= 6 {
            if early {
                if depth >= 4 { return min(totalMoves, 8) }
                if depth >= 2 { return min(totalMoves, 7) }
                return min(totalMoves, 6)
            }
            if depth >= 4 { return min(totalMoves, 10) }
            if depth >= 2 { return min(totalMoves, 9) }
            return min(totalMoves, 7)
        }
        if early {
            if depth >= 4 { return min(totalMoves, 11) }
            if depth >= 2 { return min(totalMoves, 10) }
            return min(totalMoves, 9)
        }
        if depth >= 4 { return min(totalMoves, 14) }
        if depth >= 2 { return min(totalMoves, 12) }
        return min(totalMoves, 10)
    }

    private func tacticalSearchBudgetSeconds(in state: HexpandState) -> TimeInterval {
        let early = isEarlyHexpandPhase(state)
        if sideLength >= 6 { return early ? 1.0 : 1.25 }
        if sideLength >= 4 { return early ? 0.85 : 1.0 }
        return early ? 0.7 : 0.85
    }

    private func tacticalRootMoveLimit(in state: HexpandState) -> Int {
        let early = isEarlyHexpandPhase(state)
        if sideLength >= 6 { return early ? 10 : 12 }
        if sideLength >= 4 { return early ? 12 : 14 }
        return early ? 10 : 12
    }

    private func isEarlyHexpandPhase(_ state: HexpandState) -> Bool {
        let total = boardCoords.isEmpty ? state.owners.count : boardCoords.count
        guard total > 0 else { return false }
        let occupied = (boardCoords.isEmpty ? Array(state.owners.keys) : boardCoords).reduce(into: 0) { count, coord in
            if (state.owners[coord] ?? .empty) != .empty {
                count += 1
            }
        }
        let occupancy = Double(occupied) / Double(total)
        let turnThreshold = max(12, total / 3)
        let occupancyThreshold = sideLength >= 6 ? 0.56 : 0.50
        return state.turnsPlayed < turnThreshold || occupancy < occupancyThreshold
    }

    private func emptyMoveSupportScore(at coord: AxialCoord, player: TileState, in state: HexpandState) -> Double {
        guard (state.owners[coord] ?? .empty) == .empty else { return 0 }
        var friendlyAdjacent = 0
        var enemyAdjacent = 0
        var friendlyLevels = 0
        var enemyLevels = 0
        var friendlyCritical = 0
        var enemyCritical = 0

        for neighbor in neighbors(of: coord, in: state.owners) {
            let owner = state.owners[neighbor] ?? .empty
            let level = state.levels[neighbor] ?? 0
            if owner == player {
                friendlyAdjacent += 1
                friendlyLevels += level
                if level >= 5 { friendlyCritical += 1 }
            } else if owner == player.opposite {
                enemyAdjacent += 1
                enemyLevels += level
                if level >= 5 { enemyCritical += 1 }
            }
        }

        let adjacencyTerm = Double(friendlyAdjacent - enemyAdjacent)
        let levelTerm = Double(friendlyLevels - enemyLevels) * 0.35
        let criticalTerm = Double(friendlyCritical - enemyCritical) * 1.5
        return adjacencyTerm + levelTerm + criticalTerm
    }

    private func hashHexpandState(_ state: HexpandState) -> Int {
        var hasher = Hasher()
        hasher.combine(state.turnsPlayed)
        let coords = boardCoords.isEmpty ? Array(state.owners.keys) : boardCoords
        for coord in coords {
            hasher.combine(coord.q)
            hasher.combine(coord.r)
            hasher.combine(tileStateCode(state.owners[coord] ?? .empty))
            hasher.combine(state.levels[coord] ?? 0)
        }
        return hasher.finalize()
    }

    private func tileStateCode(_ state: TileState) -> Int {
        switch state {
        case .empty: return 0
        case .red: return 1
        case .blue: return 2
        }
    }

    private func bestHexpandEvolvedMove(for player: TileState, in state: HexpandState) -> AxialCoord? {
        let moves = orderedHexpandMoves(for: player, in: state)
        guard !moves.isEmpty else { return nil }
        let playerBefore = summarizeHexpandPolicy(state: state, for: player)
        let opponentBefore = summarizeHexpandPolicy(state: state, for: player.opposite)
        let playerTacticalBefore = summarizeHexpandTactical(state: state, for: player)

        var bestScore = -Double.greatestFiniteMagnitude
        var bestMoves: [AxialCoord] = []

        for move in moves {
            let score = scoreHexpandEvolvedMove(
                move,
                player: player,
                in: state,
                playerBefore: playerBefore,
                opponentBefore: opponentBefore,
                playerTacticalBefore: playerTacticalBefore
            )
            if score > bestScore {
                bestScore = score
                bestMoves = [move]
            } else if score == bestScore {
                bestMoves.append(move)
            }
        }

        return bestMoves.randomElement()
    }

    private func bestHexpandStrategicMove(
        for player: TileState,
        difficulty: AIDifficulty,
        in state: HexpandState
    ) -> AxialCoord? {
        let moves = orderedHexpandMoves(for: player, in: state)
        guard !moves.isEmpty else { return nil }

        let replyWeight = difficulty == .hard ? hardHexpandStrategicReplyWeight : mediumHexpandStrategicReplyWeight
        let scoredMoves = moves.map { move in
            (
                move: move,
                score: scoreHexpandStrategicMove(
                    move,
                    player: player,
                    in: state,
                    replyWeight: replyWeight
                )
            )
        }
        .sorted { lhs, rhs in
            if lhs.score != rhs.score { return lhs.score > rhs.score }
            if lhs.move.q != rhs.move.q { return lhs.move.q < rhs.move.q }
            return lhs.move.r < rhs.move.r
        }

        guard let first = scoredMoves.first else { return nil }
        if difficulty == .hard {
            let closeMoves = scoredMoves.filter { abs($0.score - first.score) <= 0.5 }
            return closeMoves.randomElement()?.move ?? first.move
        }

        let topCount = min(4, scoredMoves.count)
        let weighted = scoredMoves.prefix(topCount)
        if let randomTop = weighted.randomElement(), Double.random(in: 0..<1) < 0.25 {
            return randomTop.move
        }
        return first.move
    }

    private func scoreHexpandStrategicMove(
        _ move: AxialCoord,
        player: TileState,
        in state: HexpandState,
        replyWeight: Double
    ) -> Double {
        let next = simulateHexpandMove(at: move, player: player, in: state)
        if isWinningHexpandState(next, for: player) {
            return hardHexpandStrategicWeights.immediateWin
        }

        var score = evaluateHexpandStrategicState(state: next, for: player)
        score += strategicMoveLocalAdjustment(
            move,
            player: player,
            before: state,
            after: next
        )

        let opponentImmediateWins = countImmediateHexpandWins(for: player.opposite, in: next, maxChecks: 8)
        score -= Double(opponentImmediateWins) * hardHexpandStrategicWeights.opponentImmediateWin

        let opponentReply = bestHexpandStrategicReplyScore(for: player.opposite, in: next)
        score -= opponentReply * replyWeight
        return score
    }

    private func bestHexpandStrategicReplyScore(for player: TileState, in state: HexpandState) -> Double {
        let replyMoves = orderedHexpandStrategicReplyMoves(for: player, in: state)
        guard !replyMoves.isEmpty else { return 0 }

        let replyCap = min(sideLength >= 4 ? 8 : 6, replyMoves.count)
        var bestReply = -Double.greatestFiniteMagnitude

        for move in replyMoves.prefix(replyCap) {
            let next = simulateHexpandMove(at: move, player: player, in: state)
            if isWinningHexpandState(next, for: player) {
                return hardHexpandStrategicWeights.immediateWin
            }
            let score = evaluateHexpandStrategicState(state: next, for: player)
                + strategicMoveLocalAdjustment(move, player: player, before: state, after: next)
            if score > bestReply {
                bestReply = score
            }
        }

        return bestReply == -Double.greatestFiniteMagnitude ? 0 : bestReply
    }

    private func orderedHexpandStrategicReplyMoves(for player: TileState, in state: HexpandState) -> [AxialCoord] {
        orderedHexpandMoves(for: player, in: state).sorted { lhs, rhs in
            let lhsScore = strategicReplyPreviewScore(lhs, player: player, in: state)
            let rhsScore = strategicReplyPreviewScore(rhs, player: player, in: state)
            if lhsScore != rhsScore { return lhsScore > rhsScore }
            if lhs.q != rhs.q { return lhs.q < rhs.q }
            return lhs.r < rhs.r
        }
    }

    private func strategicReplyPreviewScore(_ move: AxialCoord, player: TileState, in state: HexpandState) -> Double {
        let owner = state.owners[move] ?? .empty
        let level = state.levels[move] ?? 0
        var score = Double(level) * (owner == player ? 3.5 : 0.0)
        if owner == .empty {
            let support = emptyMoveSupportScore(at: move, player: player, in: state)
            score += support * 6.0
            if support < 0 {
                score += support * 9.0
            }
            if isEdge(move) { score -= 5.0 }
            if isCorner(move) { score -= 8.0 }
        } else if owner == player && level >= 5 {
            score += 10.0
        }
        return score
    }

    private func evaluateHexpandStrategicState(state: HexpandState, for player: TileState) -> Double {
        let opponent = player.opposite
        let playerPolicy = summarizeHexpandPolicy(state: state, for: player)
        let opponentPolicy = summarizeHexpandPolicy(state: state, for: opponent)
        let playerTactical = summarizeHexpandTactical(state: state, for: player)
        let opponentTactical = summarizeHexpandTactical(state: state, for: opponent)
        let playerStrategic = summarizeHexpandStrategic(state: state, for: player)
        let opponentStrategic = summarizeHexpandStrategic(state: state, for: opponent)

        let scoreDiff = playerPolicy.score - opponentPolicy.score
        let pieceDiff = playerPolicy.pieces - opponentPolicy.pieces
        let criticalDiff = playerPolicy.critical - opponentPolicy.critical
        let edgeDiff = playerPolicy.edgeLoad - opponentPolicy.edgeLoad
        let centerDiff = playerPolicy.centerControl - opponentPolicy.centerControl
        let chainDiff = playerTactical.criticalChains - opponentTactical.criticalChains
        let exposedDiff = playerTactical.exposedCriticalPairs - opponentTactical.exposedCriticalPairs
        let burstPotentialDiff = playerStrategic.burstPotential - opponentStrategic.burstPotential
        let supportedCriticalDiff = playerStrategic.supportedCriticals - opponentStrategic.supportedCriticals
        let vulnerableCriticalDiff = playerStrategic.vulnerableCriticals - opponentStrategic.vulnerableCriticals
        let frontierPressureDiff = playerStrategic.frontierPressure - opponentStrategic.frontierPressure
        let unsupportedFrontierDiff = playerStrategic.unsupportedFrontier - opponentStrategic.unsupportedFrontier
        let edgeTrapDiff = playerStrategic.edgeTrapLoad - opponentStrategic.edgeTrapLoad
        let cornerTrapDiff = playerStrategic.cornerTrapLoad - opponentStrategic.cornerTrapLoad

        var score = hardHexpandStrategicWeights.bias
        score += hardHexpandStrategicWeights.scoreDiff * Double(scoreDiff)
        score += hardHexpandStrategicWeights.pieceDiff * Double(pieceDiff)
        score += hardHexpandStrategicWeights.criticalDiff * Double(criticalDiff)
        score += hardHexpandStrategicWeights.centerDiff * Double(centerDiff)
        score += hardHexpandStrategicWeights.edgeDiff * Double(edgeDiff)
        score += hardHexpandStrategicWeights.burstPotentialDiff * Double(burstPotentialDiff)
        score += hardHexpandStrategicWeights.supportedCriticalDiff * Double(supportedCriticalDiff)
        score += hardHexpandStrategicWeights.chainDiff * Double(chainDiff)
        score -= hardHexpandStrategicWeights.vulnerableCriticalPenalty * Double(vulnerableCriticalDiff)
        score -= hardHexpandStrategicWeights.frontierPressurePenalty * Double(frontierPressureDiff)
        score -= hardHexpandStrategicWeights.unsupportedFrontierPenalty * Double(unsupportedFrontierDiff)
        score -= hardHexpandStrategicWeights.edgeTrapPenalty * Double(edgeTrapDiff)
        score -= hardHexpandStrategicWeights.cornerTrapPenalty * Double(cornerTrapDiff)
        score -= hardHexpandStrategicWeights.exposedCriticalPenalty * Double(exposedDiff)
        return score
    }

    private func summarizeHexpandStrategic(state: HexpandState, for player: TileState) -> HexpandStrategicSummary {
        var summary = HexpandStrategicSummary()
        let opponent = player.opposite
        let coords = boardCoords.isEmpty ? Array(state.owners.keys) : boardCoords

        for coord in coords {
            guard (state.owners[coord] ?? .empty) == player else { continue }
            let level = state.levels[coord] ?? 0
            var friendlyAdjacent = 0
            var enemyAdjacent = 0
            var friendlyCritical = 0
            var enemyCritical = 0
            var friendlyNearCritical = 0
            var enemyNearCritical = 0
            var enemyPressure = 0

            for neighbor in neighbors(of: coord, in: state.owners) {
                let owner = state.owners[neighbor] ?? .empty
                let neighborLevel = state.levels[neighbor] ?? 0
                if owner == player {
                    friendlyAdjacent += 1
                    if neighborLevel >= 5 {
                        friendlyCritical += 1
                    }
                    if neighborLevel >= 4 {
                        friendlyNearCritical += 1
                    }
                } else if owner == opponent {
                    enemyAdjacent += 1
                    enemyPressure += max(0, neighborLevel - level)
                    if neighborLevel >= 5 {
                        enemyCritical += 1
                    }
                    if neighborLevel >= 4 {
                        enemyNearCritical += 1
                    }
                }
            }

            if level >= 4 {
                let supportSwing = (friendlyCritical * 3) + (friendlyNearCritical * 2) + friendlyAdjacent
                    - (enemyCritical * 3) - (enemyNearCritical * 2) - enemyAdjacent
                summary.burstPotential += max(0, supportSwing)
                if supportSwing < 0 {
                    summary.frontierPressure += abs(supportSwing)
                }
            }

            if level >= 5 {
                if friendlyCritical + friendlyNearCritical >= enemyCritical + enemyNearCritical {
                    summary.supportedCriticals += 1
                }
                if enemyCritical > 0 || enemyNearCritical > friendlyCritical + friendlyNearCritical || enemyPressure > friendlyAdjacent {
                    summary.vulnerableCriticals += 1 + enemyCritical
                }
            }

            if enemyAdjacent > 0 {
                summary.frontierPressure += enemyPressure + max(0, enemyAdjacent - friendlyAdjacent)
            }

            if friendlyAdjacent == 0 && enemyAdjacent >= 2 {
                summary.unsupportedFrontier += enemyAdjacent + max(0, level - 1)
            } else if enemyAdjacent > friendlyAdjacent + 1 {
                summary.unsupportedFrontier += enemyAdjacent - friendlyAdjacent
            }

            if isCorner(coord) {
                let trapFactor = max(0, level - 1)
                if trapFactor > 0 && (enemyAdjacent >= friendlyAdjacent || level >= 4) {
                    summary.cornerTrapLoad += trapFactor * max(1, enemyAdjacent - friendlyAdjacent + (level >= 5 ? 1 : 0))
                }
            } else if isEdge(coord) {
                let trapFactor = max(0, level - 2)
                if trapFactor > 0 && (enemyAdjacent >= friendlyAdjacent || level >= 5) {
                    summary.edgeTrapLoad += trapFactor * max(1, enemyAdjacent - friendlyAdjacent + (level >= 5 ? 1 : 0))
                }
            }
        }

        return summary
    }

    private func strategicMoveLocalAdjustment(
        _ move: AxialCoord,
        player: TileState,
        before: HexpandState,
        after: HexpandState
    ) -> Double {
        let ownerBefore = before.owners[move] ?? .empty
        let levelBefore = before.levels[move] ?? 0
        var score = 0.0

        if ownerBefore == .empty {
            let support = emptyMoveSupportScore(at: move, player: player, in: before)
            score += support * 9.0
            if support < 0 {
                score += support * 12.0
            }
            if isEdge(move) {
                score -= max(0.0, 2.0 - support) * 8.0
            }
            if isCorner(move) {
                score -= max(0.0, 3.0 - support) * 10.0
            }
        } else {
            score += Double(levelBefore) * 1.5
        }

        let pressureAfter = higherEnemyNeighborPressure(at: move, for: player, in: after)
        score -= Double(pressureAfter) * 3.0

        if ownerBefore == player && levelBefore >= 5 {
            let exposedBefore = enemyCriticalNeighborCount(at: move, for: player, in: before)
            let exposedAfter = enemyCriticalNeighborCount(at: move, for: player, in: after)
            if exposedAfter <= exposedBefore {
                score += 8.0
            } else {
                score -= Double(exposedAfter - exposedBefore) * 10.0
            }
        }

        return score
    }

    private func enemyCriticalNeighborCount(at coord: AxialCoord, for player: TileState, in state: HexpandState) -> Int {
        neighbors(of: coord, in: state.owners).reduce(into: 0) { count, neighbor in
            guard (state.owners[neighbor] ?? .empty) == player.opposite else { return }
            if (state.levels[neighbor] ?? 0) >= 5 {
                count += 1
            }
        }
    }

    private func bestHexpandNeuralMove(for player: TileState, in state: HexpandState) -> AxialCoord? {
        guard sideLength >= 3 else { return nil }
        guard hexplodeNeuralEngine.isAvailable else { return nil }

        let moves = orderedHexpandMoves(for: player, in: state)
        guard !moves.isEmpty else { return nil }

        guard let rootInference = hexplodeNeuralEngine.infer(
            owners: state.owners,
            levels: state.levels,
            turnsPlayed: state.turnsPlayed,
            currentPlayer: player,
            boardCoords: boardCoords,
            radius: radius,
            legalMoves: moves
        ) else {
            return nil
        }

        var bestMoves: [AxialCoord] = []
        var bestScore = -Double.greatestFiniteMagnitude
        for move in moves {
            let next = simulateHexpandMove(at: move, player: player, in: state)
            if isWinningHexpandState(next, for: player) {
                return move
            }

            let prior = rootInference.policyScores[move] ?? -1_000
            let opponentMoves = orderedHexpandMoves(for: player.opposite, in: next)
            let opponentInference = hexplodeNeuralEngine.infer(
                owners: next.owners,
                levels: next.levels,
                turnsPlayed: next.turnsPlayed,
                currentPlayer: player.opposite,
                boardCoords: boardCoords,
                radius: radius,
                legalMoves: opponentMoves
            )

            // Neural-only scoring: policy prior + value estimate from the neural model.
            let valueScore = -(opponentInference?.value ?? 0)
            let score = prior
                + (valueScore * 2.0)
            if score > bestScore {
                bestScore = score
                bestMoves = [move]
            } else if score == bestScore {
                bestMoves.append(move)
            }
        }
        return bestMoves.randomElement()
    }

    private func bestHexpandRandomLegalMove(for player: TileState, in state: HexpandState) -> AxialCoord? {
        let legal = hexpandLegalMoves(for: player, in: state)
        guard !legal.isEmpty else { return nil }
        return legal.randomElement()
    }

    private func scoreHexpandEvolvedMove(
        _ move: AxialCoord,
        player: TileState,
        in state: HexpandState,
        playerBefore: HexpandPolicySummary,
        opponentBefore: HexpandPolicySummary,
        playerTacticalBefore: HexpandTacticalSummary
    ) -> Double {
        let next = simulateHexpandMove(at: move, player: player, in: state)

        if isWinningHexpandState(next, for: player) {
            return largeHardHexpandWeights.immediateWin
        }

        let playerAfter = summarizeHexpandPolicy(state: next, for: player)
        let opponentAfter = summarizeHexpandPolicy(state: next, for: player.opposite)
        let opponentImmediateWins = countImmediateHexpandWins(for: player.opposite, in: next, maxChecks: 14)

        let scoreDiff = playerAfter.score - opponentAfter.score
        let pieceDiff = playerAfter.pieces - opponentAfter.pieces
        let criticalDiff = playerAfter.critical - opponentAfter.critical
        let edgeDiff = playerAfter.edgeLoad - opponentAfter.edgeLoad
        let centerDiff = playerAfter.centerControl - opponentAfter.centerControl
        let swing = (playerAfter.score - playerBefore.score) - (opponentAfter.score - opponentBefore.score)

        var score = largeHardHexpandWeights.bias
        score += largeHardHexpandWeights.scoreDiff * Double(scoreDiff)
        score += largeHardHexpandWeights.pieceDiff * Double(pieceDiff)
        score += largeHardHexpandWeights.criticalDiff * Double(criticalDiff)
        score += largeHardHexpandWeights.edgeDiff * Double(edgeDiff)
        score += largeHardHexpandWeights.centerDiff * Double(centerDiff)
        score += largeHardHexpandWeights.swing * Double(swing)
        score -= largeHardHexpandWeights.opponentImmediateWin * Double(opponentImmediateWins)
        let tacticalScore = scoreHexpandTacticalMove(
            move,
            player: player,
            before: state,
            after: next,
            playerBefore: playerTacticalBefore
        )
        score += tacticalScore * hardHexpandEvolvedTacticalBlend
        return score
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

        let moves = orderedHexpandMoves(for: player, in: state)
        if moves.isEmpty {
            return evaluateHexpand(state: state, for: maximizingFor)
        }

        let maximizingTurn = player == maximizingFor
        var best = maximizingTurn ? Int.min : Int.max
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
                }
                localAlpha = max(localAlpha, best)
            } else {
                if score < best {
                    best = score
                }
                localBeta = min(localBeta, best)
            }
            if localAlpha >= localBeta {
                break
            }
        }
        alpha = localAlpha
        beta = localBeta
        return best
    }

    private func evaluateHexpand(state: HexpandState, for player: TileState) -> Int {
        let summary = summarizeHexpand(state: state, for: player)
        return (summary.aiScore - summary.humanScore) + 2 * (summary.aiLevel5 - summary.humanLevel5)
    }

    private func currentHexpandState() -> HexpandState {
        HexpandState(
            owners: tileStates,
            levels: hexpandLevels,
            turnsPlayed: movesMade,
            redHadTile: redHadTile,
            blueHadTile: blueHadTile
        )
    }

    private func isWinningHexpandState(_ state: HexpandState, for player: TileState) -> Bool {
        guard let terminal = terminalHexpandScore(state: state, for: player, depth: 0) else {
            return false
        }
        return terminal > 0
    }

    private func countImmediateHexpandWins(for player: TileState, in state: HexpandState, maxChecks: Int) -> Int {
        var checks = 0
        var wins = 0
        for move in orderedHexpandMoves(for: player, in: state) {
            if checks >= maxChecks { break }
            checks += 1
            let next = simulateHexpandMove(at: move, player: player, in: state)
            if isWinningHexpandState(next, for: player) {
                wins += 1
                if wins >= 2 {
                    return wins
                }
            }
        }
        return wins
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
            let burstCount = currentLevel / 6
            let remainder = currentLevel % 6
            if burstCount <= 0 { continue }

            if remainder > 0 {
                state.owners[current] = explodingPlayer
                state.levels[current] = remainder
            } else {
                state.owners[current] = .empty
                state.levels[current] = 0
            }

            for direction in HexGrid.directions {
                let neighbor = current.adding(direction)
                guard state.owners[neighbor] != nil else { continue }
                state.owners[neighbor] = explodingPlayer
                state.levels[neighbor] = (state.levels[neighbor] ?? 0) + burstCount
                if (state.levels[neighbor] ?? 0) > 5, !queued.contains(neighbor) {
                    queue.append(neighbor)
                    queued.insert(neighbor)
                }
            }
        }
    }

    private func orderedHexpandMoves(
        for player: TileState,
        in state: HexpandState
    ) -> [AxialCoord] {
        hexpandLegalMoves(for: player, in: state).sorted { lhs, rhs in
            let lhsScore = hexpandMoveOrderingScore(lhs, player: player, in: state)
            let rhsScore = hexpandMoveOrderingScore(rhs, player: player, in: state)
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
        in state: HexpandState
    ) -> Int {
        let owner = state.owners[coord] ?? .empty
        return owner == player ? (state.levels[coord] ?? 0) : 0
    }

    private func summarizeHexpand(state: HexpandState, for player: TileState) -> HexpandSummary {
        var summary = HexpandSummary()
        let opponent = player.opposite
        let coords = boardCoords.isEmpty ? Array(state.owners.keys) : boardCoords
        for coord in coords {
            let owner = state.owners[coord] ?? .empty
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

    private func summarizeHexpandPolicy(state: HexpandState, for player: TileState) -> HexpandPolicySummary {
        var summary = HexpandPolicySummary()
        let coords = boardCoords.isEmpty ? Array(state.owners.keys) : boardCoords
        for coord in coords {
            guard (state.owners[coord] ?? .empty) == player else { continue }
            let level = state.levels[coord] ?? 0
            let x = coord.q
            let z = coord.r
            let y = -x - z
            let distanceFromCenter = max(abs(x), max(abs(y), abs(z)))
            summary.score += level
            summary.pieces += 1
            if level >= 5 {
                summary.critical += 1
            }
            if isCorner(coord) {
                summary.edgeLoad += level * 2
            } else if isEdge(coord) {
                summary.edgeLoad += level
            }
            summary.centerControl += max(0, radius - distanceFromCenter) * level
        }
        return summary
    }

    private func summarizeHexpandTactical(state: HexpandState, for player: TileState) -> HexpandTacticalSummary {
        var summary = HexpandTacticalSummary()
        let opponent = player.opposite
        let coords = boardCoords.isEmpty ? Array(state.owners.keys) : boardCoords
        for coord in coords {
            guard (state.owners[coord] ?? .empty) == player else { continue }
            let level = state.levels[coord] ?? 0
            if level >= 4 {
                summary.nearCritical += 1
            }
            if level >= 5 {
                summary.critical += 1
                for neighbor in neighbors(of: coord, in: state.owners) where shouldCountPair(coord, neighbor) {
                    let neighborOwner = state.owners[neighbor] ?? .empty
                    let neighborLevel = state.levels[neighbor] ?? 0
                    if neighborOwner == player && neighborLevel >= 5 {
                        summary.criticalChains += 1
                    } else if neighborOwner == opponent && neighborLevel >= 5 {
                        summary.exposedCriticalPairs += 1
                    }
                }
            }
        }
        return summary
    }

    private func scoreHexpandTacticalMove(
        _ move: AxialCoord,
        player: TileState,
        before: HexpandState,
        after: HexpandState,
        playerBefore: HexpandTacticalSummary
    ) -> Double {
        let weights = hardHexpandTacticalWeights
        let playerAfter = summarizeHexpandTactical(state: after, for: player)
        let ownerBefore = before.owners[move] ?? .empty
        let levelBefore = before.levels[move] ?? 0

        let nearCriticalDelta = playerAfter.nearCritical - playerBefore.nearCritical
        let criticalDelta = playerAfter.critical - playerBefore.critical
        let chainDelta = playerAfter.criticalChains - playerBefore.criticalChains
        let exposedDelta = playerAfter.exposedCriticalPairs - playerBefore.exposedCriticalPairs

        var score = 0.0
        score += weights.nearCriticalGain * Double(nearCriticalDelta)
        score += weights.criticalGain * Double(criticalDelta)
        score += weights.chainGain * Double(chainDelta)
        score -= weights.exposedCriticalPenalty * Double(exposedDelta)

        if ownerBefore == player {
            let localPressure = higherEnemyNeighborPressure(at: move, for: player, in: after)
            score -= weights.localPressurePenalty * Double(localPressure)
        }

        if ownerBefore == player && levelBefore >= 5 {
            if exposedDelta <= 0 {
                score += weights.safeBurstBonus * Double(1 - exposedDelta)
            } else {
                score -= weights.riskyBurstPenalty * Double(exposedDelta)
            }
        }

        return score
    }

    private func higherEnemyNeighborPressure(at coord: AxialCoord, for player: TileState, in state: HexpandState) -> Int {
        guard (state.owners[coord] ?? .empty) == player else { return 0 }
        let ownLevel = state.levels[coord] ?? 0
        var pressure = 0
        for neighbor in neighbors(of: coord, in: state.owners) where (state.owners[neighbor] ?? .empty) == player.opposite {
            let enemyLevel = state.levels[neighbor] ?? 0
            if enemyLevel > ownLevel {
                pressure += enemyLevel - ownLevel
            }
        }
        return pressure
    }

    private func shouldCountPair(_ lhs: AxialCoord, _ rhs: AxialCoord) -> Bool {
        lhs.q < rhs.q || (lhs.q == rhs.q && lhs.r < rhs.r)
    }

    private func terminalHexpandScore(state: HexpandState, for player: TileState, depth: Int) -> Int? {
        let summary = summarizeHexpand(state: state, for: player)
        if state.turnsPlayed >= 2 {
            if summary.aiPieces > 0 && summary.humanPieces == 0 {
                return 10000 + depth
            }
            if summary.humanPieces > 0 && summary.aiPieces == 0 {
                return -10000 - depth
            }
            if summary.aiPieces == 0 && summary.humanPieces == 0 {
                return 0
            }
        }
        return nil
    }

    private func isEdge(_ coord: AxialCoord) -> Bool {
        isEdge(coord, radius: radius)
    }

    private func isEdge(_ coord: AxialCoord, radius: Int) -> Bool {
        let x = coord.q
        let z = coord.r
        let y = -x - z
        return max(abs(x), max(abs(y), abs(z))) == radius
    }

    private func isCorner(_ coord: AxialCoord) -> Bool {
        isCorner(coord, radius: radius)
    }

    private func isCorner(_ coord: AxialCoord, radius: Int) -> Bool {
        return (coord.q == radius && coord.r == 0)
            || (coord.q == 0 && coord.r == radius)
            || (coord.q == -radius && coord.r == radius)
            || (coord.q == -radius && coord.r == 0)
            || (coord.q == 0 && coord.r == -radius)
            || (coord.q == radius && coord.r == -radius)
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
            self.refreshHexelloMoveHighlights()
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

    private func playTakeover() {
        guard let takeoverAudio else { return }
        boardNode.runAction(SCNAction.playAudio(takeoverAudio, waitForCompletion: false))
    }

    private func playJump() {
        guard let jumpAudio else { return }
        boardNode.runAction(SCNAction.playAudio(jumpAudio, waitForCompletion: false))
    }

    private func playSplit() {
        guard let splitAudio else { return }
        boardNode.runAction(SCNAction.playAudio(splitAudio, waitForCompletion: false))
    }

    private func playRaise() {
        guard let raiseAudio else { return }
        boardNode.runAction(SCNAction.playAudio(raiseAudio, waitForCompletion: false))
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

    private func scheduleHexelloAIMove() {
        let delay: TimeInterval = movesMade == 0 ? 1.0 : 0.4
        let player = currentPlayer
        let stateSnapshot = tileStates
        let difficulty = aiDifficulty
        let sideLength = self.sideLength
        let token = aiComputationToken
        setAIScheduling(true, player: player)

        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            self.aiComputeQueue.async { [weak self] in
                guard let self else { return }
                let move = self.bestMove(
                    for: player,
                    in: stateSnapshot,
                    difficulty: difficulty,
                    sideLength: sideLength
                )
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    defer { self.setAIScheduling(false) }
                    guard self.aiComputationToken == token else { return }
                    guard self.currentPlayer == player else { return }
                    let aiEnabled = (player == .red && self.aiRedEnabled) || (player == .blue && self.aiBlueEnabled)
                    guard aiEnabled, !self.isAnimatingMove, self.introCompleted, !self.isGameOver else { return }
                    guard let move else { return }
                    self.applyMove(at: move)
                }
            }
        }
    }

    private func scheduleHexfectionAIMove() {
        let delay: TimeInterval = movesMade == 0 ? 1.0 : 0.4
        let player = currentPlayer
        let stateSnapshot = tileStates
        let difficulty = aiDifficulty
        let token = aiComputationToken
        setAIScheduling(true, player: player)

        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            self.aiComputeQueue.async { [weak self] in
                guard let self else { return }
                let move = self.bestHexfectionMove(for: player, in: stateSnapshot, difficulty: difficulty)
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    defer { self.setAIScheduling(false) }
                    guard self.aiComputationToken == token else { return }
                    guard self.currentPlayer == player else { return }
                    let aiEnabled = (player == .red && self.aiRedEnabled) || (player == .blue && self.aiBlueEnabled)
                    guard aiEnabled, !self.isAnimatingMove, self.introCompleted, !self.isGameOver else { return }
                    guard let move else { return }
                    self.startHexfectionMove(
                        from: move.source,
                        to: move.destination,
                        type: move.type,
                        player: player
                    )
                }
            }
        }
    }

    private func scheduleHexpandAIMove() {
        let delay: TimeInterval = movesMade == 0 ? 1.0 : 0.4
        let player = currentPlayer
        let stateSnapshot = currentHexpandState()
        let difficulty = aiDifficulty
        let token = aiComputationToken
        setAIScheduling(true, player: player)

        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            self.aiComputeQueue.async { [weak self] in
                guard let self else { return }
                let move = self.bestHexpandMove(
                    for: player,
                    in: stateSnapshot,
                    difficulty: difficulty
                )
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    defer { self.setAIScheduling(false) }
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

    private func bestHexpandMove(
        for player: TileState,
        in state: HexpandState,
        difficulty: AIDifficulty
    ) -> AxialCoord? {
        let engine = HexplodeSearchEngine(boardCoords: boardCoords, radius: radius)
        let bestMove = engine.chooseMove(
            owners: state.owners,
            levels: state.levels,
            turnsPlayed: state.turnsPlayed,
            currentPlayer: player,
            redHadTile: state.redHadTile,
            blueHadTile: state.blueHadTile,
            timeBudget: hexplodeHardSearchBudget()
        )?.bestMove
            ?? bestHexpandStrategicMove(for: player, difficulty: .hard, in: state)
            ?? bestHexpandEvolvedMove(for: player, in: state)
            ?? bestHexpandRandomLegalMove(for: player, in: state)

        guard let bestMove else { return nil }

        let randomOverrideDenominator: Int?
        switch difficulty {
        case .easy:
            randomOverrideDenominator = 3
        case .medium:
            randomOverrideDenominator = 5
        case .hard:
            randomOverrideDenominator = nil
        }

        guard let denominator = randomOverrideDenominator else {
            return bestMove
        }
        guard Int.random(in: 1...denominator) == 1 else {
            return bestMove
        }

        let alternatives = hexpandLegalMoves(for: player, in: state).filter { $0 != bestMove }
        return alternatives.randomElement() ?? bestMove
    }

    private func hexpandSearchDepth() -> Int {
        guard aiDifficulty != .easy else {
            return 2
        }
        return 5
    }

    private func hexplodeHardSearchBudget() -> TimeInterval {
        switch boardCoords.count {
        case ..<20:
            return 1.0
        case ..<40:
            return 1.5
        default:
            return 2.25
        }
    }

    private func hexfectionHardSearchBudget() -> TimeInterval {
        switch boardCoords.count {
        case ..<40:
            return 0.8
        case ..<70:
            return 1.2
        default:
            return 1.8
        }
    }
}
