import SwiftUI
import SceneKit
import AVFoundation
import Combine

struct ContentView: View {
    @State private var turnText: String = ""
    @State private var messageText: String = ""
    @State private var showTitle = true
    @State private var redAI = false
    @State private var blueAI = true
    @State private var startToken = UUID()
    @State private var redScore: Int = 0
    @State private var blueScore: Int = 0
    @State private var turnCount: Int = 0
    @State private var gameOverText: String = ""
    @State private var winnerText: String = ""
    @State private var gameEnded = false
    @State private var pulsePhase: CGFloat = 0
    @State private var activePlayer: TileState = .empty
    @State private var boardSizeOption: BoardSizeOption = .large
    @State private var yawValue: Float = 0
    @State private var pitchValue: Float = 0
    @State private var resetToken = UUID()
    @State private var gameMode: GameMode = .hexello
    @State private var aiDifficulty: AIDifficulty = .hard
    @StateObject private var soundPlayer = SoundPlayer()
    @State private var canReturnToTitleFromHUD = true

    private func returnToTitleFromHUD() {
        guard !showTitle, canReturnToTitleFromHUD else { return }
        resetToken = UUID()
        showTitle = true
    }

    private var gameSceneLayer: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.2, green: 0.2, blue: 0.45),
                    .black
                ],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()
            GameSceneView(
                turnText: $turnText,
                messageText: $messageText,
                redAI: $redAI,
                blueAI: $blueAI,
                boardSizeOption: $boardSizeOption,
                gameMode: $gameMode,
                aiDifficulty: $aiDifficulty,
                startToken: $startToken,
                resetToken: $resetToken,
                redScore: $redScore,
                blueScore: $blueScore,
                gameOverText: $gameOverText,
                winnerText: $winnerText,
                gameEnded: $gameEnded,
                activePlayer: $activePlayer,
                turnCount: $turnCount,
                yawValue: $yawValue,
                pitchValue: $pitchValue
            )
            .ignoresSafeArea()
        }
    }

    private var hudLayer: some View {
        VStack {
            HStack {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text("Red")
                        .font(.system(size: 28, weight: .heavy, design: .rounded))
                        .foregroundStyle(.white.opacity(0.85))
                    Text("\(redScore)")
                        .font(.system(size: 28, weight: .heavy, design: .rounded))
                        .foregroundStyle(.red)
                }
                .frame(width: 120, height: 32, alignment: .leading)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color.white.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(Color.black.opacity(0.85), lineWidth: 3)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(
                            Color.yellow.opacity(activePlayer == .red && !gameEnded ? 0.9 : 0.0),
                            lineWidth: activePlayer == .red && !gameEnded ? (2.5 + 1.2 * pulsePhase) : 0
                        )
                        .shadow(
                            color: Color.yellow.opacity(activePlayer == .red && !gameEnded ? (0.18 + 0.18 * pulsePhase) : 0),
                            radius: activePlayer == .red && !gameEnded ? (2 + 3 * pulsePhase) : 0,
                            x: 0,
                            y: 0
                        )
                )
                .clipShape(RoundedRectangle(cornerRadius: 14))

                Spacer()

                VStack(spacing: 1) {
                    Text("Turn")
                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white.opacity(0.82))
                    Text("\(turnCount)")
                        .font(.system(size: 26, weight: .heavy, design: .rounded))
                        .foregroundStyle(.white)
                }
                .frame(width: 88, height: 46)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.white.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.black.opacity(0.85), lineWidth: 3)
                )
                .clipShape(RoundedRectangle(cornerRadius: 12))

                Spacer()

                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text("Blue")
                        .font(.system(size: 28, weight: .heavy, design: .rounded))
                        .foregroundStyle(.white.opacity(0.85))
                    Text("\(blueScore)")
                        .font(.system(size: 28, weight: .heavy, design: .rounded))
                        .foregroundStyle(.blue)
                }
                .frame(width: 120, height: 32, alignment: .trailing)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color.white.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(Color.black.opacity(0.85), lineWidth: 3)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(
                            Color.yellow.opacity(activePlayer == .blue && !gameEnded ? 0.9 : 0.0),
                            lineWidth: activePlayer == .blue && !gameEnded ? (2.5 + 1.2 * pulsePhase) : 0
                        )
                        .shadow(
                            color: Color.yellow.opacity(activePlayer == .blue && !gameEnded ? (0.18 + 0.18 * pulsePhase) : 0),
                            radius: activePlayer == .blue && !gameEnded ? (2 + 3 * pulsePhase) : 0,
                            x: 0,
                            y: 0
                        )
                )
                .clipShape(RoundedRectangle(cornerRadius: 14))
            }
            .padding(.horizontal, 16)
            .padding(.top, 16)

            if !messageText.isEmpty && !gameEnded {
                Text(messageText)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.white.opacity(0.85))
                    .padding(.top, 6)
            }

            Spacer()

            if gameEnded {
                VStack(spacing: 6) {
                    Text(gameOverText)
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(.white)
                    Text(winnerText)
                        .font(.system(size: 22, weight: .heavy, design: .rounded))
                        .foregroundStyle(.white)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(Color.black.opacity(0.85), lineWidth: 3)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(
                            Color.yellow.opacity(0.85),
                            lineWidth: 2.5 + 1.2 * pulsePhase
                        )
                        .shadow(
                            color: Color.yellow.opacity(0.18 + 0.18 * pulsePhase),
                            radius: 2 + 3 * pulsePhase,
                            x: 0,
                            y: 0
                        )
                )
                .padding(.bottom, 24)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .allowsHitTesting(true)
        .onTapGesture(count: 2) {
            returnToTitleFromHUD()
        }
        .onTapGesture {
            guard gameEnded else { return }
            returnToTitleFromHUD()
        }
        .onAppear {
            pulsePhase = 0
            withAnimation(.easeInOut(duration: 0.45).repeatForever(autoreverses: true)) {
                pulsePhase = 1
            }
        }
        .onChange(of: turnText) {
            guard !turnText.isEmpty, !gameEnded else { return }
            pulsePhase = 0
            withAnimation(.easeInOut(duration: 0.45).repeatForever(autoreverses: true)) {
                pulsePhase = 1
            }
        }
        .onChange(of: activePlayer) {
            guard activePlayer != .empty, !gameEnded else { return }
            pulsePhase = 0
            withAnimation(.easeInOut(duration: 0.45).repeatForever(autoreverses: true)) {
                pulsePhase = 1
            }
        }
        .onChange(of: gameEnded) {
            guard gameEnded else { return }
            pulsePhase = 0
            withAnimation(.easeInOut(duration: 0.45).repeatForever(autoreverses: true)) {
                pulsePhase = 1
            }
        }
    }

    var body: some View {
        ZStack {
            if !showTitle {
                gameSceneLayer
            }

            if !showTitle {
                hudLayer
            }

            if showTitle {
                TitleScreen(
                    redAI: $redAI,
                    blueAI: $blueAI,
                    boardSizeOption: $boardSizeOption,
                    gameMode: $gameMode,
                    aiDifficulty: $aiDifficulty,
                    onToggle: {
                        soundPlayer.playFlip()
                    },
                    onStart: {
                        showTitle = false
                        canReturnToTitleFromHUD = false
                        turnText = ""
                        messageText = ""
                        gameOverText = ""
                        winnerText = ""
                        gameEnded = false
                        redScore = 0
                        blueScore = 0
                        turnCount = 0
                        activePlayer = .empty
                        startToken = UUID()
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2.8) {
                            if !showTitle {
                                canReturnToTitleFromHUD = true
                            }
                        }
                    }
                )
            }
        }
    }
}

private struct GameSceneView: UIViewRepresentable {
    @Binding var turnText: String
    @Binding var messageText: String
    @Binding var redAI: Bool
    @Binding var blueAI: Bool
    @Binding var boardSizeOption: BoardSizeOption
    @Binding var gameMode: GameMode
    @Binding var aiDifficulty: AIDifficulty
    @Binding var startToken: UUID
    @Binding var resetToken: UUID
    @Binding var redScore: Int
    @Binding var blueScore: Int
    @Binding var gameOverText: String
    @Binding var winnerText: String
    @Binding var gameEnded: Bool
    @Binding var activePlayer: TileState
    @Binding var turnCount: Int
    @Binding var yawValue: Float
    @Binding var pitchValue: Float

    func makeCoordinator() -> Coordinator {
        Coordinator(
            onTurn: { text in
                turnText = text
            },
            onMessage: { text in
                messageText = text
            },
            onScore: { red, blue in
                redScore = red
                blueScore = blue
            },
            onGameOver: { message, winner in
                gameOverText = message
                winnerText = winner
                gameEnded = true
            },
            onActive: { player in
                activePlayer = player
            },
            onTurnCount: { value in
                turnCount = value
            },
            onYaw: { yaw in
                yawValue = yaw
            },
            onPitch: { pitch in
                pitchValue = pitch
            }
        )
    }

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        context.coordinator.configure(view)
        return view
    }

    static func dismantleUIView(_ uiView: SCNView, coordinator: Coordinator) {
        coordinator.shutdown()
        uiView.gestureRecognizers?.forEach { uiView.removeGestureRecognizer($0) }
        uiView.scene = nil
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        context.coordinator.controller.setGameMode(gameMode)
        context.coordinator.controller.setAIModes(redAI: redAI, blueAI: blueAI)
        let effectiveLength: Int
        switch gameMode {
        case .hexello, .hexfection:
            effectiveLength = boardSizeOption == .large ? 6 : 4
        case .hexpand:
            effectiveLength = boardSizeOption == .large ? 4 : 3
        }
        context.coordinator.controller.setBoardSize(effectiveLength)
        context.coordinator.controller.setAIDifficulty(aiDifficulty)
        context.coordinator.handleReset(token: resetToken)
        context.coordinator.handleStart(token: startToken)
    }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        let controller = GameSceneController()
        private let onTurn: (String) -> Void
        private let onMessage: (String) -> Void
        private let onScore: (Int, Int) -> Void
        private let onGameOver: (String, String) -> Void
        private let onActive: (TileState) -> Void
        private let onTurnCount: (Int) -> Void
        private let onYaw: (Float) -> Void
        private let onPitch: (Float) -> Void
        private var lastStartToken = UUID()
        private var lastResetToken = UUID()

        init(
            onTurn: @escaping (String) -> Void,
            onMessage: @escaping (String) -> Void,
            onScore: @escaping (Int, Int) -> Void,
            onGameOver: @escaping (String, String) -> Void,
            onActive: @escaping (TileState) -> Void,
            onTurnCount: @escaping (Int) -> Void,
            onYaw: @escaping (Float) -> Void,
            onPitch: @escaping (Float) -> Void
        ) {
            self.onTurn = onTurn
            self.onMessage = onMessage
            self.onScore = onScore
            self.onGameOver = onGameOver
            self.onActive = onActive
            self.onTurnCount = onTurnCount
            self.onYaw = onYaw
            self.onPitch = onPitch
            super.init()
            controller.onTurnUpdate = { state in
                switch state {
                case .red:
                    onTurn("Red's turn")
                case .blue:
                    onTurn("Blue's turn")
                case .empty:
                    onTurn("")
                }
            }
            controller.onActiveUpdate = { state in
                onActive(state)
            }
            controller.onYawUpdate = { value in
                onYaw(value)
            }
            controller.onPitchUpdate = { value in
                onPitch(value)
            }
            controller.onMessageUpdate = { text in
                onMessage(text)
            }
            controller.onScoreUpdate = { red, blue in
                onScore(red, blue)
            }
            controller.onTurnCountUpdate = { count in
                onTurnCount(count)
            }
            controller.onGameOver = { [weak self] message, winner in
                onGameOver(message, winner)
                self?.controller.startGameOverAnimation()
            }
        }

        deinit {
            shutdown()
        }

        func shutdown() {
            controller.prepareForMenuReturn()
            controller.onTurnUpdate = nil
            controller.onMessageUpdate = nil
            controller.onScoreUpdate = nil
            controller.onGameOver = nil
            controller.onActiveUpdate = nil
            controller.onTurnCountUpdate = nil
            controller.onYawUpdate = nil
            controller.onPitchUpdate = nil
        }

        func configure(_ view: SCNView) {
            controller.configureView(view)

            let pan = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            pan.maximumNumberOfTouches = 1
            view.addGestureRecognizer(pan)

            let twoFingerPan = UIPanGestureRecognizer(target: self, action: #selector(handleTwoFingerPan(_:)))
            twoFingerPan.minimumNumberOfTouches = 2
            twoFingerPan.maximumNumberOfTouches = 2
            twoFingerPan.delegate = self
            view.addGestureRecognizer(twoFingerPan)

            let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
            view.addGestureRecognizer(tap)

            let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
            pinch.delegate = self
            view.addGestureRecognizer(pinch)
        }

        @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
            controller.handlePan(gesture)
        }

        @objc private func handleTwoFingerPan(_ gesture: UIPanGestureRecognizer) {
            controller.handleTwoFingerPan(gesture)
        }

        @objc private func handleTap(_ gesture: UITapGestureRecognizer) {
            guard let view = gesture.view as? SCNView else { return }
            controller.handleTap(gesture, in: view)
        }

        @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            controller.handlePinch(gesture)
        }

        func gestureRecognizer(
            _ gestureRecognizer: UIGestureRecognizer,
            shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer
        ) -> Bool {
            (gestureRecognizer is UIPinchGestureRecognizer && otherGestureRecognizer is UIPanGestureRecognizer)
                || (gestureRecognizer is UIPanGestureRecognizer && otherGestureRecognizer is UIPinchGestureRecognizer)
        }

        func handleStart(token: UUID) {
            guard token != lastStartToken else { return }
            lastStartToken = token
            controller.stopGameOverAnimation()
            controller.startNewGame()
            onActive(controller.currentPlayerState())
            onTurnCount(0)
        }

        func handleReset(token: UUID) {
            guard token != lastResetToken else { return }
            lastResetToken = token
            controller.prepareForMenuReturn()
        }
    }
}

#Preview {
    ContentView()
}
private struct TitleScreen: View {
    private enum TitleStep {
        case mode
        case options
    }

    private struct ModeCardMeta {
        let mode: GameMode
        let title: String
        let subtitle: String
        let detail: String
        let symbol: String
        let colors: [Color]
    }

    @Binding var redAI: Bool
    @Binding var blueAI: Bool
    @Binding var boardSizeOption: BoardSizeOption
    @Binding var gameMode: GameMode
    @Binding var aiDifficulty: AIDifficulty
    @Environment(\.verticalSizeClass) private var verticalSizeClass
    let onToggle: () -> Void
    let onStart: () -> Void
    @State private var logoScale: CGFloat = 0.1
    @State private var startScale: CGFloat = 1.0
    @State private var step: TitleStep = .mode

    private var titleLogoName: String {
        UIImage(named: "hex2") != nil ? "hex2" : "hex"
    }

    private var buildLabel: String {
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"
        let buildStamp = bundleBuildStamp()
        return "Build \(build)  \(buildStamp)  R43"
    }

    private var modeCards: [ModeCardMeta] {
        [
            ModeCardMeta(
                mode: .hexello,
                title: "Hexello",
                subtitle: "Classic Territory Control",
                detail: "Capture lines and dominate key positions.",
                symbol: "circle.hexagongrid.fill",
                colors: [Color.blue.opacity(0.95), Color.cyan.opacity(0.8)]
            ),
            ModeCardMeta(
                mode: .hexpand,
                title: "Hexplode",
                subtitle: "Chain Reaction Battles",
                detail: "Stack pressure and trigger cascading blasts.",
                symbol: "burst.fill",
                colors: [Color.red.opacity(0.95), Color.orange.opacity(0.85)]
            ),
            ModeCardMeta(
                mode: .hexfection,
                title: "Hexfection",
                subtitle: "Spread and Convert",
                detail: "Expand influence and outgrow your opponent.",
                symbol: "waveform.path.ecg.rectangle.fill",
                colors: [Color.purple.opacity(0.92), Color.indigo.opacity(0.85)]
            )
        ]
    }

    private func bundleBuildStamp() -> String {
        guard let executableURL = Bundle.main.executableURL,
              let values = try? executableURL.resourceValues(forKeys: [.contentModificationDateKey]),
              let date = values.contentModificationDate else {
            return "Stamp ?"
        }
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone.current
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        return "Stamp \(formatter.string(from: date))"
    }

    private var smallBoardLabel: String {
        gameMode == .hexpand ? "3 per side" : "4 per side"
    }

    private var largeBoardLabel: String {
        gameMode == .hexpand ? "4 per side" : "6 per side"
    }

    @ViewBuilder
    private func optionButton(
        _ title: String,
        selected: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 14, weight: .bold, design: .rounded))
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
                .frame(maxWidth: .infinity, minHeight: 38)
                .background(selected ? Color.white.opacity(0.18) : Color.white.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(selected ? Color.yellow.opacity(0.92) : Color.white.opacity(0.24), lineWidth: selected ? 2 : 1)
                )
                .clipShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(PressableScaleButtonStyle())
    }

    @ViewBuilder
    private func sectionCard<Content: View>(
        title: String,
        subtitle: String? = nil,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                Text(title)
                    .font(.system(size: 15, weight: .heavy, design: .rounded))
                    .foregroundStyle(.white)
                Spacer()
                if let subtitle {
                    Text(subtitle)
                        .font(.system(size: 11, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white.opacity(0.66))
                        .lineLimit(1)
                        .minimumScaleFactor(0.72)
                }
            }
            content()
        }
        .padding(12)
        .background(Color.black.opacity(0.28))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color.white.opacity(0.18), lineWidth: 1)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color.black.opacity(0.78), lineWidth: 2)
        )
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    @ViewBuilder
    private func modeCard(
        _ meta: ModeCardMeta,
        selected: Bool,
        compact: Bool = false,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 10) {
                RoundedRectangle(cornerRadius: 12)
                    .fill(
                        LinearGradient(
                            colors: [
                                meta.colors[0].opacity(selected ? 0.92 : 0.72),
                                meta.colors[1].opacity(selected ? 0.88 : 0.66)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(height: compact ? 54 : 86)
                    .overlay {
                        Image(systemName: meta.symbol)
                            .font(.system(size: compact ? 24 : 36, weight: .black))
                            .foregroundStyle(.white.opacity(0.95))
                    }

                Text(meta.title)
                    .font(.system(size: compact ? 19 : 22, weight: .heavy, design: .rounded))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                Text(meta.subtitle)
                    .font(.system(size: compact ? 11 : 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white.opacity(0.86))
                Text(meta.detail)
                    .font(.system(size: compact ? 10 : 12, weight: .regular, design: .rounded))
                    .foregroundStyle(.white.opacity(0.74))
                    .lineLimit(compact ? 3 : 2)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(compact ? 10 : 12)
            .background(Color.black.opacity(selected ? 0.38 : 0.26))
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .stroke(selected ? Color.yellow.opacity(0.92) : Color.white.opacity(0.24), lineWidth: selected ? 2.2 : 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .shadow(color: selected ? Color.yellow.opacity(0.2) : .clear, radius: 8, x: 0, y: 4)
        }
        .buttonStyle(PressableScaleButtonStyle())
    }

    private var startButton: some View {
        Button {
            onToggle()
            withAnimation(.easeInOut(duration: 0.11)) {
                startScale = 0.94
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.11) {
                withAnimation(.easeInOut(duration: 0.13)) {
                    startScale = 1.0
                }
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.28) {
                onStart()
            }
        } label: {
            Image("starthex")
                .resizable()
                .scaledToFit()
                .frame(width: 182)
                .scaleEffect(startScale)
                .shadow(color: .black.opacity(0.45), radius: 8, x: 0, y: 4)
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private func playerRow(
        title: String,
        color: Color,
        isAI: Bool,
        onHuman: @escaping () -> Void,
        onAI: @escaping () -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 6) {
                Circle()
                    .fill(color)
                    .frame(width: 8, height: 8)
                Text(title)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white.opacity(0.9))
            }
            HStack(spacing: 10) {
                optionButton("Human", selected: !isAI, action: onHuman)
                optionButton("AI", selected: isAI, action: onAI)
            }
        }
    }

    @ViewBuilder
    private var boardSizeSection: some View {
        sectionCard(title: "Board Size") {
            HStack(spacing: 10) {
                optionButton("Small  \(smallBoardLabel)", selected: boardSizeOption == .small) {
                    guard boardSizeOption != .small else { return }
                    boardSizeOption = .small
                    onToggle()
                }
                optionButton("Large  \(largeBoardLabel)", selected: boardSizeOption == .large) {
                    guard boardSizeOption != .large else { return }
                    boardSizeOption = .large
                    onToggle()
                }
            }
        }
    }

    @ViewBuilder
    private var playersSection: some View {
        sectionCard(title: "Players", subtitle: "Set Red and Blue independently") {
            VStack(spacing: 10) {
                playerRow(title: "Red Player", color: .red, isAI: redAI) {
                    guard redAI else { return }
                    redAI = false
                    onToggle()
                } onAI: {
                    guard !redAI else { return }
                    redAI = true
                    onToggle()
                }

                playerRow(title: "Blue Player", color: .blue, isAI: blueAI) {
                    guard blueAI else { return }
                    blueAI = false
                    onToggle()
                } onAI: {
                    guard !blueAI else { return }
                    blueAI = true
                    onToggle()
                }
            }
        }
    }

    @ViewBuilder
    private var difficultySection: some View {
        if redAI || blueAI {
            sectionCard(title: "AI Difficulty") {
                HStack(spacing: 10) {
                    optionButton("Easy", selected: aiDifficulty == .easy) {
                        guard aiDifficulty != .easy else { return }
                        aiDifficulty = .easy
                        onToggle()
                    }
                    optionButton("Hard", selected: aiDifficulty == .hard) {
                        guard aiDifficulty != .hard else { return }
                        aiDifficulty = .hard
                        onToggle()
                    }
                }
            }
        } else {
            sectionCard(title: "AI Difficulty") {
                Text("Set one or both players to AI to choose difficulty.")
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.72))
            }
        }
    }

    var body: some View {
        ZStack {
            TitleHexBoardBackground(
                offsetX: 260,
                offsetY: 100,
                zoom: 2.0,
                tilt: 1.0,
                rotationSpeed: 0.1,
                baseAngleDegrees: 0
            )
                .ignoresSafeArea()

            LinearGradient(
                colors: [
                    Color.black.opacity(0.18),
                    Color.black.opacity(0.52)
                ],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            GeometryReader { proxy in
                let isLandscape = proxy.size.width > proxy.size.height || verticalSizeClass == .compact
                let safeInsets = proxy.safeAreaInsets
                let horizontalMargin: CGFloat = isLandscape ? 22 : 18
                let availableWidth = max(240, proxy.size.width - (horizontalMargin * 2))
                let contentWidth: CGFloat = min(availableWidth, isLandscape ? 960 : 360)
                let modeSpacing: CGFloat = 12
                let landscapeModeCardWidth = max(150, (contentWidth - (modeSpacing * 2)) / 3)
                let landscapeColumnWidth = max(180, (contentWidth - 12) / 2)
                let logoMaxWidth: CGFloat = isLandscape
                    ? min(contentWidth * 0.42, 260)
                    : min(contentWidth * 0.9, 340)
                let layoutKey = "title-scroll-\(isLandscape ? "landscape" : "portrait")-\(Int(proxy.size.width))x\(Int(proxy.size.height))-\(step == .mode ? "mode" : "options")"

                ScrollView(.vertical, showsIndicators: true) {
                    VStack(spacing: 14) {
                        Color.clear
                            .frame(height: 1)

                        Image(titleLogoName)
                            .resizable()
                            .scaledToFit()
                            .frame(width: logoMaxWidth)
                            .scaleEffect(logoScale)
                            .shadow(color: .black.opacity(0.45), radius: 12, x: 0, y: 6)
                            .allowsHitTesting(false)
                            .frame(width: contentWidth)

                        if step == .mode {
                            VStack(alignment: .leading, spacing: 10) {
                                Text("Choose Your Game")
                                    .font(.system(size: 21, weight: .heavy, design: .rounded))
                                    .foregroundStyle(.white)
                                Text("Select a mode to continue.")
                                    .font(.system(size: 13, weight: .medium, design: .rounded))
                                    .foregroundStyle(.white.opacity(0.78))
                            }
                            .frame(width: contentWidth, alignment: .leading)

                            if isLandscape {
                                HStack(alignment: .top, spacing: modeSpacing) {
                                    ForEach(Array(modeCards.enumerated()), id: \.offset) { _, card in
                                        modeCard(card, selected: gameMode == card.mode, compact: true) {
                                            if gameMode != card.mode {
                                                gameMode = card.mode
                                                onToggle()
                                            }
                                            withAnimation(.easeInOut(duration: 0.22)) {
                                                step = .options
                                            }
                                        }
                                        .frame(width: landscapeModeCardWidth, alignment: .top)
                                    }
                                }
                                .frame(width: contentWidth, alignment: .leading)
                            } else {
                                VStack(spacing: 12) {
                                    ForEach(Array(modeCards.enumerated()), id: \.offset) { _, card in
                                        modeCard(card, selected: gameMode == card.mode) {
                                            if gameMode != card.mode {
                                                gameMode = card.mode
                                                onToggle()
                                            }
                                            withAnimation(.easeInOut(duration: 0.22)) {
                                                step = .options
                                            }
                                        }
                                    }
                                }
                                .frame(width: contentWidth)
                            }
                        } else {
                            HStack {
                                Button {
                                    withAnimation(.easeInOut(duration: 0.22)) {
                                        step = .mode
                                    }
                                } label: {
                                    HStack(spacing: 4) {
                                        Image(systemName: "chevron.left")
                                        Text("Modes")
                                    }
                                    .font(.system(size: 13, weight: .bold, design: .rounded))
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .background(Color.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
                                }
                                .buttonStyle(.plain)

                                Spacer()

                                Text("Game Setup")
                                    .font(.system(size: 20, weight: .heavy, design: .rounded))
                                    .foregroundStyle(.white)
                            }
                            .frame(width: contentWidth)

                            if isLandscape {
                                HStack(alignment: .top, spacing: 12) {
                                    VStack(spacing: 10) {
                                        boardSizeSection
                                        difficultySection
                                    }
                                    .frame(width: landscapeColumnWidth, alignment: .top)

                                    VStack(spacing: 10) {
                                        playersSection
                                        startButton
                                            .padding(.top, 2)
                                    }
                                    .frame(width: landscapeColumnWidth, alignment: .top)
                                }
                                .frame(width: contentWidth)
                            } else {
                                VStack(spacing: 10) {
                                    boardSizeSection
                                    playersSection
                                    difficultySection
                                    startButton
                                        .padding(.top, 2)
                                }
                                .frame(width: contentWidth)
                            }
                        }

                        Color.clear.frame(height: 8)
                    }
                    .padding(.top, safeInsets.top + 12)
                    .padding(.bottom, safeInsets.bottom + 14)
                    .padding(.horizontal, horizontalMargin)
                    .frame(maxWidth: .infinity, alignment: .top)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                .scrollBounceBehavior(.always)
                .id(layoutKey)
            }
        }
        .overlay(alignment: .top) {
            Text(buildLabel)
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .lineLimit(1)
                .minimumScaleFactor(0.72)
                .foregroundStyle(.white)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.white.opacity(0.35), lineWidth: 1)
                )
                .padding(.horizontal, 16)
                .padding(.top, 16)
                .allowsHitTesting(false)
        }
        .overlay(alignment: .bottom) {
            Text("UI R43")
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .foregroundStyle(.white.opacity(0.9))
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(Color.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 8))
                .padding(.bottom, 6)
                .allowsHitTesting(false)
        }
        .onAppear {
            step = .mode
            logoScale = 0.2
            withAnimation(.interpolatingSpring(mass: 0.8, stiffness: 72, damping: 5.5, initialVelocity: 0)) {
                logoScale = 1.0
            }
        }
    }
}

private struct TitleHexBoardBackground: View {
    let offsetX: CGFloat
    let offsetY: CGFloat
    let zoom: CGFloat
    let tilt: CGFloat
    let rotationSpeed: CGFloat
    let baseAngleDegrees: CGFloat

    private static let sideLength = 8
    private static let boardRadius = sideLength - 1
    private static let tileScale: CGFloat = 0.94
    private static let unitCenters: [CGPoint] = {
        let radius = boardRadius
        var points: [CGPoint] = []
        for q in -radius...radius {
            for r in -radius...radius {
                let s = -q - r
                guard abs(s) <= radius else { continue }
                let x = sqrt(3.0) * (Double(q) + (Double(r) * 0.5))
                let y = 1.5 * Double(r)
                points.append(CGPoint(x: x, y: y))
            }
        }
        return points
    }()
    private static let unitBounds: CGRect = {
        var minX = CGFloat.greatestFiniteMagnitude
        var minY = CGFloat.greatestFiniteMagnitude
        var maxX = -CGFloat.greatestFiniteMagnitude
        var maxY = -CGFloat.greatestFiniteMagnitude
        for p in unitCenters {
            minX = min(minX, p.x - 1)
            minY = min(minY, p.y - 1)
            maxX = max(maxX, p.x + 1)
            maxY = max(maxY, p.y + 1)
        }
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }()
    private static let unitHexPath: Path = {
        var path = Path()
        for index in 0..<6 {
            let angle = Double.pi / 180.0 * (Double(index) * 60.0 - 30.0)
            let point = CGPoint(x: cos(angle), y: sin(angle))
            if index == 0 {
                path.move(to: point)
            } else {
                path.addLine(to: point)
            }
        }
        path.closeSubpath()
        return path
    }()

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: false)) { timeline in
            Canvas { context, size in
                let screenRect = CGRect(origin: .zero, size: size)
                context.fill(
                    Path(screenRect),
                    with: .linearGradient(
                        Gradient(colors: [
                            Color(red: 0.07, green: 0.07, blue: 0.08),
                            Color(red: 0.03, green: 0.03, blue: 0.035)
                        ]),
                        startPoint: CGPoint(x: size.width * 0.25, y: 0),
                        endPoint: CGPoint(x: size.width * 0.75, y: size.height)
                    )
                )

                let seconds = timeline.date.timeIntervalSinceReferenceDate
                let angle = CGFloat((baseAngleDegrees * .pi / 180.0) + (seconds * Double(rotationSpeed)))
                let boardBounds = Self.unitBounds
                let boardMax = max(boardBounds.width, boardBounds.height)
                let zoomScale = (max(size.width, size.height) / boardMax) * zoom

                var boardTransform = CGAffineTransform.identity
                boardTransform = boardTransform.translatedBy(
                    x: size.width * 0.5 + offsetX,
                    y: size.height * 0.55 + offsetY
                )
                boardTransform = boardTransform.rotated(by: angle)
                boardTransform = boardTransform.scaledBy(x: zoomScale, y: zoomScale * tilt)
                boardTransform = boardTransform.translatedBy(x: -boardBounds.midX, y: -boardBounds.midY)

                for center in Self.unitCenters {
                    let toneKey = Int((center.x * 10).rounded()) ^ (Int((center.y * 7).rounded()) << 1)
                    let highlight = (toneKey & 1) == 0 ? 0.58 : 0.5
                    let shadow = (toneKey & 1) == 0 ? 0.22 : 0.18

                    let centerTransform = CGAffineTransform(translationX: center.x, y: center.y)
                    let hexPath = Self.unitHexPath
                        .applying(CGAffineTransform(scaleX: Self.tileScale, y: Self.tileScale))
                        .applying(centerTransform)
                        .applying(boardTransform)
                    let rect = hexPath.boundingRect

                    context.fill(
                        hexPath,
                        with: .linearGradient(
                            Gradient(colors: [
                                Color(white: highlight).opacity(0.7),
                                Color(white: shadow).opacity(0.8)
                            ]),
                            startPoint: CGPoint(x: rect.minX, y: rect.minY),
                            endPoint: CGPoint(x: rect.maxX, y: rect.maxY)
                        )
                    )
                    context.stroke(hexPath, with: .color(Color.black.opacity(0.38)), lineWidth: 0.8)
                }

                context.fill(
                    Path(screenRect),
                    with: .radialGradient(
                        Gradient(colors: [Color.clear, Color.black.opacity(0.45)]),
                        center: CGPoint(x: size.width * 0.5, y: size.height * 0.5),
                        startRadius: min(size.width, size.height) * 0.2,
                        endRadius: max(size.width, size.height) * 0.85
                    )
                )
            }
        }
    }
}

private final class SoundPlayer: ObservableObject {
    private var player: AVAudioPlayer?

    func playFlip() {
        if player == nil {
            guard let url = Bundle.main.url(forResource: "flip", withExtension: "wav") else { return }
            player = try? AVAudioPlayer(contentsOf: url)
            player?.prepareToPlay()
        }
        player?.play()
    }
}
private struct PlayerControlRow: View {
    let title: String
    let titleColor: Color
    let isAI: Bool
    let compact: Bool
    let rowLabelWidth: CGFloat
    let onHumanTap: () -> Void
    let onComputerTap: () -> Void

    var body: some View {
        Group {
            if compact {
                VStack(spacing: 6) {
                    label
                        .frame(maxWidth: .infinity, alignment: .leading)

                    options
                }
            } else {
                HStack(spacing: 10) {
                    label
                        .frame(width: rowLabelWidth, alignment: .leading)

                    options
                }
            }
        }
    }

    private var label: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(titleColor)
                .frame(width: 9, height: 9)
            Text(title)
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .foregroundStyle(.white)
        }
    }

    private var options: some View {
        HStack(spacing: 10) {
            ModeButton(title: "Human", isSelected: !isAI, height: 32, onTap: onHumanTap)
                .frame(maxWidth: .infinity)
            ModeButton(title: "Computer", isSelected: isAI, height: 32, onTap: onComputerTap)
                .frame(maxWidth: .infinity)
        }
        .frame(maxWidth: .infinity)
    }
}
private struct SetupOptionRow<Content: View>: View {
    let title: String
    let compact: Bool
    let rowLabelWidth: CGFloat
    @ViewBuilder let content: Content

    var body: some View {
        Group {
            if compact {
                VStack(spacing: 6) {
                    label
                        .frame(maxWidth: .infinity, alignment: .leading)
                    content
                        .frame(maxWidth: .infinity)
                }
            } else {
                HStack(alignment: .center, spacing: 10) {
                    label
                        .frame(width: rowLabelWidth, alignment: .leading)
                    content
                        .frame(maxWidth: .infinity)
                }
            }
        }
    }

    private var label: some View {
        Text(title)
            .font(.system(size: 12, weight: .semibold, design: .rounded))
            .foregroundStyle(.white.opacity(0.72))
    }
}
private enum BoardSizeOption {
    case small
    case large
}

private struct BoardSizeButton: View {
    let title: String
    let subtitle: String
    let isSelected: Bool
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 4) {
                Text(title)
                    .font(.system(size: 14, weight: .heavy, design: .rounded))
                Text(subtitle)
                    .font(.system(size: 10, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.82))
                    .lineLimit(1)
                    .minimumScaleFactor(0.85)
            }
            .foregroundStyle(.white)
            .frame(height: 34)
            .frame(maxWidth: .infinity)
            .background(Color.white.opacity(0.08))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(isSelected ? Color.yellow.opacity(0.9) : Color.black.opacity(0.8), lineWidth: 2)
            )
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(PressableScaleButtonStyle())
    }
}

private struct ModeButton: View {
    let title: String
    let isSelected: Bool
    let height: CGFloat
    let onTap: () -> Void

    init(
        title: String,
        isSelected: Bool,
        height: CGFloat = 36,
        onTap: @escaping () -> Void
    ) {
        self.title = title
        self.isSelected = isSelected
        self.height = height
        self.onTap = onTap
    }

    var body: some View {
        Button(action: onTap) {
            Text(title)
                .font(.system(size: 14, weight: .heavy, design: .rounded))
                .lineLimit(1)
                .minimumScaleFactor(0.78)
                .foregroundStyle(.white)
                .padding(.horizontal, 8)
                .frame(height: height)
                .frame(maxWidth: .infinity)
                .background(Color.white.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(isSelected ? Color.yellow.opacity(0.9) : Color.black.opacity(0.8), lineWidth: 2)
                )
                .clipShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(PressableScaleButtonStyle())
    }
}

private struct PressableScaleButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.96 : 1.0)
            .animation(.easeInOut(duration: 0.14), value: configuration.isPressed)
    }
}
