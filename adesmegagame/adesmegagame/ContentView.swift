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
            showTitle = true
            resetToken = UUID()
        }
        .onTapGesture {
            guard gameEnded else { return }
            showTitle = true
            resetToken = UUID()
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
            controller.stopGameOverAnimation()
        }
    }
}

#Preview {
    ContentView()
}
private struct TitleScreen: View {
    @Binding var redAI: Bool
    @Binding var blueAI: Bool
    @Binding var boardSizeOption: BoardSizeOption
    @Binding var gameMode: GameMode
    @Binding var aiDifficulty: AIDifficulty
    let onToggle: () -> Void
    let onStart: () -> Void
    @State private var logoScale: CGFloat = 0.1
    @State private var startScale: CGFloat = 1.0
    private var titleLogoName: String {
        UIImage(named: "hex2") != nil ? "hex2" : "hex"
    }
    private var buildLabel: String {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"
        let buildStamp = bundleBuildStamp()
        return "Version \(version) (\(build))  \(buildStamp)"
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

    var body: some View {
        GeometryReader { proxy in
            let contentWidth = min(max(proxy.size.width - 40, 300), 560)
            let compactRows = contentWidth < 430
            let rowLabelWidth: CGFloat = 110
            let rowControlsWidth = compactRows
                ? contentWidth
                : max(220, contentWidth - rowLabelWidth - 10)
            let buttonWidth = (rowControlsWidth - 10) * 0.5
            let gameTypeButtonWidth = (rowControlsWidth - 20) / 3
            let logoWidth = min(contentWidth, 440)

            ZStack {
                LinearGradient(
                    colors: [
                        Color(red: 0.24, green: 0.04, blue: 0.08),
                        Color(red: 0.04, green: 0.10, blue: 0.25)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
                    .ignoresSafeArea()

                ScrollView(showsIndicators: false) {
                    VStack(spacing: 14) {
                        Image(titleLogoName)
                            .resizable()
                            .scaledToFit()
                            .frame(width: logoWidth)
                            .scaleEffect(logoScale)
                            .shadow(color: .black.opacity(0.4), radius: 12, x: 0, y: 6)
                            .onAppear {
                                logoScale = 0.2
                                withAnimation(.interpolatingSpring(mass: 0.8, stiffness: 70, damping: 5, initialVelocity: 0)) {
                                    logoScale = 1.0
                                }
                            }
                            .allowsHitTesting(false)

                        VStack(spacing: 10) {
                            VStack(spacing: 8) {
                                Text("Players")
                                    .font(.system(size: 13, weight: .bold, design: .rounded))
                                    .foregroundStyle(.white.opacity(0.9))
                                    .frame(maxWidth: .infinity, alignment: .leading)

                                PlayerControlRow(
                                    title: "Red Player",
                                    titleColor: .red,
                                    isAI: redAI,
                                    compact: compactRows,
                                    rowLabelWidth: rowLabelWidth,
                                    buttonWidth: buttonWidth,
                                    onHumanTap: {
                                        guard redAI else { return }
                                        redAI = false
                                        onToggle()
                                    },
                                    onComputerTap: {
                                        guard !redAI else { return }
                                        redAI = true
                                        onToggle()
                                    }
                                )

                                PlayerControlRow(
                                    title: "Blue Player",
                                    titleColor: .blue,
                                    isAI: blueAI,
                                    compact: compactRows,
                                    rowLabelWidth: rowLabelWidth,
                                    buttonWidth: buttonWidth,
                                    onHumanTap: {
                                        guard blueAI else { return }
                                        blueAI = false
                                        onToggle()
                                    },
                                    onComputerTap: {
                                        guard !blueAI else { return }
                                        blueAI = true
                                        onToggle()
                                    }
                                )
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(Color.white.opacity(0.08))
                            .overlay(
                                RoundedRectangle(cornerRadius: 14)
                                    .stroke(Color.black.opacity(0.8), lineWidth: 2)
                            )
                            .clipShape(RoundedRectangle(cornerRadius: 14))

                            VStack(spacing: 8) {
                                Text("Game Setup")
                                    .font(.system(size: 13, weight: .bold, design: .rounded))
                                    .foregroundStyle(.white.opacity(0.9))
                                    .frame(maxWidth: .infinity, alignment: .leading)

                                SetupOptionRow(title: "Game Type", compact: compactRows, rowLabelWidth: rowLabelWidth) {
                                    HStack(spacing: 10) {
                                        ModeButton(title: "Hexello", isSelected: gameMode == .hexello, width: gameTypeButtonWidth, height: 32) {
                                            gameMode = .hexello
                                            onToggle()
                                        }
                                        ModeButton(title: "Hexplode", isSelected: gameMode == .hexpand, width: gameTypeButtonWidth, height: 32) {
                                            gameMode = .hexpand
                                            onToggle()
                                        }
                                        ModeButton(title: "Hexfection", isSelected: gameMode == .hexfection, width: gameTypeButtonWidth, height: 32) {
                                            gameMode = .hexfection
                                            onToggle()
                                        }
                                    }
                                }

                                SetupOptionRow(title: "Difficulty", compact: compactRows, rowLabelWidth: rowLabelWidth) {
                                    HStack(spacing: 10) {
                                        ModeButton(title: "Easy", isSelected: aiDifficulty == .easy, width: buttonWidth, height: 32) {
                                            aiDifficulty = .easy
                                            onToggle()
                                        }
                                        ModeButton(title: "Hard", isSelected: aiDifficulty == .hard, width: buttonWidth, height: 32) {
                                            aiDifficulty = .hard
                                            onToggle()
                                        }
                                    }
                                }

                                let smallSubtitle = gameMode == .hexpand ? "3 per side" : "4 per side"
                                let largeSubtitle = gameMode == .hexpand ? "4 per side" : "6 per side"

                                SetupOptionRow(title: "Board Size", compact: compactRows, rowLabelWidth: rowLabelWidth) {
                                    HStack(spacing: 10) {
                                        BoardSizeButton(
                                            title: "Small",
                                            subtitle: smallSubtitle,
                                            isSelected: boardSizeOption == .small,
                                            width: buttonWidth
                                        ) {
                                            boardSizeOption = .small
                                            onToggle()
                                        }

                                        BoardSizeButton(
                                            title: "Large",
                                            subtitle: largeSubtitle,
                                            isSelected: boardSizeOption == .large,
                                            width: buttonWidth
                                        ) {
                                            boardSizeOption = .large
                                            onToggle()
                                        }
                                    }
                                }

                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(Color.white.opacity(0.08))
                            .overlay(
                                RoundedRectangle(cornerRadius: 14)
                                    .stroke(Color.black.opacity(0.8), lineWidth: 2)
                            )
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                        }
                        .frame(width: contentWidth)

                        Button {
                            startScale = 0.9
                            withAnimation(.easeInOut(duration: 0.25)) {
                                startScale = 0.9
                            }
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                                withAnimation(.easeInOut(duration: 0.25)) {
                                    startScale = 1.0
                                }
                            }
                            onToggle()
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                onStart()
                            }
                        } label: {
                            Image("starthex")
                                .resizable()
                                .scaledToFit()
                                .frame(width: 150)
                                .scaleEffect(startScale)
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 18)
                    .frame(maxWidth: .infinity)
                }
            }
        }
        .overlay(alignment: .topTrailing) {
            Text(buildLabel)
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .foregroundStyle(.white)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.white.opacity(0.35), lineWidth: 1)
                )
                .padding(.top, 16)
                .padding(.trailing, 16)
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
    let buttonWidth: CGFloat
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
            ModeButton(title: "Human", isSelected: !isAI, width: buttonWidth, height: 32, onTap: onHumanTap)
            ModeButton(title: "Computer", isSelected: isAI, width: buttonWidth, height: 32, onTap: onComputerTap)
        }
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
                }
            } else {
                HStack(alignment: .center, spacing: 10) {
                    label
                        .frame(width: rowLabelWidth, alignment: .leading)
                    content
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
    let width: CGFloat
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
            .frame(width: width, height: 34)
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
    let width: CGFloat
    let height: CGFloat
    let onTap: () -> Void

    init(
        title: String,
        isSelected: Bool,
        width: CGFloat = 140,
        height: CGFloat = 36,
        onTap: @escaping () -> Void
    ) {
        self.title = title
        self.isSelected = isSelected
        self.width = width
        self.height = height
        self.onTap = onTap
    }

    var body: some View {
        Button(action: onTap) {
            Text(title)
                .font(.system(size: 14, weight: .heavy, design: .rounded))
                .foregroundStyle(.white)
                .frame(width: width, height: height)
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
