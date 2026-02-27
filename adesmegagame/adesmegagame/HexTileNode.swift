import SceneKit
import UIKit

enum TileState {
    case empty
    case red
    case blue

    var opposite: TileState {
        switch self {
        case .red:
            return .blue
        case .blue:
            return .red
        case .empty:
            return .empty
        }
    }
}

final class HexTileNode: SCNNode {
    private enum TileStyle {
        case hexello
        case hexpand
        case hexfection
    }

    let coord: AxialCoord
    private let material: SCNMaterial
    private let tileSize: CGFloat
    private let baseHeight: Float
    private let selectedDepth: Float = -0.2
    private var heightScale: Float = 1.0
    private var style: TileStyle = .hexello

    init(coord: AxialCoord, size: CGFloat, height: CGFloat) {
        self.coord = coord
        self.tileSize = size
        self.baseHeight = Float(height * 0.5)
        self.material = HexTileNode.makeMaterial()
        super.init()

        let hex = HexTileNode.makeGeometry(size: size, height: height)
        hex.materials = [material]
        geometry = hex

        // Rotate so extrusion height is along Y (up).
        eulerAngles.x = -.pi / 2
        applyHeightScaleImmediate(1.0)
    }

    required init?(coder: NSCoder) {
        return nil
    }

    func setState(_ state: TileState) {
        applyMaterial(for: state)
    }

    func setStyle(for mode: GameMode) {
        switch mode {
        case .hexello:
            style = .hexello
        case .hexpand:
            style = .hexpand
        case .hexfection:
            style = .hexfection
        }
        let chamferScale: Float
        switch style {
        case .hexello:
            chamferScale = 0.12
        case .hexpand:
            chamferScale = 0.18
        case .hexfection:
            chamferScale = 0.14
        }
        let chamfer = (Float(baseHeight) * 2) * chamferScale
        if let shape = geometry as? SCNShape {
            shape.chamferRadius = CGFloat(chamfer)
        }
    }

    func pulseSelection() {
        removeAction(forKey: "height")
        let down = SCNAction.move(
            to: SCNVector3(position.x, baseHeight * heightScale + selectedDepth, position.z),
            duration: 0.05
        )
        down.timingMode = .easeInEaseOut
        let up = SCNAction.move(
            to: SCNVector3(position.x, baseHeight * heightScale, position.z),
            duration: 0.1
        )
        up.timingMode = .easeInEaseOut
        let sequence = SCNAction.sequence([down, up])
        runAction(sequence, forKey: "height")
    }

    func animatePlacement(to state: TileState, completion: @escaping () -> Void) {
        removeAction(forKey: "height")
        setState(.empty)

        let down = SCNAction.move(
            to: SCNVector3(position.x, baseHeight * heightScale + selectedDepth, position.z),
            duration: 0.1
        )
        down.timingMode = .easeInEaseOut

        let startColorFade = SCNAction.run { [weak self] _ in
            guard let self else { return }
            self.animateColor(to: self.paletteColor(for: state), duration: 0.1)
            self.applyMaterialForState(state)
        }

        let up = SCNAction.move(
            to: SCNVector3(position.x, baseHeight * heightScale, position.z),
            duration: 0.5
        )
        up.timingMode = .easeInEaseOut

        let finish = SCNAction.run { _ in
            completion()
        }

        let sequence = SCNAction.sequence([down, startColorFade, up, finish])
        runAction(sequence, forKey: "height")
    }

    func animateFlip(to state: TileState, axis: SCNVector3, duration: TimeInterval, completion: @escaping () -> Void) {
        let targetColor = paletteColor(for: state)
        animateColor(to: targetColor, duration: duration)
        applyMaterialForState(state)
        let rotate = SCNAction.rotate(by: .pi, around: axis, duration: duration)
        rotate.timingMode = .easeInEaseOut
        let finish = SCNAction.run { _ in
            completion()
        }
        runAction(.sequence([rotate, finish]))
    }

    func setHeightScale(_ scale: Float, duration: TimeInterval = 0.2) {
        heightScale = max(1.0, scale)
        removeAction(forKey: "heightScale")
        let targetScale = CGFloat(heightScale)
        let targetY = baseHeight * heightScale
        guard duration > 0 else {
            self.scale = SCNVector3(self.scale.x, self.scale.y, Float(targetScale))
            self.position = SCNVector3(self.position.x, targetY, self.position.z)
            return
        }
        let startScale = CGFloat(self.scale.z)
        let startY = position.y
        let scaleDelta = targetScale - startScale
        let yDelta = CGFloat(targetY - startY)
        let action = SCNAction.customAction(duration: duration) { [weak self] _, elapsed in
            guard let self else { return }
            let t = Float(min(1, elapsed / duration))
            let eased = t * t * (3 - 2 * t)
            let currentScale = startScale + scaleDelta * CGFloat(eased)
            let currentY = startY + Float(yDelta) * eased
            self.scale = SCNVector3(self.scale.x, self.scale.y, Float(currentScale))
            self.position = SCNVector3(self.position.x, currentY, self.position.z)
        }
        runAction(action, forKey: "heightScale")
    }

    func animateState(to state: TileState, duration: TimeInterval) {
        animateColor(to: paletteColor(for: state), duration: duration)
        applyMaterialForState(state)
    }

    func animateTintColor(to color: UIColor, duration: TimeInterval) {
        animateColor(to: color, duration: duration)
    }

    private func applyHeightScaleImmediate(_ scale: Float) {
        heightScale = max(1.0, scale)
        self.scale = SCNVector3(self.scale.x, self.scale.y, heightScale)
        self.position = SCNVector3(self.position.x, baseHeight * heightScale, self.position.z)
    }

    private static func makeGeometry(size: CGFloat, height: CGFloat) -> SCNGeometry {
        let path = UIBezierPath()
        let radius = size
        for index in 0..<6 {
            let angle = CGFloat(index) * .pi / 3 - .pi / 2
            let point = CGPoint(x: cos(angle) * radius, y: sin(angle) * radius)
            if index == 0 {
                path.move(to: point)
            } else {
                path.addLine(to: point)
            }
        }
        path.close()

        let shape = SCNShape(path: path, extrusionDepth: height)
        shape.chamferRadius = height * 0.6
        return shape
    }

    private static func makeMaterial() -> SCNMaterial {
        let material = SCNMaterial()
        material.diffuse.contents = hexelloPalette.emptyColor
        material.emission.contents = UIColor.black
        material.emission.intensity = 0
        material.lightingModel = .physicallyBased
        material.metalness.contents = 0.3
        material.roughness.contents = 0.3
        material.specular.contents = UIColor(white: 0.8, alpha: 1.0)
        material.isDoubleSided = false
        return material
    }


    private func applyMaterialForState(_ state: TileState) {
        let palette = paletteForStyle()
        switch state {
        case .empty:
            material.metalness.contents = palette.emptyMetalness
            material.roughness.contents = palette.emptyRoughness
        case .red, .blue:
            material.metalness.contents = palette.filledMetalness
            material.roughness.contents = palette.filledRoughness
        }
        material.specular.contents = palette.specular
        material.clearCoat.contents = palette.clearCoat
        material.clearCoatRoughness.contents = palette.clearCoatRoughness
    }

    private func applyMaterial(for state: TileState) {
        material.diffuse.contents = paletteColor(for: state)
        applyMaterialForState(state)
    }

    private func animateColor(to target: UIColor, duration: TimeInterval) {
        removeAction(forKey: "colorFade")
        let start = (material.diffuse.contents as? UIColor) ?? paletteColor(for: .empty)
        let action = SCNAction.customAction(duration: duration) { [weak self] _, elapsed in
            guard let self else { return }
            let t = max(0, min(1, elapsed / duration))
            let blended = Self.interpolate(from: start, to: target, t: CGFloat(t))
            self.material.diffuse.contents = blended
        }
        action.timingMode = .easeInEaseOut
        runAction(action, forKey: "colorFade")
    }

    private static func interpolate(from: UIColor, to: UIColor, t: CGFloat) -> UIColor {
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

    private func paletteColor(for state: TileState) -> UIColor {
        let palette = paletteForStyle()
        if style == .hexfection {
            return palette.emptyColor
        }
        switch state {
        case .empty:
            return palette.emptyColor
        case .red:
            return palette.redColor
        case .blue:
            return palette.blueColor
        }
    }

    private func paletteForStyle() -> TilePalette {
        switch style {
        case .hexello:
            return Self.hexelloPalette
        case .hexpand:
            return Self.hexpandPalette
        case .hexfection:
            return Self.hexfectionPalette
        }
    }

    // MARK: - Hexello (Mode 1) Tile Style
    private static let hexelloPalette = TilePalette(
        emptyColor: UIColor(red: 0.2, green: 0.2, blue: 0.25, alpha: 1.0),
        redColor: UIColor(red: 0.8, green: 0.18, blue: 0.2, alpha: 1.0),
        blueColor: UIColor(red: 0.18, green: 0.42, blue: 0.8, alpha: 1.0),
        emptyMetalness: 0.85,
        emptyRoughness: 0.25,
        filledMetalness: 0.85,
        filledRoughness: 0.25,
        specular: UIColor(white: 1.0, alpha: 1.0),
        clearCoat: 0.0,
        clearCoatRoughness: 1.0
    )

    // MARK: - Hexpand (Mode 2) Tile Style
    private static let hexpandPalette = TilePalette(
        emptyColor: UIColor(red: 0.5, green: 0.5, blue: 0.55, alpha: 1.0),
        redColor: UIColor(red: 0.76, green: 0.28, blue: 0.2, alpha: 1.0),
        blueColor: UIColor(red: 0.2, green: 0.3, blue: 0.78, alpha: 1.0),
        emptyMetalness: 0.18,
        emptyRoughness: 0.3,
        filledMetalness: 0.26,
        filledRoughness: 0.35,
        specular: UIColor(white: 1.0, alpha: 1.0),
        clearCoat: 0.6,
        clearCoatRoughness: 0.2
    )

    // MARK: - Hexfection (Mode 3) Tile Style
    private static let hexfectionPalette = TilePalette(
        emptyColor: UIColor(red: 0.56, green: 0.58, blue: 0.62, alpha: 1.0),
        redColor: UIColor(red: 0.56, green: 0.58, blue: 0.62, alpha: 1.0),
        blueColor: UIColor(red: 0.56, green: 0.58, blue: 0.62, alpha: 1.0),
        emptyMetalness: 0.55,
        emptyRoughness: 0.28,
        filledMetalness: 0.55,
        filledRoughness: 0.28,
        specular: UIColor(white: 0.95, alpha: 1.0),
        clearCoat: 0.38,
        clearCoatRoughness: 0.22
    )

    private struct TilePalette {
        let emptyColor: UIColor
        let redColor: UIColor
        let blueColor: UIColor
        let emptyMetalness: CGFloat
        let emptyRoughness: CGFloat
        let filledMetalness: CGFloat
        let filledRoughness: CGFloat
        let specular: UIColor
        let clearCoat: CGFloat
        let clearCoatRoughness: CGFloat
    }

    private struct ColorAdjust {
        static func darker(_ color: UIColor, by amount: CGFloat) -> UIColor {
            var r: CGFloat = 0
            var g: CGFloat = 0
            var b: CGFloat = 0
            var a: CGFloat = 0
            color.getRed(&r, green: &g, blue: &b, alpha: &a)
            return UIColor(
                red: max(0, r - amount),
                green: max(0, g - amount),
                blue: max(0, b - amount),
                alpha: a
            )
        }

        static func lighter(_ color: UIColor, by amount: CGFloat) -> UIColor {
            var r: CGFloat = 0
            var g: CGFloat = 0
            var b: CGFloat = 0
            var a: CGFloat = 0
            color.getRed(&r, green: &g, blue: &b, alpha: &a)
            return UIColor(
                red: min(1, r + amount),
                green: min(1, g + amount),
                blue: min(1, b + amount),
                alpha: a
            )
        }
    }

    static func darker(by amount: CGFloat, color: UIColor) -> UIColor {
        ColorAdjust.darker(color, by: amount)
    }

    static func lighter(by amount: CGFloat, color: UIColor) -> UIColor {
        ColorAdjust.lighter(color, by: amount)
    }

    static func hexpandColor(for state: TileState) -> UIColor {
        switch state {
        case .empty:
            return hexpandPalette.emptyColor
        case .red:
            return hexpandPalette.redColor
        case .blue:
            return hexpandPalette.blueColor
        }
    }

    static func hexfectionPieceColor(for state: TileState) -> UIColor {
        switch state {
        case .red:
            return UIColor(red: 0.88, green: 0.22, blue: 0.2, alpha: 1.0)
        case .blue:
            return UIColor(red: 0.22, green: 0.4, blue: 0.9, alpha: 1.0)
        case .empty:
            return hexfectionPalette.emptyColor
        }
    }

    static func hexfectionTileBaseColor() -> UIColor {
        hexfectionPalette.emptyColor
    }
}
