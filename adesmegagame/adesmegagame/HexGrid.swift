import Foundation
import SceneKit

// Pointy-top axial coordinates on the XZ plane (Y is up).
struct AxialCoord: Hashable {
    let q: Int
    let r: Int

    func adding(_ other: AxialCoord) -> AxialCoord {
        AxialCoord(q: q + other.q, r: r + other.r)
    }
}

enum HexGrid {
    static let directions: [AxialCoord] = [
        AxialCoord(q: 1, r: 0),
        AxialCoord(q: 1, r: -1),
        AxialCoord(q: 0, r: -1),
        AxialCoord(q: -1, r: 0),
        AxialCoord(q: -1, r: 1),
        AxialCoord(q: 0, r: 1)
    ]

    static func generateHexagon(radius: Int) -> [AxialCoord] {
        var coords: [AxialCoord] = []
        for q in -radius...radius {
            let rMin = max(-radius, -q - radius)
            let rMax = min(radius, -q + radius)
            for r in rMin...rMax {
                coords.append(AxialCoord(q: q, r: r))
            }
        }
        return coords
    }

    static func ringCoordinates(radius: Int) -> [AxialCoord] {
        guard radius > 0 else { return [] }
        var results: [AxialCoord] = []
        var coord = AxialCoord(q: directions[4].q * radius, r: directions[4].r * radius)
        for direction in directions {
            for _ in 0..<radius {
                results.append(coord)
                coord = coord.adding(direction)
            }
        }
        return results
    }

    // Pointy-top axial to world mapping on XZ plane.
    // x = size * sqrt(3) * (q + r/2)
    // z = size * 3/2 * r
    static func axialToWorld(q: Int, r: Int, tileSize: Float) -> SCNVector3 {
        let x = tileSize * Float(sqrt(3.0)) * (Float(q) + Float(r) * 0.5)
        let z = tileSize * 1.5 * Float(r)
        return SCNVector3(x, 0, z)
    }
}
