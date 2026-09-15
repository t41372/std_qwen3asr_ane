// Replay real decoder inputs with Core ML-owned arrays and native array chaining.
// Build with Swift 6 and a macOS 15+ target. No pointers escape scoped access.
import CoreML
import CryptoKit
import Foundation

enum BenchError: LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case let .invalid(message): message
        }
    }
}

func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func containedURL(_ relative: String, in root: URL) throws -> URL {
    let url = root.appendingPathComponent(relative).resolvingSymlinksInPath()
    guard !relative.hasPrefix("/"), url.path.hasPrefix(root.path + "/") else {
        throw BenchError.invalid("Fixture or model path escapes its root: \(relative)")
    }
    return url
}

func elementCount(_ shape: [Int]) throws -> Int {
    guard !shape.isEmpty else { throw BenchError.invalid("Empty tensor shape") }
    return try shape.reduce(1) { count, dimension in
        let (product, overflow) = count.multipliedReportingOverflow(by: dimension)
        guard dimension > 0, !overflow, product <= Int.max / 2 else {
            throw BenchError.invalid("Invalid tensor dimensions")
        }
        return product
    }
}

func contiguous(_ shape: [Int], _ strides: [Int]) -> Bool {
    guard shape.count == strides.count else { return false }
    var expected = 1
    for axis in shape.indices.reversed() {
        if shape[axis] > 1, strides[axis] != expected { return false }
        expected *= shape[axis]
    }
    return true
}

func offset(_ index: Int, shape: [Int], strides: [Int]) -> Int {
    var remaining = index
    var result = 0
    for axis in shape.indices.reversed() {
        result += (remaining % shape[axis]) * strides[axis]
        remaining /= shape[axis]
    }
    return result
}

struct TensorFile: Decodable {
    let file: String
    let shape: [Int]
    let dtype: String
    let sha256: String

    func load(from root: URL) throws -> TensorData {
        let count = try elementCount(shape)
        let data = try Data(contentsOf: containedURL(file, in: root), options: .mappedIfSafe)
        guard dtype == "float16-le", data.count == count * 2,
              SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == sha256 else {
            throw BenchError.invalid("Tensor type, size or hash mismatch: \(file)")
        }
        return TensorData(shape: shape, bytes: data, count: count)
    }
}

struct TensorData {
    let shape: [Int]
    let bytes: Data
    let count: Int

    func makeArray() throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        try write(to: array)
        return array
    }

    func write(to array: MLMultiArray) throws {
        guard array.dataType == .float16, array.shape.map(\.intValue) == shape else {
            throw BenchError.invalid("State or input tensor has a different shape/type")
        }
        try array.withUnsafeMutableBytes { destination, strides in
            // Mutable access can replace the backing store; use these strides,
            // not a value read from the MLMultiArray before entering the closure.
            try bytes.withUnsafeBytes { source in
                if contiguous(shape, strides) {
                    guard destination.count >= source.count else {
                        throw BenchError.invalid("Core ML supplied a short tensor buffer")
                    }
                    destination.copyMemory(from: source)
                } else {
                    for index in 0..<count {
                        let byteOffset = offset(index, shape: shape, strides: strides) * 2
                        guard byteOffset >= 0, byteOffset + 2 <= destination.count else {
                            throw BenchError.invalid("Tensor strides exceed its buffer")
                        }
                        destination.storeBytes(
                            of: source.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self),
                            toByteOffset: byteOffset, as: UInt16.self
                        )
                    }
                }
            }
        }
    }

    func mismatches(with array: MLMultiArray) throws -> Int {
        guard array.dataType == .float16, array.shape.map(\.intValue) == shape else {
            throw BenchError.invalid("Decoder output has a different shape/type")
        }
        return try array.withUnsafeBytes { actual in
            let strides = array.strides.map(\.intValue)
            let isContiguous = contiguous(shape, strides)
            return try bytes.withUnsafeBytes { expected in
                var differences = 0
                for index in 0..<count {
                    let byteOffset = (isContiguous ? index : offset(index, shape: shape, strides: strides)) * 2
                    guard byteOffset >= 0, byteOffset + 2 <= actual.count else {
                        throw BenchError.invalid("Output strides exceed its buffer")
                    }
                    let bits = actual.loadUnaligned(fromByteOffset: byteOffset, as: UInt16.self)
                    guard bits & 0x7c00 != 0x7c00 else {
                        throw BenchError.invalid("Decoder produced a non-finite output")
                    }
                    let reference = expected.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                    // Match NumPy's exact value comparison, including signed zero.
                    if bits != reference, !(bits & 0x7fff == 0 && reference & 0x7fff == 0) {
                        differences += 1
                    }
                }
                return differences
            }
        }
    }
}

struct StepFile: Decodable {
    let inputs: [String: TensorFile]
    let expected: TensorFile
}

struct SegmentFile: Decodable {
    let states: [[String: TensorFile]]
    let steps: [StepFile]
}

struct Fixture: Decodable {
    let schemaVersion: Int
    let complete: Bool
    let stepCount: Int
    let modelManifestSHA256: String
    let segments: [SegmentFile]

    enum CodingKeys: String, CodingKey {
        case complete, segments
        case schemaVersion = "schema_version"
        case stepCount = "step_count"
        case modelManifestSHA256 = "model_manifest_sha256"
    }
}

struct BundleManifest: Decodable {
    let decoderPartitions: [String]
    enum CodingKeys: String, CodingKey { case decoderPartitions = "decoder_partitions" }
}

// These objects own immutable, preloaded Core ML input arrays for repeated replay.
final class ReplayStep {
    let features: [String: MLFeatureValue]
    let expected: TensorData

    init(_ file: StepFile, root: URL) throws {
        features = try file.inputs.mapValues { MLFeatureValue(multiArray: try $0.load(from: root).makeArray()) }
        expected = try file.expected.load(from: root)
    }
}

struct ReplaySegment {
    let states: [[String: TensorData]]
    let steps: [ReplayStep]

    init(_ file: SegmentFile, root: URL) throws {
        states = try file.states.map { try $0.mapValues { try $0.load(from: root) } }
        steps = try file.steps.map { try ReplayStep($0, root: root) }
    }
}

struct Report: Encodable {
    let runtime = "swift"
    let steps: Int
    let fixtureSHA256: String
    let timingScope = "decoder predictions and native MLMultiArray chaining; excludes fixture loading, KV restoration, output verification and LM head"
    var complete = false
    var exactHiddenParity = false
    var mismatchedElements = 0
    var seconds: [Double] = []
    var medianSeconds: Double?
    var error: String?
}

@main
enum DecoderBench {
    static func main() {
        do { try run() }
        catch {
            FileHandle.standardError.write(Data(("Decoder benchmark failed: \(error.localizedDescription)\n").utf8))
            exit(1)
        }
    }

    static func run() throws {
        let arguments = CommandLine.arguments
        guard arguments.count == 4 else {
            throw BenchError.invalid("Usage: DecoderBench MODEL_DIR FIXTURE_DIR OUTPUT_JSON")
        }
        let modelRoot = URL(fileURLWithPath: arguments[1], isDirectory: true).resolvingSymlinksInPath()
        let fixtureRoot = URL(fileURLWithPath: arguments[2], isDirectory: true).resolvingSymlinksInPath()
        let outputURL = URL(fileURLWithPath: arguments[3])
        guard !FileManager.default.fileExists(atPath: outputURL.path) else {
            throw BenchError.invalid("Use a fresh output path")
        }
        let manifestData = try Data(contentsOf: modelRoot.appendingPathComponent("manifest.json"))
        let fixtureData = try Data(contentsOf: fixtureRoot.appendingPathComponent("fixture.json"))
        let decoder = JSONDecoder()
        let fixture = try decoder.decode(Fixture.self, from: fixtureData)
        let bundle = try decoder.decode(BundleManifest.self, from: manifestData)
        guard fixture.schemaVersion == 1, fixture.complete, fixture.stepCount > 0,
              fixture.stepCount == fixture.segments.reduce(0, { $0 + $1.steps.count }),
              fixture.modelManifestSHA256 == sha256(manifestData), !bundle.decoderPartitions.isEmpty else {
            throw BenchError.invalid("Incomplete fixture or different model bundle")
        }
        var report = Report(steps: fixture.stepCount, fixtureSHA256: sha256(fixtureData))
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            let segments = try fixture.segments.map { try ReplaySegment($0, root: fixtureRoot) }
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            let models = try bundle.decoderPartitions.map {
                try MLModel(contentsOf: containedURL($0, in: modelRoot), configuration: configuration)
            }
            let clock = ContinuousClock()
            for repetition in -1..<5 {
                var elapsed = 0.0
                for segment in segments {
                    guard segment.states.count == models.count else {
                        throw BenchError.invalid("Fixture state partition count differs")
                    }
                    let states = models.map { $0.makeState() }
                    for index in models.indices {
                        let expectedNames = Set(models[index].modelDescription.stateDescriptionsByName.keys)
                        guard Set(segment.states[index].keys) == expectedNames else {
                            throw BenchError.invalid("Fixture state names differ from model")
                        }
                        for (name, tensor) in segment.states[index] {
                            try states[index].withMultiArray(for: name) { try tensor.write(to: $0) }
                        }
                    }
                    var results: [MLMultiArray] = []
                    let started = clock.now
                    for step in segment.steps {
                        let hidden: MLMultiArray = try autoreleasepool {
                            var features = step.features
                            for index in models.indices {
                                let input = try MLDictionaryFeatureProvider(dictionary: features)
                                let prediction = try models[index].prediction(from: input, using: states[index])
                                guard let value = prediction.featureValue(for: "output_hidden_states") else {
                                    throw BenchError.invalid("Decoder omitted hidden output")
                                }
                                features["hidden_states"] = value
                            }
                            guard let result = features["hidden_states"]?.multiArrayValue else {
                                throw BenchError.invalid("Decoder output is not an MLMultiArray")
                            }
                            return result
                        }
                        results.append(hidden)
                    }
                    let duration = started.duration(to: clock.now).components
                    elapsed += Double(duration.seconds) + Double(duration.attoseconds) / 1e18
                    for (step, result) in zip(segment.steps, results) {
                        report.mismatchedElements += try step.expected.mismatches(with: result)
                    }
                }
                if repetition >= 0 { report.seconds.append(elapsed) }
            }
            report.medianSeconds = report.seconds.sorted()[report.seconds.count / 2]
            report.exactHiddenParity = report.mismatchedElements == 0
            report.complete = true
        } catch {
            report.error = error.localizedDescription
            try encoder.encode(report).write(to: outputURL, options: .atomic)
            throw error
        }
        try encoder.encode(report).write(to: outputURL, options: .atomic)
        guard report.exactHiddenParity else {
            throw BenchError.invalid("Native replay changed decoder hidden states")
        }
    }
}
