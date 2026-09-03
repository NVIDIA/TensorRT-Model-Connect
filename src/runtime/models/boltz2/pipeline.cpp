/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/boltz2/pipeline.h"

#include "utils/sha256.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>
#include <unordered_set>
#include <utility>

namespace trtmc::boltz2 {
namespace {

using Vec3 = std::array<float, 3>;
using Mat3 = std::array<std::array<float, 3>, 3>;

constexpr float kSigmaMin = 0.0001F;
constexpr float kSigmaMax = 160.0F;
constexpr float kSigmaData = 16.0F;
constexpr float kRho = 7.0F;
constexpr float kGamma0 = 0.8F;
constexpr float kGammaMin = 1.0F;
constexpr float kNoiseScale = 1.003F;
constexpr float kStepScale = 1.5F;

void validateRequestInput(const std::string& input, const std::string& input_path) {
    if (input_path.empty())
        throw std::invalid_argument(
            "Boltz-2 direct YAML preprocessing requires the request file path");
    std::ifstream request_stream(input_path, std::ios::binary);
    if (!request_stream)
        throw std::invalid_argument("failed to open the Boltz-2 request file");
    const std::string request_on_disk((std::istreambuf_iterator<char>(request_stream)),
                                      std::istreambuf_iterator<char>());
    if (request_on_disk != input)
        throw std::invalid_argument("Boltz-2 request content differs from its request file path");
}

std::string createPreparedRequestPath() {
    auto pattern =
        (std::filesystem::temp_directory_path() / "trtmc-boltz2-request-XXXXXX").string();
    std::vector<char> temporary(pattern.begin(), pattern.end());
    temporary.push_back('\0');
    const int descriptor = mkstemp(temporary.data());
    if (descriptor < 0)
        throw std::runtime_error("failed to create Boltz-2 prepared-request temporary file");
    close(descriptor);
    return std::string(temporary.data());
}

std::vector<char*> mutablePointers(std::vector<std::string>& values) {
    std::vector<char*> pointers;
    pointers.reserve(values.size() + 1);
    for (auto& value : values)
        pointers.push_back(value.data());
    pointers.push_back(nullptr);
    return pointers;
}

std::vector<std::string> requestPreprocessorEnvironment() {
    std::vector<std::string> environment;
    std::string source_pythonpath;
#ifdef TRTMC_SOURCE_DIR
    const auto source_python = std::filesystem::path(TRTMC_SOURCE_DIR) / "python";
    std::error_code source_error;
    if (std::filesystem::is_directory(source_python, source_error))
        source_pythonpath = source_python.string();
#endif
    for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
        const std::string_view value(*entry);
        if (!source_pythonpath.empty() && value.compare(0, 11, "PYTHONPATH=") == 0) {
            source_pythonpath += ":" + std::string(value.substr(11));
        } else {
            environment.emplace_back(*entry);
        }
    }
    if (!source_pythonpath.empty())
        environment.push_back("PYTHONPATH=" + source_pythonpath);
    return environment;
}

void waitForRequestPreprocessor(pid_t child, const std::string& output_path) {
    int status = 0;
    while (true) {
        if (waitpid(child, &status, 0) >= 0)
            break;
        if (errno == EINTR)
            continue;
        std::filesystem::remove(output_path);
        throw std::runtime_error("failed waiting for the Boltz-2 request preprocessor");
    }
    if (!WIFEXITED(status)) {
        std::filesystem::remove(output_path);
        throw std::runtime_error(
            "Boltz-2 request preprocessing failed; see the Python error above");
    }
    if (WEXITSTATUS(status) != 0) {
        std::filesystem::remove(output_path);
        throw std::runtime_error(
            "Boltz-2 request preprocessing failed; see the Python error above");
    }
}

void executeRequestPreprocessor(std::vector<std::string>& arguments,
                                std::vector<std::string>& environment,
                                const std::string& output_path) {
    auto argv = mutablePointers(arguments);
    auto envp = mutablePointers(environment);
    pid_t child = -1;
    const int spawn_status =
        posix_spawnp(&child, argv[0], nullptr, nullptr, argv.data(), envp.data());
    if (spawn_status != 0) {
        std::filesystem::remove(output_path);
        throw std::system_error(spawn_status, std::generic_category(),
                                "failed to start the Boltz-2 request preprocessor");
    }
    waitForRequestPreprocessor(child, output_path);
}

std::string readPreparedRequest(const std::string& output_path) {
    std::ifstream stream(output_path, std::ios::binary);
    const std::string payload((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>());
    stream.close();
    std::filesystem::remove(output_path);
    if (payload.empty())
        throw std::runtime_error("Boltz-2 request preprocessor produced no data");
    return payload;
}

std::string runRequestPreprocessor(const std::string& python, const std::string& input,
                                   const std::string& input_path, int token_count, int atom_count,
                                   int msa_depth) {
    validateRequestInput(input, input_path);
    const std::string output_path = createPreparedRequestPath();
    std::vector<std::string> arguments{
        python.empty() ? "python3" : python,
        "-m",
        "tensorrt_model_connect.families.boltz2.runtime_preprocess",
        "--input",
        input_path,
        "--output",
        output_path,
        "--token-count",
        std::to_string(token_count),
        "--atom-count",
        std::to_string(atom_count),
        "--msa-depth",
        std::to_string(msa_depth),
    };
    auto environment = requestPreprocessorEnvironment();
    executeRequestPreprocessor(arguments, environment, output_path);
    return readPreparedRequest(output_path);
}

std::unordered_set<std::string> names(const std::vector<TensorInfo>& tensors) {
    std::unordered_set<std::string> result;
    for (const auto& tensor : tensors) {
        if (!result.insert(tensor.name).second)
            throw std::invalid_argument("Boltz-2 engine contains a duplicate tensor name");
    }
    return result;
}

void requireNames(ITrtModule& module, std::initializer_list<std::string_view> inputs,
                  std::initializer_list<std::string_view> outputs) {
    const auto actual_inputs = names(module.input_info());
    const auto actual_outputs = names(module.output_info());
    for (const auto expected : inputs) {
        if (actual_inputs.count(std::string(expected)) == 0)
            throw std::invalid_argument("Boltz-2 engine is missing input: " +
                                        std::string(expected));
    }
    for (const auto expected : outputs) {
        if (actual_outputs.count(std::string(expected)) == 0)
            throw std::invalid_argument("Boltz-2 engine is missing output: " +
                                        std::string(expected));
    }
}

void requireExactCounts(ITrtModule& module, std::size_t inputs, std::size_t outputs) {
    if (module.input_info().size() != inputs || module.output_info().size() != outputs)
        throw std::invalid_argument("Boltz-2 engine tensor inventory differs from its contract");
}

void bindPointer(ITrtModule& consumer, std::string_view input, ITrtModule& producer,
                 std::string_view output) {
    if (!consumer.has_input(std::string(input)) || !producer.has_output(std::string(output)))
        throw std::invalid_argument("Boltz-2 attempted to bind an undeclared graph edge");
    if (consumer.tensor_dtype(std::string(input)) != producer.tensor_dtype(std::string(output)) ||
        consumer.tensor_shape(std::string(input)) != producer.tensor_shape(std::string(output))) {
        throw std::invalid_argument("Boltz-2 graph edge dtype or shape differs: " +
                                    std::string(output) + " -> " + std::string(input));
    }
    consumer.bind_external(std::string(input), producer.device_ptr(std::string(output)));
}

float bfloat16ToFloat(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<float> softmaxExpectedBfloat16(const std::vector<uint16_t>& logits, std::size_t rows,
                                           std::size_t bins, float end) {
    if (logits.size() != rows * bins)
        throw std::invalid_argument("Boltz-2 confidence tensor has the wrong element count");
    std::vector<float> result(rows);
    const float width = end / static_cast<float>(bins);
    for (std::size_t row = 0; row < rows; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t bin = 0; bin < bins; ++bin)
            maximum = std::max(maximum, bfloat16ToFloat(logits[row * bins + bin]));
        float denominator = 0.0F;
        float numerator = 0.0F;
        for (std::size_t bin = 0; bin < bins; ++bin) {
            const float probability = std::exp(bfloat16ToFloat(logits[row * bins + bin]) - maximum);
            denominator += probability;
            numerator += probability * (static_cast<float>(bin) + 0.5F) * width;
        }
        result[row] = numerator / denominator;
    }
    return result;
}

std::vector<float> softmaxExpectedTmBfloat16(const std::vector<uint16_t>& logits, std::size_t rows,
                                             std::size_t bins, float d0) {
    if (logits.size() != rows * bins)
        throw std::invalid_argument("Boltz-2 PAE tensor has the wrong element count");
    std::vector<float> result(rows);
    constexpr float kPaeEnd = 32.0F;
    const float width = kPaeEnd / static_cast<float>(bins);
    for (std::size_t row = 0; row < rows; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t bin = 0; bin < bins; ++bin)
            maximum = std::max(maximum, bfloat16ToFloat(logits[row * bins + bin]));
        float denominator = 0.0F;
        float numerator = 0.0F;
        for (std::size_t bin = 0; bin < bins; ++bin) {
            const float probability = std::exp(bfloat16ToFloat(logits[row * bins + bin]) - maximum);
            const float distance = (static_cast<float>(bin) + 0.5F) * width;
            const float tm_value = 1.0F / (1.0F + (distance / d0) * (distance / d0));
            denominator += probability;
            numerator += probability * tm_value;
        }
        result[row] = numerator / denominator;
    }
    return result;
}

std::vector<Vec3> unpackCoordinates(const std::vector<float>& flat, int atom_count) {
    if (atom_count <= 0 || flat.size() != static_cast<std::size_t>(atom_count * 3))
        throw std::invalid_argument("Boltz-2 coordinate tensor differs from its bundle profile");
    std::vector<Vec3> result(static_cast<std::size_t>(atom_count));
    for (int atom = 0; atom < atom_count; ++atom)
        for (int axis = 0; axis < 3; ++axis)
            result[atom][axis] = flat[static_cast<std::size_t>(atom * 3 + axis)];
    return result;
}

Mat3 quaternionRotation(const std::array<float, 4>& quaternion) {
    const float w = quaternion[0];
    const float x = quaternion[1];
    const float y = quaternion[2];
    const float z = quaternion[3];
    return {{{1.0F - 2.0F * (y * y + z * z), 2.0F * (x * y - z * w), 2.0F * (x * z + y * w)},
             {2.0F * (x * y + z * w), 1.0F - 2.0F * (x * x + z * z), 2.0F * (y * z - x * w)},
             {2.0F * (x * z - y * w), 2.0F * (y * z + x * w), 1.0F - 2.0F * (x * x + y * y)}}};
}

std::vector<float> packCoordinates(const std::vector<Vec3>& coordinates) {
    std::vector<float> result(coordinates.size() * 3);
    for (std::size_t atom = 0; atom < coordinates.size(); ++atom)
        for (int axis = 0; axis < 3; ++axis)
            result[atom * 3 + static_cast<std::size_t>(axis)] = coordinates[atom][axis];
    return result;
}

Vec3 transform(const Vec3& point, const Mat3& rotation, const Vec3& translation) {
    Vec3 result{};
    for (int row = 0; row < 3; ++row) {
        result[row] = translation[row];
        for (int column = 0; column < 3; ++column)
            result[row] += rotation[row][column] * point[column];
    }
    return result;
}

void applyAugmentation(std::vector<Vec3>& coordinates, std::vector<Vec3>* denoised,
                       const std::array<float, 9>& flat_rotation,
                       const std::array<float, 3>& translation) {
    Mat3 rotation{};
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            rotation[row][column] = flat_rotation[static_cast<std::size_t>(column * 3 + row)];
    const auto apply = [&](std::vector<Vec3>& values) {
        Vec3 centroid{};
        for (const auto& point : values)
            for (int axis = 0; axis < 3; ++axis)
                centroid[axis] += point[axis];
        for (float& value : centroid)
            value /= static_cast<float>(values.size());
        for (auto& point : values) {
            for (int axis = 0; axis < 3; ++axis)
                point[axis] -= centroid[axis];
            point = transform(point, rotation, translation);
        }
    };
    apply(coordinates);
    if (denoised != nullptr)
        apply(*denoised);
}

struct Centroids {
    Vec3 source{};
    Vec3 target{};
};

Centroids weightedCentroids(const std::vector<Vec3>& source, const std::vector<Vec3>& target,
                            const float* weights) {
    Centroids centroids;
    float weight_sum = 0.0F;
    for (std::size_t point = 0; point < source.size(); ++point) {
        const float weight = weights[point] * weights[point];
        weight_sum += weight;
        for (int axis = 0; axis < 3; ++axis) {
            centroids.source[axis] += weight * source[point][axis];
            centroids.target[axis] += weight * target[point][axis];
        }
    }
    if (weight_sum <= 0.0F)
        throw std::invalid_argument("Boltz-2 atom mask contains no valid atoms");
    for (int axis = 0; axis < 3; ++axis) {
        centroids.source[axis] /= weight_sum;
        centroids.target[axis] /= weight_sum;
    }
    return centroids;
}

std::array<std::array<double, 3>, 3> weightedCovariance(const std::vector<Vec3>& source,
                                                        const std::vector<Vec3>& target,
                                                        const float* weights,
                                                        const Centroids& centroids) {
    std::array<std::array<double, 3>, 3> covariance{};
    for (std::size_t point = 0; point < source.size(); ++point) {
        const float weight = weights[point] * weights[point];
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                covariance[row][column] += weight * (source[point][row] - centroids.source[row]) *
                                           (target[point][column] - centroids.target[column]);
            }
        }
    }
    return covariance;
}

using HornMatrix = std::array<std::array<double, 4>, 4>;

HornMatrix hornMatrix(const std::array<std::array<double, 3>, 3>& covariance) {
    const double trace = covariance[0][0] + covariance[1][1] + covariance[2][2];
    return {{{{trace, covariance[1][2] - covariance[2][1], covariance[2][0] - covariance[0][2],
               covariance[0][1] - covariance[1][0]}},
             {{covariance[1][2] - covariance[2][1],
               covariance[0][0] - covariance[1][1] - covariance[2][2],
               covariance[0][1] + covariance[1][0], covariance[0][2] + covariance[2][0]}},
             {{covariance[2][0] - covariance[0][2], covariance[0][1] + covariance[1][0],
               -covariance[0][0] + covariance[1][1] - covariance[2][2],
               covariance[1][2] + covariance[2][1]}},
             {{covariance[0][1] - covariance[1][0], covariance[0][2] + covariance[2][0],
               covariance[1][2] + covariance[2][1],
               -covariance[0][0] - covariance[1][1] + covariance[2][2]}}}};
}

void jacobiRotate(HornMatrix& matrix, HornMatrix& eigenvectors, int first, int second) {
    const double off_diagonal = matrix[first][second];
    const double tau = (matrix[second][second] - matrix[first][first]) / (2.0 * off_diagonal);
    const double tangent = std::copysign(1.0 / (std::abs(tau) + std::sqrt(1.0 + tau * tau)), tau);
    const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
    const double sine = tangent * cosine;
    matrix[first][first] -= tangent * off_diagonal;
    matrix[second][second] += tangent * off_diagonal;
    matrix[first][second] = 0.0;
    matrix[second][first] = 0.0;
    for (int row = 0; row < 4; ++row) {
        if (row == first || row == second)
            continue;
        const double first_value = matrix[row][first];
        const double second_value = matrix[row][second];
        matrix[row][first] = cosine * first_value - sine * second_value;
        matrix[first][row] = matrix[row][first];
        matrix[row][second] = sine * first_value + cosine * second_value;
        matrix[second][row] = matrix[row][second];
    }
    for (int row = 0; row < 4; ++row) {
        const double first_value = eigenvectors[row][first];
        const double second_value = eigenvectors[row][second];
        eigenvectors[row][first] = cosine * first_value - sine * second_value;
        eigenvectors[row][second] = sine * first_value + cosine * second_value;
    }
}

double jacobiSweep(HornMatrix& matrix, HornMatrix& eigenvectors) {
    double largest_off_diagonal = 0.0;
    for (int first = 0; first < 4; ++first) {
        for (int second = first + 1; second < 4; ++second) {
            const double off_diagonal = matrix[first][second];
            largest_off_diagonal = std::max(largest_off_diagonal, std::abs(off_diagonal));
            if (std::abs(off_diagonal) > 1.0e-14)
                jacobiRotate(matrix, eigenvectors, first, second);
        }
    }
    return largest_off_diagonal;
}

std::array<float, 4> dominantQuaternion(HornMatrix matrix) {
    HornMatrix eigenvectors{};
    for (int axis = 0; axis < 4; ++axis)
        eigenvectors[axis][axis] = 1.0;
    for (int sweep = 0; sweep < 32; ++sweep) {
        if (jacobiSweep(matrix, eigenvectors) <= 1.0e-12)
            break;
    }
    int largest = 0;
    for (int axis = 1; axis < 4; ++axis) {
        if (matrix[axis][axis] > matrix[largest][largest])
            largest = axis;
    }
    std::array<float, 4> quaternion{};
    for (int axis = 0; axis < 4; ++axis)
        quaternion[axis] = static_cast<float>(eigenvectors[axis][largest]);
    return quaternion;
}

std::vector<Vec3> weightedRigidAlign(const std::vector<Vec3>& source,
                                     const std::vector<Vec3>& target, const float* weights) {
    if (source.size() != target.size())
        throw std::invalid_argument("Boltz-2 alignment point clouds differ in size");
    // Horn's matrix is symmetric. Power iteration is not valid here because
    // it selects the eigenvalue with largest magnitude; the optimal rotation
    // requires the largest algebraic eigenvalue. Diagonalize the 4x4 matrix
    // with Jacobi rotations and select that eigenvector explicitly.
    const auto centroids = weightedCentroids(source, target, weights);
    const auto covariance = weightedCovariance(source, target, weights, centroids);
    const Mat3 rotation = quaternionRotation(dominantQuaternion(hornMatrix(covariance)));
    std::vector<Vec3> result(source.size());
    for (std::size_t point = 0; point < source.size(); ++point) {
        Vec3 centered{};
        for (int axis = 0; axis < 3; ++axis)
            centered[axis] = source[point][axis] - centroids.source[axis];
        result[point] = transform(centered, rotation, centroids.target);
    }
    return result;
}

std::vector<float> sigmaSchedule(int32_t steps) {
    std::vector<float> result(static_cast<std::size_t>(steps) + 1U, 0.0F);
    const float begin = std::pow(kSigmaMax, 1.0F / kRho);
    const float end = std::pow(kSigmaMin, 1.0F / kRho);
    for (int32_t step = 0; step < steps; ++step) {
        const float fraction = static_cast<float>(step) / static_cast<float>(steps - 1);
        result[static_cast<std::size_t>(step)] =
            std::pow(begin + fraction * (end - begin), kRho) * kSigmaData;
    }
    return result;
}

std::string atomElement(std::string_view atom_name) {
    for (const char character : atom_name) {
        if ((character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z'))
            return std::string(1, static_cast<char>(std::toupper(character)));
    }
    return "C";
}

bool validTokenShape(const std::vector<int64_t>& shape) {
    return shape.size() == 3 && shape[0] == 1 && shape[1] > 0 && shape[1] <= kMaxTokenCount &&
           shape[1] <= std::numeric_limits<int>::max() && shape[2] == 33;
}

bool validAtomShape(const std::vector<int64_t>& shape) {
    return shape.size() == 3 && shape[0] == 1 && shape[1] > 0 && shape[1] <= kMaxAtomCount &&
           shape[1] <= std::numeric_limits<int>::max() && shape[2] == 3 &&
           shape[1] % kAtomWindowQueries == 0;
}

bool validMsaShape(const std::vector<int64_t>& shape, int64_t token_count) {
    return shape.size() == 3 && shape[0] == 1 && shape[1] == kMsaDepth && shape[2] == token_count;
}

bool hasFeatureStorage(const FeatureTensor& feature, DType dtype, std::size_t element_count) {
    return feature.dtype == dtype && feature.data.size() == element_count * dtype_size(dtype);
}

void requireProfileShapes(const std::vector<int64_t>& token_shape,
                          const std::vector<int64_t>& atom_shape,
                          const std::vector<int64_t>& msa_shape) {
    if (!validTokenShape(token_shape))
        throw std::invalid_argument("Boltz-2 feature shapes do not define a valid bundle profile");
    if (!validAtomShape(atom_shape))
        throw std::invalid_argument("Boltz-2 feature shapes do not define a valid bundle profile");
    if (!validMsaShape(msa_shape, token_shape[1]))
        throw std::invalid_argument("Boltz-2 feature shapes do not define a valid bundle profile");
}

void requireFeatureStorage(const FeatureTensor& feature, DType dtype, std::size_t element_count) {
    if (!hasFeatureStorage(feature, dtype, element_count))
        throw std::invalid_argument("Boltz-2 host features differ from the bundle profile");
}

int activePrefixCount(const FeatureTensor& mask, int profile_count, std::string_view label) {
    const auto* values = reinterpret_cast<const float*>(mask.data.data());
    int active_count = 0;
    bool saw_padding = false;
    for (int index = 0; index < profile_count; ++index) {
        const float value = values[index];
        if (value != 0.0F && value != 1.0F)
            throw std::invalid_argument("Boltz-2 " + std::string(label) +
                                        "_pad_mask must be binary");
        if (value == 0.0F) {
            saw_padding = true;
            continue;
        }
        if (saw_padding)
            throw std::invalid_argument("Boltz-2 active " + std::string(label) +
                                        "s must precede profile padding");
        ++active_count;
    }
    if (active_count == 0)
        throw std::invalid_argument("Boltz-2 request must contain active " + std::string(label) +
                                    "s");
    return active_count;
}

bool hasRequiredEngines(const EngineSet& engines) {
    return engines.input && engines.trunk_init && engines.msa && engines.conditioning &&
           engines.score_input && engines.score_output && engines.confidence;
}

void requireStream(const std::unique_ptr<ITrtModule>& module, cudaStream_t stream) {
    if (!module || !module->ok() || module->stream() != stream)
        throw std::invalid_argument("Boltz-2 engines must be valid on one CUDA stream");
}

bool matchesRandomProfile(const RandomSamples& samples, int32_t seed, int32_t sampling_steps,
                          int atom_count) {
    return samples.seed == seed && samples.sampling_steps == sampling_steps &&
           samples.atom_count == atom_count;
}

std::vector<Vec3> initialCoordinates(const RandomSamples& random, const std::vector<float>& sigmas,
                                     int atom_count) {
    std::vector<Vec3> coordinates(static_cast<std::size_t>(atom_count));
    for (int atom = 0; atom < random.atom_count; ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
            coordinates[atom][axis] =
                sigmas[0] * random.initial[static_cast<std::size_t>(atom * 3 + axis)];
        }
    }
    return coordinates;
}

std::vector<Vec3> addStepNoise(const std::vector<Vec3>& coordinates, const RandomSamples& random,
                               int32_t step, float noise_scale) {
    std::vector<Vec3> noisy = coordinates;
    for (int atom = 0; atom < random.atom_count; ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
            noisy[atom][axis] +=
                noise_scale *
                random
                    .noise[static_cast<std::size_t>((step * random.atom_count + atom) * 3 + axis)];
        }
    }
    return noisy;
}

std::vector<float> scaledCoordinates(const std::vector<Vec3>& coordinates, float scale) {
    auto scaled = coordinates;
    for (auto& point : scaled)
        for (float& value : point)
            value *= scale;
    return packCoordinates(scaled);
}

std::vector<Vec3> denoisedCoordinates(const std::vector<Vec3>& noisy,
                                      const std::vector<float>& update, float t_hat) {
    const float sigma_data_squared = kSigmaData * kSigmaData;
    const float c_skip = sigma_data_squared / (t_hat * t_hat + sigma_data_squared);
    const float c_out = t_hat * kSigmaData / std::sqrt(sigma_data_squared + t_hat * t_hat);
    std::vector<Vec3> denoised(noisy.size());
    for (std::size_t atom = 0; atom < noisy.size(); ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
            denoised[atom][axis] = c_skip * noisy[atom][axis] +
                                   c_out * update[atom * 3 + static_cast<std::size_t>(axis)];
        }
    }
    return denoised;
}

std::vector<Vec3> advanceCoordinates(std::vector<Vec3> noisy, const std::vector<Vec3>& denoised,
                                     const float* atom_mask, float sigma_t, float t_hat) {
    noisy = weightedRigidAlign(noisy, denoised, atom_mask);
    const float scale = kStepScale * (sigma_t - t_hat) / t_hat;
    std::vector<Vec3> coordinates(noisy.size());
    for (std::size_t atom = 0; atom < noisy.size(); ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
            coordinates[atom][axis] =
                noisy[atom][axis] + scale * (noisy[atom][axis] - denoised[atom][axis]);
        }
    }
    return coordinates;
}

float maskedMean(const std::vector<float>& values, const float* mask, int count) {
    float total = 0.0F;
    float weight = 0.0F;
    for (int index = 0; index < count; ++index) {
        total += values[static_cast<std::size_t>(index)] * mask[index];
        weight += mask[index];
    }
    return total / weight;
}

bool validFrameIndices(const std::array<int32_t, 3>& frame, int atom_count) {
    return frame[0] >= 0 && frame[1] >= 0 && frame[2] >= 0 && frame[0] < atom_count &&
           frame[1] < atom_count && frame[2] < atom_count;
}

bool validFrameGeometry(const std::vector<Vec3>& points, const std::array<int32_t, 3>& frame) {
    Vec3 first_vector{};
    Vec3 second_vector{};
    float first_norm = 0.0F;
    float second_norm = 0.0F;
    float dot = 0.0F;
    for (int axis = 0; axis < 3; ++axis) {
        first_vector[axis] = points[frame[1]][axis] - points[frame[0]][axis];
        second_vector[axis] = points[frame[1]][axis] - points[frame[2]][axis];
        first_norm += first_vector[axis] * first_vector[axis];
        second_norm += second_vector[axis] * second_vector[axis];
        dot += first_vector[axis] * second_vector[axis];
    }
    first_norm = std::sqrt(first_norm);
    second_norm = std::sqrt(second_norm);
    return first_norm > 1.0e-2F && second_norm > 1.0e-2F &&
           std::abs(dot / ((first_norm + 1.0e-6F) * (second_norm + 1.0e-6F))) < 0.9063F;
}

std::vector<float> confidenceFrameMask(const std::vector<float>& coordinates, const int32_t* frames,
                                       const float* token_mask, int token_count, int atom_count) {
    const auto points = unpackCoordinates(coordinates, atom_count);
    std::vector<float> frame_mask(static_cast<std::size_t>(token_count));
    for (int token = 0; token < token_count; ++token) {
        const std::array<int32_t, 3> frame{frames[token * 3], frames[token * 3 + 1],
                                           frames[token * 3 + 2]};
        if (validFrameIndices(frame, atom_count) && validFrameGeometry(points, frame))
            frame_mask[static_cast<std::size_t>(token)] = token_mask[token];
    }
    return frame_mask;
}

float maximumTmScore(const std::vector<float>& expected, const std::vector<float>& frame_mask,
                     const float* token_mask, int token_count) {
    float maximum = 0.0F;
    for (int row = 0; row < token_count; ++row) {
        float numerator = 0.0F;
        float denominator = 0.0F;
        for (int column = 0; column < token_count; ++column) {
            const float mask =
                frame_mask[static_cast<std::size_t>(row)] * token_mask[column] * token_mask[row];
            numerator += expected[static_cast<std::size_t>(row * token_count + column)] * mask;
            denominator += mask;
        }
        maximum = std::max(maximum, numerator / (denominator + 1.0e-5F));
    }
    return maximum;
}

struct AtomRow {
    std::string atom;
    std::string residue;
    std::string chain;
    int residue_index{0};
    std::size_t token_index{0};
};

void applyResidueMetadata(std::vector<AtomRow>& rows, const nlohmann::json& residues) {
    for (std::size_t token = 0; token < residues.size(); ++token) {
        const auto& residue = residues.at(token);
        const auto begin = residue.at("atom_index").get<std::size_t>();
        const auto count = residue.at("atom_count").get<std::size_t>();
        if (begin > rows.size() || count > rows.size() - begin)
            throw std::invalid_argument("Boltz-2 residue metadata exceeds atom inventory");
        for (std::size_t atom = begin; atom < begin + count; ++atom) {
            rows[atom].residue = residue.at("name").get<std::string>();
            rows[atom].residue_index = residue.at("index").get<int>();
            rows[atom].token_index = token;
        }
    }
}

void applyChainMetadata(std::vector<AtomRow>& rows, const nlohmann::json& chains) {
    for (const auto& chain : chains) {
        const auto begin = chain.at("atom_index").get<std::size_t>();
        const auto count = chain.at("atom_count").get<std::size_t>();
        if (begin > rows.size() || count > rows.size() - begin)
            throw std::invalid_argument("Boltz-2 chain metadata exceeds atom inventory");
        for (std::size_t atom = begin; atom < begin + count; ++atom)
            rows[atom].chain = chain.at("name").get<std::string>();
    }
}

std::vector<AtomRow> structureRows(const nlohmann::json& document, int atom_count) {
    const auto& atoms = document.at("atoms");
    if (!atoms.is_array() || atoms.size() > static_cast<std::size_t>(atom_count))
        throw std::invalid_argument("Boltz-2 structure metadata atom inventory is invalid");
    std::vector<AtomRow> rows(atoms.size());
    for (std::size_t atom = 0; atom < rows.size(); ++atom)
        rows[atom].atom = atoms.at(atom).get<std::string>();
    applyResidueMetadata(rows, document.at("residues"));
    applyChainMetadata(rows, document.at("chains"));
    for (const auto& row : rows) {
        if (row.residue.empty())
            throw std::invalid_argument("Boltz-2 residue metadata does not cover every atom");
    }
    return rows;
}

std::string writePdb(const std::vector<AtomRow>& rows, const std::vector<float>& coordinates,
                     const StructureConfidence& confidence) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(3);
    for (std::size_t atom = 0; atom < rows.size(); ++atom) {
        const auto& row = rows[atom];
        output << "ATOM  " << std::setw(5) << atom + 1 << ' ' << std::left << std::setw(4)
               << row.atom << std::right << ' ' << std::setw(3) << row.residue << ' '
               << (row.chain.empty() ? 'A' : row.chain.front()) << std::setw(4) << row.residue_index
               << "    " << std::setw(8) << coordinates[atom * 3] << std::setw(8)
               << coordinates[atom * 3 + 1] << std::setw(8) << coordinates[atom * 3 + 2]
               << "  1.00 " << std::setw(5) << 100.0F * confidence.plddt.at(row.token_index)
               << "          " << std::setw(2) << atomElement(row.atom) << '\n';
    }
    output << "END\n";
    return output.str();
}

std::string writeMmcif(const std::vector<AtomRow>& rows, const std::vector<float>& coordinates,
                       const StructureConfidence& confidence) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(3)
           << "data_boltz2\n#\nloop_\n"
              "_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
              "_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n"
              "_atom_site.label_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n"
              "_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n"
              "_atom_site.pdbx_PDB_model_num\n";
    for (std::size_t atom = 0; atom < rows.size(); ++atom) {
        const auto& row = rows[atom];
        output << "ATOM " << atom + 1 << ' ' << atomElement(row.atom) << ' ' << row.atom << ' '
               << row.residue << ' ' << (row.chain.empty() ? "A" : row.chain) << ' '
               << row.residue_index << ' ' << coordinates[atom * 3] << ' '
               << coordinates[atom * 3 + 1] << ' ' << coordinates[atom * 3 + 2] << " 1.00 "
               << 100.0F * confidence.plddt.at(row.token_index) << " 1\n";
    }
    output << "#\n";
    return output.str();
}

} // namespace

Boltz2Pipeline::Boltz2Pipeline(EngineSet engines, BundleArtifacts artifacts, std::string model_id,
                               std::string preprocessor_python)
    : engines_(std::move(engines)), artifacts_(std::move(artifacts)),
      model_id_(std::move(model_id)), preprocessor_python_(std::move(preprocessor_python)) {
    if (model_id_.empty())
        model_id_ = std::string(kModelId);
    validateAndBindEngines();
}

std::string Boltz2Pipeline::prepare_structure_input(const std::string& input,
                                                    const std::string& input_path) {
    if (PreparedRequest::isPrepared(input.data(), input.size())) {
        activateRequest(PreparedRequest::parse(input.data(), input.size()));
        return artifacts_.request;
    }
    if (input == artifacts_.request)
        return input;
    const auto payload =
        runRequestPreprocessor(preprocessor_python_, input, input_path, token_count_, atom_count_,
                               static_cast<int>(feature("msa").shape.at(1)));
    activateRequest(PreparedRequest::parse(payload.data(), payload.size()));
    return artifacts_.request;
}

const FeatureTensor& Boltz2Pipeline::feature(std::string_view name) const {
    return artifacts_.features.require(name);
}

void Boltz2Pipeline::bindFeature(ITrtModule& module, std::string_view name) {
    const auto found = device_features_.find(std::string(name));
    if (found == device_features_.end())
        throw std::logic_error("Boltz-2 device feature was not uploaded: " + std::string(name));
    if (!module.has_input(std::string(name)) ||
        module.tensor_dtype(std::string(name)) != found->second.dtype() ||
        module.tensor_shape(std::string(name)) != found->second.shape()) {
        throw std::invalid_argument("Boltz-2 feature binding differs from engine contract: " +
                                    std::string(name));
    }
    module.bind_external(std::string(name), found->second.data());
}

void Boltz2Pipeline::validateAndBindEngines() {
    validateStreams();
    configureProfile();
    uploadFeatures();
    allocateRuntimeTensors();
    bindTrunkEngines();
    ITrtModule* trunk_output = bindPairformerEngines();
    bindDiffusionEngines(*trunk_output);
    bindConfidenceEngine(*trunk_output);
}

void Boltz2Pipeline::validateStreams() {
    if (!hasRequiredEngines(engines_))
        throw std::invalid_argument("Boltz-2 engine set is incomplete");
    stream_ = engines_.input->stream();
    if (stream_ == nullptr)
        throw std::invalid_argument("Boltz-2 TensorRT stream is null");
    requireStream(engines_.trunk_init, stream_);
    requireStream(engines_.msa, stream_);
    requireStream(engines_.conditioning, stream_);
    requireStream(engines_.score_input, stream_);
    requireStream(engines_.score_output, stream_);
    requireStream(engines_.confidence, stream_);
    for (const auto& module : engines_.pairformer)
        requireStream(module, stream_);
    for (const auto& module : engines_.score_token)
        requireStream(module, stream_);
}

void Boltz2Pipeline::configureProfile() {
    const auto& token_shape = feature("res_type").shape;
    const auto& atom_shape = feature("ref_pos").shape;
    const auto& msa_shape = feature("msa").shape;
    requireProfileShapes(token_shape, atom_shape, msa_shape);
    token_count_ = static_cast<int>(token_shape[1]);
    atom_count_ = static_cast<int>(atom_shape[1]);
    const auto& atom_mask = feature("atom_pad_mask");
    const auto& token_mask = feature("token_pad_mask");
    const auto& frames = feature("frames_idx");
    requireFeatureStorage(atom_mask, DType::kFloat32, static_cast<std::size_t>(atom_count_));
    requireFeatureStorage(token_mask, DType::kFloat32, static_cast<std::size_t>(token_count_));
    requireFeatureStorage(frames, DType::kInt32, static_cast<std::size_t>(token_count_) * 3U);
    active_atom_count_ = activePrefixCount(atom_mask, atom_count_, "atom");
    active_token_count_ = activePrefixCount(token_mask, token_count_, "token");
    if (!matchesRandomProfile(artifacts_.random_samples, 42, 200, atom_count_))
        throw std::invalid_argument("Boltz-2 random samples differ from feature atom count");
}

void Boltz2Pipeline::activateRequest(PreparedRequest request) {
    const auto& token_shape = request.features.require("res_type").shape;
    const auto& atom_shape = request.features.require("ref_pos").shape;
    const auto& msa_shape = request.features.require("msa").shape;
    if (!validTokenShape(token_shape) || !validAtomShape(atom_shape) ||
        !validMsaShape(msa_shape, token_shape[1]) || token_shape[1] != token_count_ ||
        atom_shape[1] != atom_count_ || msa_shape != artifacts_.features.require("msa").shape) {
        throw std::invalid_argument(
            "Boltz-2 prepared request differs from the compiled bundle profile");
    }
    auto previous_features = artifacts_.features;
    auto previous_random_samples = artifacts_.random_samples;
    auto previous_request = artifacts_.request;
    auto previous_metadata = artifacts_.structure_metadata_json;
    try {
        artifacts_.features = std::move(request.features);
        artifacts_.random_samples = std::move(request.random_samples);
        artifacts_.request = std::move(request.request);
        artifacts_.structure_metadata_json = std::move(request.structure_metadata_json);
        configureProfile();
        const auto rows =
            structureRows(nlohmann::json::parse(artifacts_.structure_metadata_json), atom_count_);
        if (rows.size() != static_cast<std::size_t>(active_atom_count_))
            throw std::invalid_argument(
                "Boltz-2 structure metadata atom count differs from the active request");
        uploadFeatures();
    } catch (...) {
        artifacts_.features = std::move(previous_features);
        artifacts_.random_samples = std::move(previous_random_samples);
        artifacts_.request = std::move(previous_request);
        artifacts_.structure_metadata_json = std::move(previous_metadata);
        configureProfile();
        uploadFeatures();
        throw;
    }
}

void Boltz2Pipeline::uploadFeatures() {
    for (const auto name : kFeatureNames) {
        const auto& host = feature(name);
        auto [found, inserted] =
            device_features_.try_emplace(std::string(name), host.shape, host.dtype, stream_);
        if ((!inserted &&
             (found->second.shape() != host.shape || found->second.dtype() != host.dtype)) ||
            !found->second.ok() || !found->second.copy_from_host(host.data.data()))
            throw std::runtime_error("Boltz-2 failed to upload feature: " + std::string(name));
    }
}

void Boltz2Pipeline::allocateRuntimeTensors() {
    zero_s_ = DeviceTensor::zeros({1, token_count_, 384}, DType::kFloat32, stream_);
    zero_z_ = DeviceTensor::zeros({1, token_count_, token_count_, 128}, DType::kFloat32, stream_);
    r_noisy_ = DeviceTensor({1, atom_count_, 3}, DType::kFloat32, stream_);
    time_ = DeviceTensor({1}, DType::kFloat32, stream_);
    x_pred_ = DeviceTensor({1, atom_count_, 3}, DType::kFloat32, stream_);
    if (!zero_s_.ok() || !zero_z_.ok() || !r_noisy_.ok() || !time_.ok() || !x_pred_.ok())
        throw std::runtime_error("Boltz-2 failed to allocate runtime tensors");
}

void Boltz2Pipeline::bindTrunkEngines() {
    requireExactCounts(*engines_.input, 14, 1);
    requireNames(*engines_.input,
                 {"ref_pos", "ref_space_uid", "ref_charge", "ref_element", "ref_atom_name_chars",
                  "atom_to_token", "atom_pad_mask", "res_type", "profile", "deletion_mean",
                  "method_feature", "modified", "cyclic_period", "mol_type"},
                 {"s_inputs"});
    for (const auto name :
         {"ref_pos", "ref_space_uid", "ref_charge", "ref_element", "ref_atom_name_chars",
          "atom_to_token", "atom_pad_mask", "res_type", "profile", "deletion_mean",
          "method_feature", "modified", "cyclic_period", "mol_type"})
        bindFeature(*engines_.input, name);

    requireExactCounts(*engines_.trunk_init, 12, 3);
    requireNames(*engines_.trunk_init,
                 {"s_inputs", "recycle_s", "recycle_z", "asym_id", "residue_index", "entity_id",
                  "token_index", "sym_id", "token_bonds", "type_bonds", "contact_conditioning",
                  "contact_threshold"},
                 {"s", "z", "relative_position_encoding"});
    bindPointer(*engines_.trunk_init, "s_inputs", *engines_.input, "s_inputs");
    for (const auto name :
         {"asym_id", "residue_index", "entity_id", "token_index", "sym_id", "token_bonds",
          "type_bonds", "contact_conditioning", "contact_threshold"})
        bindFeature(*engines_.trunk_init, name);

    requireExactCounts(*engines_.msa, 8, 1);
    requireNames(*engines_.msa,
                 {"z", "s_inputs", "msa", "has_deletion", "deletion_value", "msa_paired",
                  "msa_mask", "token_mask"},
                 {"z_out"});
    bindPointer(*engines_.msa, "z", *engines_.trunk_init, "z");
    bindPointer(*engines_.msa, "s_inputs", *engines_.input, "s_inputs");
    for (const auto name : {"msa", "has_deletion", "deletion_value", "msa_paired", "msa_mask"})
        bindFeature(*engines_.msa, name);
    engines_.msa->bind_external("token_mask", device_features_.at("token_pad_mask").data());
}

ITrtModule* Boltz2Pipeline::bindPairformerEngines() {
    ITrtModule* previous = engines_.msa.get();
    std::string previous_s = "s";
    for (auto& module : engines_.pairformer) {
        requireExactCounts(*module, 3, 2);
        requireNames(*module, {"s", "z", "token_mask"}, {"s_out", "z_out"});
        if (previous == engines_.msa.get()) {
            bindPointer(*module, "s", *engines_.trunk_init, "s");
            bindPointer(*module, "z", *engines_.msa, "z_out");
        } else {
            bindPointer(*module, "s", *previous, previous_s);
            bindPointer(*module, "z", *previous, "z_out");
        }
        module->bind_external("token_mask", device_features_.at("token_pad_mask").data());
        previous = module.get();
        previous_s = "s_out";
    }
    return previous;
}

void Boltz2Pipeline::bindDiffusionEngines(ITrtModule& trunk_output) {
    requireExactCounts(*engines_.conditioning, 10, 5);
    requireNames(*engines_.conditioning,
                 {"s_trunk", "z_trunk", "relative_position_encoding", "ref_pos", "ref_space_uid",
                  "ref_charge", "ref_element", "ref_atom_name_chars", "atom_to_token",
                  "atom_pad_mask"},
                 {"q", "c", "atom_enc_bias", "atom_dec_bias", "token_trans_bias"});
    bindPointer(*engines_.conditioning, "s_trunk", trunk_output, "s_out");
    bindPointer(*engines_.conditioning, "z_trunk", trunk_output, "z_out");
    bindPointer(*engines_.conditioning, "relative_position_encoding", *engines_.trunk_init,
                "relative_position_encoding");
    for (const auto name : {"ref_pos", "ref_space_uid", "ref_charge", "ref_element",
                            "ref_atom_name_chars", "atom_to_token", "atom_pad_mask"})
        bindFeature(*engines_.conditioning, name);

    requireExactCounts(*engines_.score_input, 9, 40);
    requireNames(*engines_.score_input,
                 {"s_inputs", "s_trunk", "q_static", "c_static", "atom_enc_bias", "r_noisy", "time",
                  "atom_to_token", "atom_pad_mask"},
                 {"a", "single_condition", "q_skip", "c_skip"});
    bindPointer(*engines_.score_input, "s_inputs", *engines_.input, "s_inputs");
    bindPointer(*engines_.score_input, "s_trunk", trunk_output, "s_out");
    bindPointer(*engines_.score_input, "q_static", *engines_.conditioning, "q");
    bindPointer(*engines_.score_input, "c_static", *engines_.conditioning, "c");
    bindPointer(*engines_.score_input, "atom_enc_bias", *engines_.conditioning, "atom_enc_bias");
    engines_.score_input->bind_external("r_noisy", r_noisy_.data());
    engines_.score_input->bind_external("time", time_.data());
    bindFeature(*engines_.score_input, "atom_to_token");
    bindFeature(*engines_.score_input, "atom_pad_mask");

    ITrtModule* previous_token = engines_.score_input.get();
    std::string previous_a = "a";
    for (auto& module : engines_.score_token) {
        requireExactCounts(*module, 4, 1);
        requireNames(*module, {"a", "single_condition", "token_trans_bias", "token_mask"},
                     {"a_out"});
        bindPointer(*module, "a", *previous_token, previous_a);
        bindPointer(*module, "single_condition", *engines_.score_input, "single_condition");
        bindPointer(*module, "token_trans_bias", *engines_.conditioning, "token_trans_bias");
        module->bind_external("token_mask", device_features_.at("token_pad_mask").data());
        previous_token = module.get();
        previous_a = "a_out";
    }

    requireExactCounts(*engines_.score_output, 6, 37);
    requireNames(*engines_.score_output,
                 {"a", "q_skip", "c_skip", "atom_dec_bias", "atom_to_token", "atom_pad_mask"},
                 {"r_update"});
    bindPointer(*engines_.score_output, "a", *previous_token, "a_out");
    bindPointer(*engines_.score_output, "q_skip", *engines_.score_input, "q_skip");
    bindPointer(*engines_.score_output, "c_skip", *engines_.score_input, "c_skip");
    bindPointer(*engines_.score_output, "atom_dec_bias", *engines_.conditioning, "atom_dec_bias");
    bindFeature(*engines_.score_output, "atom_to_token");
    bindFeature(*engines_.score_output, "atom_pad_mask");
}

void Boltz2Pipeline::bindConfidenceEngine(ITrtModule& trunk_output) {
    requireExactCounts(*engines_.confidence, 15, 7);
    requireNames(*engines_.confidence,
                 {"s_inputs", "s", "z", "x_pred", "token_to_rep_atom", "asym_id", "residue_index",
                  "entity_id", "token_index", "sym_id", "token_bonds", "type_bonds",
                  "contact_conditioning", "contact_threshold", "token_mask"},
                 {"pae_logits", "pde_logits", "plddt_logits", "resolved_logits",
                  "representative_distance", "pdistogram", "pbfactor"});
    bindPointer(*engines_.confidence, "s_inputs", *engines_.input, "s_inputs");
    bindPointer(*engines_.confidence, "s", trunk_output, "s_out");
    bindPointer(*engines_.confidence, "z", trunk_output, "z_out");
    engines_.confidence->bind_external("x_pred", x_pred_.data());
    for (const auto name :
         {"token_to_rep_atom", "asym_id", "residue_index", "entity_id", "token_index", "sym_id",
          "token_bonds", "type_bonds", "contact_conditioning", "contact_threshold"})
        bindFeature(*engines_.confidence, name);
    engines_.confidence->bind_external("token_mask", device_features_.at("token_pad_mask").data());
}

void Boltz2Pipeline::runTrunk() {
    engines_.input->forward_device_async({});
    for (int pass = 0; pass < 4; ++pass) {
        if (pass == 0) {
            engines_.trunk_init->bind_external("recycle_s", zero_s_.data());
            engines_.trunk_init->bind_external("recycle_z", zero_z_.data());
        } else {
            engines_.trunk_init->bind_external("recycle_s",
                                               engines_.pairformer.back()->device_ptr("s_out"));
            engines_.trunk_init->bind_external("recycle_z",
                                               engines_.pairformer.back()->device_ptr("z_out"));
        }
        engines_.trunk_init->forward_device_async({});
        engines_.msa->forward_device_async({});
        for (auto& module : engines_.pairformer)
            module->forward_device_async({});
    }
}

void Boltz2Pipeline::runConditioning() {
    engines_.conditioning->forward_device_async({});
}

std::vector<float> Boltz2Pipeline::runDiffusionScore(const std::vector<float>& model_input,
                                                     float time_value) {
    if (!r_noisy_.copy_from_host(model_input.data()) || !time_.copy_from_host(&time_value))
        throw std::runtime_error("Boltz-2 failed to upload diffusion step inputs");
    engines_.score_input->forward_device_async({});
    for (auto& module : engines_.score_token)
        module->forward_device_async({});
    engines_.score_output->forward_device_async({});
    std::vector<float> update(static_cast<std::size_t>(atom_count_ * 3));
    if (cudaMemcpyAsync(update.data(), engines_.score_output->device_ptr("r_update"),
                        update.size() * sizeof(float), cudaMemcpyDeviceToHost,
                        stream_) != cudaSuccess ||
        cudaStreamSynchronize(stream_) != cudaSuccess) {
        throw std::runtime_error("Boltz-2 failed to read diffusion score output");
    }
    return update;
}

std::vector<float> Boltz2Pipeline::sampleCoordinates(int32_t seed, int32_t sampling_steps) {
    const auto& mask_feature = feature("atom_pad_mask");
    const auto* atom_mask = reinterpret_cast<const float*>(mask_feature.data.data());
    const auto& random = artifacts_.random_samples;
    if (!matchesRandomProfile(random, seed, sampling_steps, atom_count_))
        throw std::invalid_argument("Boltz-2 request differs from bundled random samples");
    const auto sigmas = sigmaSchedule(sampling_steps);
    auto coordinates = initialCoordinates(random, sigmas, atom_count_);
    std::vector<Vec3> denoised;
    bool has_denoised = false;
    for (int32_t step = 0; step < sampling_steps; ++step) {
        const float sigma_tm = sigmas[static_cast<std::size_t>(step)];
        const float sigma_t = sigmas[static_cast<std::size_t>(step + 1)];
        const float gamma = sigma_t > kGammaMin ? kGamma0 : 0.0F;
        applyAugmentation(coordinates, has_denoised ? &denoised : nullptr,
                          random.rotations[static_cast<std::size_t>(step)],
                          random.translations[static_cast<std::size_t>(step)]);
        const float t_hat = sigma_tm * (1.0F + gamma);
        const float variance =
            kNoiseScale * kNoiseScale * std::max(0.0F, t_hat * t_hat - sigma_tm * sigma_tm);
        const float noise_scale = std::sqrt(variance);
        auto noisy = addStepNoise(coordinates, random, step, noise_scale);
        const float c_in = 1.0F / std::sqrt(t_hat * t_hat + kSigmaData * kSigmaData);
        const float time_value = std::log(t_hat / kSigmaData) * 0.25F;
        const auto update = runDiffusionScore(scaledCoordinates(noisy, c_in), time_value);
        denoised = denoisedCoordinates(noisy, update, t_hat);
        has_denoised = true;
        coordinates = advanceCoordinates(std::move(noisy), denoised, atom_mask, sigma_t, t_hat);
    }
    return packCoordinates(coordinates);
}

StructureConfidence Boltz2Pipeline::runConfidence(const std::vector<float>& coordinates) {
    if (!x_pred_.copy_from_host(coordinates.data()))
        throw std::runtime_error("Boltz-2 failed to upload confidence coordinates");
    engines_.confidence->forward_device_async({});
    engines_.confidence->sync();

    const std::size_t plddt_elements = static_cast<std::size_t>(token_count_) * 50U;
    const std::size_t pae_elements =
        static_cast<std::size_t>(token_count_) * static_cast<std::size_t>(token_count_) * 64U;
    std::vector<uint16_t> plddt_logits(plddt_elements);
    std::vector<uint16_t> pae_logits(pae_elements);
    if (cudaMemcpy(plddt_logits.data(), engines_.confidence->device_ptr("plddt_logits"),
                   plddt_logits.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(pae_logits.data(), engines_.confidence->device_ptr("pae_logits"),
                   pae_logits.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("Boltz-2 failed to read confidence logits");
    }
    StructureConfidence result;
    result.plddt =
        softmaxExpectedBfloat16(plddt_logits, static_cast<std::size_t>(token_count_), 50, 1.0F);
    const auto* token_mask = reinterpret_cast<const float*>(feature("token_pad_mask").data.data());
    result.complex_plddt = maskedMean(result.plddt, token_mask, token_count_);
    result.complex_iplddt = result.complex_plddt;

    const auto* frames = reinterpret_cast<const int32_t*>(feature("frames_idx").data.data());
    const auto frame_mask =
        confidenceFrameMask(coordinates, frames, token_mask, token_count_, atom_count_);
    float valid_tokens = 0.0F;
    for (int token = 0; token < token_count_; ++token)
        valid_tokens += token_mask[token];
    const float d0 = 1.24F * std::cbrt(std::max(valid_tokens, 19.0F) - 15.0F) - 1.8F;
    const auto tm_expected = softmaxExpectedTmBfloat16(
        pae_logits, static_cast<std::size_t>(token_count_) * token_count_, 64, d0);
    result.ptm = maximumTmScore(tm_expected, frame_mask, token_mask, token_count_);
    result.confidence_score = (4.0F * result.complex_plddt + result.ptm) / 5.0F;
    result.plddt.resize(static_cast<std::size_t>(active_token_count_));
    return result;
}

std::string Boltz2Pipeline::writeStructure(const std::vector<float>& coordinates,
                                           StructureFormat format,
                                           const StructureConfidence& confidence) const {
    const auto document = nlohmann::json::parse(artifacts_.structure_metadata_json);
    const auto rows = structureRows(document, atom_count_);
    if (rows.size() != static_cast<std::size_t>(active_atom_count_))
        throw std::invalid_argument(
            "Boltz-2 structure metadata atom count differs from the active request");
    return format == StructureFormat::kPdb ? writePdb(rows, coordinates, confidence)
                                           : writeMmcif(rows, coordinates, confidence);
}

std::string Boltz2Pipeline::resultMetadata(const StructurePredictionConfig& cfg,
                                           const StructureConfidence& confidence) const {
    internal::Sha256 request_hash;
    request_hash.update(artifacts_.request);
    const nlohmann::json metadata{
        {"schema_version", 1},
        {"family", "boltz2"},
        {"boltz_version", "2.2.1"},
        {"profile",
         "tokens_" + std::to_string(token_count_) + "_atoms_" + std::to_string(atom_count_)},
        {"active_token_count", active_token_count_},
        {"active_atom_count", active_atom_count_},
        {"recycling_steps", cfg.recycling_steps},
        {"sampling_steps", cfg.sampling_steps},
        {"diffusion_samples", cfg.diffusion_samples},
        {"seed", cfg.seed},
        {"request_sha256", request_hash.hex_digest()},
        {"sample_rank", 0},
        {"confidence_score", confidence.confidence_score},
        {"complex_plddt", confidence.complex_plddt},
        {"complex_iplddt", confidence.complex_iplddt},
        {"ptm", confidence.ptm},
        {"iptm", confidence.iptm},
        {"ligand_iptm", confidence.ligand_iptm},
        {"protein_iptm", confidence.protein_iptm},
        {"plddt", confidence.plddt},
        {"chain_pair_confidence", nlohmann::json::array()},
    };
    return metadata.dump(2) + "\n";
}

StructurePredictionResult Boltz2Pipeline::predict_structure(const std::string& input,
                                                            const StructurePredictionConfig& cfg) {
    if (PreparedRequest::isPrepared(input.data(), input.size())) {
        activateRequest(PreparedRequest::parse(input.data(), input.size()));
    } else if (input != artifacts_.request) {
        throw std::invalid_argument(
            "Boltz-2 YAML must first be converted to a prepared request for this reusable "
            "profile; the embedded qualification request remains accepted for native smoke "
            "testing");
    }
    if (cfg.recycling_steps != 3 || cfg.sampling_steps != 200 || cfg.diffusion_samples != 1 ||
        cfg.seed != 42 || cfg.output_format != StructureFormat::kMmcif) {
        throw std::invalid_argument(
            "Boltz-2 bundle supports only recycling=3, sampling=200, samples=1, seed=42, mmCIF");
    }
    runTrunk();
    runConditioning();
    auto coordinates = sampleCoordinates(cfg.seed, cfg.sampling_steps);
    auto confidence = runConfidence(coordinates);
    StructurePredictionResult result;
    result.structure = writeStructure(coordinates, cfg.output_format, confidence);
    result.format = cfg.output_format;
    result.confidence = std::move(confidence);
    result.metadata_json = resultMetadata(cfg, result.confidence);
    return result;
}

} // namespace trtmc::boltz2
