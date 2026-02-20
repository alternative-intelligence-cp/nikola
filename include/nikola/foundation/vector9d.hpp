/**
 * @file include/nikola/foundation/vector9d.hpp
 * @brief 9-dimensional vector for toroidal phase space coordinates.
 *
 * Core mathematical primitive for Nikola's 9D toroidal topology.
 * Represents points in phase space (3D position × 3D momentum × 3D spin).
 *
 * Phase 1 Foundation - Required for all physics operations.
 */

#pragma once

#include <array>
#include <cmath>
#include <initializer_list>
#include <ostream>

namespace nikola::foundation {

/**
 * @class Vector9D
 * @brief 9-dimensional vector with standard operations.
 *
 * Coordinates represent:
 * - [0,1,2]: Spatial position (x, y, z)
 * - [3,4,5]: Momentum (px, py, pz)
 * - [6,7,8]: Spin orientation (sx, sy, sz)
 *
 * All dimensions have toroidal topology (periodic boundary conditions).
 */
class Vector9D {
public:
    static constexpr size_t DIMENSIONS = 9;
    using StorageType = std::array<double, DIMENSIONS>;

    /**
     * @brief Default constructor - zero vector.
     */
    constexpr Vector9D() noexcept : data_{} {}

    /**
     * @brief Construct from initializer list.
     * @param values Up to 9 values (remaining filled with zeros)
     */
    constexpr Vector9D(std::initializer_list<double> values) noexcept : data_{} {
        size_t i = 0;
        for (double val : values) {
            if (i >= DIMENSIONS) break;
            data_[i++] = val;
        }
    }

    /**
     * @brief Construct from raw array.
     */
    explicit constexpr Vector9D(const StorageType& data) noexcept : data_(data) {}

    // Element access
    constexpr double& operator[](size_t i) noexcept { return data_[i]; }
    constexpr const double& operator[](size_t i) const noexcept { return data_[i]; }

    constexpr double& at(size_t i) { 
        if (i >= DIMENSIONS) throw std::out_of_range("Vector9D index out of range");
        return data_[i]; 
    }
    constexpr const double& at(size_t i) const { 
        if (i >= DIMENSIONS) throw std::out_of_range("Vector9D index out of range");
        return data_[i]; 
    }

    // Arithmetic operations
    constexpr Vector9D& operator+=(const Vector9D& other) noexcept {
        for (size_t i = 0; i < DIMENSIONS; ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    constexpr Vector9D& operator-=(const Vector9D& other) noexcept {
        for (size_t i = 0; i < DIMENSIONS; ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    constexpr Vector9D& operator*=(double scalar) noexcept {
        for (size_t i = 0; i < DIMENSIONS; ++i) {
            data_[i] *= scalar;
        }
        return *this;
    }

    constexpr Vector9D& operator/=(double scalar) noexcept {
        for (size_t i = 0; i < DIMENSIONS; ++i) {
            data_[i] /= scalar;
        }
        return *this;
    }

    // Vector operations
    
    /**
     * @brief Euclidean norm (L2).
     */
    double norm() const noexcept {
        double sum = 0.0;
        for (size_t i = 0; i < DIMENSIONS; ++i) {
            sum += data_[i] * data_[i];
        }
        return std::sqrt(sum);
    }

    /**
     * @brief Squared norm (avoids sqrt).
     */
    constexpr double norm_squared() const noexcept {
        double sum = 0.0;
        for (size_t i = 0; i < DIMENSIONS; ++i) {
            sum += data_[i] * data_[i];
        }
        return sum;
    }

    /**
     * @brief Normalize to unit length.
     */
    Vector9D& normalize() noexcept {
        double n = norm();
        if (n > 1e-10) {
            *this /= n;
        }
        return *this;
    }

    /**
     * @brief Get normalized copy.
     */
    Vector9D normalized() const noexcept {
        Vector9D result(*this);
        return result.normalize();
    }

    /**
     * @brief Dot product.
     */
    constexpr double dot(const Vector9D& other) const noexcept {
        double sum = 0.0;
        for (size_t i = 0; i < DIMENSIONS; ++i) {
            sum += data_[i] * other.data_[i];
        }
        return sum;
    }

    /**
     * @brief Check if all components are finite.
     */
    bool is_finite() const noexcept {
        for (size_t i = 0; i < DIMENSIONS; ++i) {
            if (!std::isfinite(data_[i])) return false;
        }
        return true;
    }

    /**
     * @brief Apply toroidal wrap (modulo operation for periodic boundaries).
     * @param period Period for each dimension
     */
    Vector9D& wrap(double period = 1.0) noexcept {
        for (size_t i = 0; i < DIMENSIONS; ++i) {
            data_[i] = std::fmod(data_[i], period);
            if (data_[i] < 0.0) data_[i] += period;
        }
        return *this;
    }

    // Data access
    const StorageType& data() const noexcept { return data_; }
    StorageType& data() noexcept { return data_; }

private:
    StorageType data_;
};

// Binary operators (non-member functions)

inline constexpr Vector9D operator+(Vector9D lhs, const Vector9D& rhs) noexcept {
    return lhs += rhs;
}

inline constexpr Vector9D operator-(Vector9D lhs, const Vector9D& rhs) noexcept {
    return lhs -= rhs;
}

inline constexpr Vector9D operator*(Vector9D vec, double scalar) noexcept {
    return vec *= scalar;
}

inline constexpr Vector9D operator*(double scalar, Vector9D vec) noexcept {
    return vec *= scalar;
}

inline constexpr Vector9D operator/(Vector9D vec, double scalar) noexcept {
    return vec /= scalar;
}

// Comparison operators

inline constexpr bool operator==(const Vector9D& lhs, const Vector9D& rhs) noexcept {
    for (size_t i = 0; i < Vector9D::DIMENSIONS; ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

inline constexpr bool operator!=(const Vector9D& lhs, const Vector9D& rhs) noexcept {
    return !(lhs == rhs);
}

// Stream output

inline std::ostream& operator<<(std::ostream& os, const Vector9D& vec) {
    os << "[";
    for (size_t i = 0; i < Vector9D::DIMENSIONS; ++i) {
        os << vec[i];
        if (i < Vector9D::DIMENSIONS - 1) os << ", ";
    }
    os << "]";
    return os;
}

// Distance functions

/**
 * @brief Euclidean distance in 9D space.
 */
inline double distance(const Vector9D& a, const Vector9D& b) noexcept {
    return (a - b).norm();
}

/**
 * @brief Toroidal distance (accounts for periodic wrap).
 * @param period Period for wraparound
 */
inline double toroidal_distance(const Vector9D& a, const Vector9D& b, double period = 1.0) noexcept {
    Vector9D diff = a - b;
    for (size_t i = 0; i < Vector9D::DIMENSIONS; ++i) {
        double& d = diff[i];
        d = std::abs(d);
        if (d > period / 2.0) {
            d = period - d;  // Shorter path wraps around
        }
    }
    return diff.norm();
}

} // namespace nikola::foundation
