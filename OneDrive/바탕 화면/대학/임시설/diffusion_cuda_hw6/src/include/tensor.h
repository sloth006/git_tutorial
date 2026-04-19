#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

enum class TensorDType : std::uint8_t { Float32 = 0, Float16 = 1 };

// Unified-memory arena: no per-tensor new/malloc. Storage is padded to a multiple of 64
// float elements (256 bytes) for memory-controller alignment on Ampere.
class Tensor {
private:
    std::size_t _offset_bytes = 0;
    int _logical_elements = 0;
    int _storage_elements = 0;
    std::vector<int> _shape;
    TensorDType _dtype = TensorDType::Float32;

public:
    static void arena_init(std::size_t bytes);
    static void arena_set_perm_watermark();
    static void arena_reset_scratch();

    // When true, Tensor::from_shape allocates from the scratch arena (for activations).
    // Keep false during weight load / persistent buffers (default).
    static void set_from_shape_uses_scratch(bool v);
    static bool from_shape_uses_scratch();
    static std::size_t arena_checkpoint_scratch();
    static void arena_rewind_scratch_to(std::size_t ck);

    static std::size_t align_storage_elements(int count);
    static std::size_t arena_alloc_perm(std::size_t bytes, std::size_t align = 256);
    static std::size_t arena_alloc_scratch(std::size_t bytes, std::size_t align = 256);

    Tensor() = default;
    Tensor(const Tensor& x);
    Tensor(Tensor&& o) noexcept;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&& o) noexcept;

    explicit Tensor(const std::vector<int>& shape_vec);
    Tensor(const std::vector<int>& shape_vec, float constant);

    float& operator[](int idx);
    const float& operator[](int idx) const;

    void* raw_ptr() const;
    float* fp32() const;

    int elem_size_bytes() const { return (_dtype == TensorDType::Float16) ? 2 : 4; }
    TensorDType dtype() const { return _dtype; }
    void set_dtype(TensorDType d) { _dtype = d; }

    int dim() const;
    int size() const;
    int storage_size() const { return _storage_elements; }
    const std::vector<int>& shape() const { return _shape; }
    int shape_dim(int idx) const { return _shape[idx]; }

    static Tensor from_shape(const std::vector<int>& shape_vec, TensorDType dt = TensorDType::Float32);
    static Tensor scratch_from_shape(const std::vector<int>& shape_vec, TensorDType dt = TensorDType::Float32);
    static Tensor like(const Tensor& x);
    static Tensor copy(const Tensor& x);
    void copy_from(const Tensor& src);

    static Tensor randn(const std::vector<int>& shape);
    static void set_random_seed(unsigned int seed);
    static void use_random_device();

    static Tensor load_npy(const std::string& filename);
    void dump_npy(const std::string& filename) const;

    void reshape(int idx, int val);
    bool allclose(const Tensor& other, float rtol = 1e-5f, float atol = 1e-8f) const;

    Tensor sin() const;
    Tensor cos() const;
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log() const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
};

#endif
