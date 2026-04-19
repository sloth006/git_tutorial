#include <tensor.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {

std::byte* g_arena = nullptr;
std::size_t g_arena_bytes = 0;
std::size_t g_perm_end = 0;
// Scratch is allocated from the high end of the arena downward so it never shares
// byte ranges with the upward-growing permanent bump (avoids overlap when perm
// continues after the first scratch alloc, e.g. DiffusionModel ctor + UNet forward).
std::size_t g_stack_top = 0;
bool g_watermark_set = false;
bool g_from_shape_scratch = false;

inline std::size_t align_up(std::size_t x, std::size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

inline std::size_t align_down(std::size_t x, std::size_t a) {
    return (x / a) * a;
}

void ensure_arena(std::size_t min_extra = 0) {
    if (g_arena) {
        return;
    }
    const std::size_t default_bytes = 3ull * 1024ull * 1024ull * 1024ull;
    std::size_t bytes = std::max(default_bytes, static_cast<std::size_t>(min_extra + (512ull << 20)));
    void* p = nullptr;
    if (cudaMallocManaged(&p, bytes) != cudaSuccess) {
        throw std::runtime_error("cudaMallocManaged failed for Tensor arena");
    }
    g_arena = static_cast<std::byte*>(p);
    g_arena_bytes = bytes;
    g_perm_end = 0;
    g_stack_top = align_down(bytes, 256);
    g_watermark_set = false;
}

} // namespace

void Tensor::arena_init(std::size_t bytes) {
    if (g_arena) {
        cudaFree(g_arena);
        g_arena = nullptr;
    }
    void* p = nullptr;
    if (cudaMallocManaged(&p, bytes) != cudaSuccess) {
        throw std::runtime_error("cudaMallocManaged failed (arena_init)");
    }
    g_arena = static_cast<std::byte*>(p);
    g_arena_bytes = bytes;
    g_perm_end = 0;
    g_stack_top = align_down(bytes, 256);
    g_watermark_set = false;
}

void Tensor::arena_set_perm_watermark() {
    ensure_arena();
    if (g_stack_top == 0) {
        g_stack_top = align_down(g_arena_bytes, 256);
    }
    g_watermark_set = true;
}

void Tensor::arena_reset_scratch() {
    if (!g_watermark_set) {
        arena_set_perm_watermark();
    }
    g_stack_top = align_down(g_arena_bytes, 256);
}

std::size_t Tensor::arena_checkpoint_scratch() {
    if (!g_watermark_set) {
        arena_set_perm_watermark();
    }
    return g_stack_top;
}

void Tensor::set_from_shape_uses_scratch(bool v) { g_from_shape_scratch = v; }

bool Tensor::from_shape_uses_scratch() { return g_from_shape_scratch; }

void Tensor::arena_rewind_scratch_to(std::size_t ck) {
    if (!g_watermark_set) {
        arena_set_perm_watermark();
    }
    const std::size_t perm_limit = align_up(g_perm_end, 256);
    const std::size_t arena_top = align_down(g_arena_bytes, 256);
    if (ck < perm_limit || ck > arena_top) {
        throw std::runtime_error("arena_rewind_scratch_to: invalid checkpoint");
    }
    if (ck < g_stack_top) {
        throw std::runtime_error("arena_rewind_scratch_to: cannot rewind past live scratch");
    }
    g_stack_top = ck;
}

std::size_t Tensor::align_storage_elements(int count) {
    if (count <= 0) {
        return 64;
    }
    return ((static_cast<std::size_t>(count) + 63u) / 64u) * 64u;
}

std::size_t Tensor::arena_alloc_perm(std::size_t size_bytes, std::size_t align) {
    ensure_arena(size_bytes + align);
    if (g_stack_top == 0) {
        g_stack_top = align_down(g_arena_bytes, 256);
    }
    std::size_t off = align_up(g_perm_end, align);
    const std::size_t new_end = off + size_bytes;
    if (new_end > g_arena_bytes) {
        throw std::runtime_error("Tensor arena (permanent) OOM");
    }
    if (align_up(new_end, 256) > g_stack_top) {
        throw std::runtime_error("Tensor arena (permanent) OOM: would cross active scratch region");
    }
    g_perm_end = new_end;
    return off;
}

std::size_t Tensor::arena_alloc_scratch(std::size_t size_bytes, std::size_t align) {
    ensure_arena();
    if (!g_watermark_set) {
        arena_set_perm_watermark();
    }
    if (g_stack_top == 0) {
        g_stack_top = align_down(g_arena_bytes, 256);
    }
    const std::size_t sz = align_up(size_bytes, align);
    const std::size_t perm_limit = align_up(g_perm_end, align);
    if (g_stack_top < perm_limit || g_stack_top - perm_limit < sz) {
        throw std::runtime_error("Tensor arena (scratch) OOM: not enough space above permanent tail");
    }
    std::size_t off = align_down(g_stack_top - sz, align);
    if (off < perm_limit) {
        throw std::runtime_error("Tensor arena (scratch) OOM: alignment pushed block into permanent tail");
    }
    g_stack_top = off;
    return off;
}

Tensor::Tensor(const Tensor& x) {
    _shape = x._shape;
    _dtype = x._dtype;
    _logical_elements = x._logical_elements;
    _storage_elements = x._storage_elements;
    const int esz = x.elem_size_bytes();
    _offset_bytes = arena_alloc_perm(static_cast<std::size_t>(_storage_elements) * static_cast<std::size_t>(esz), 256);
    std::memcpy(raw_ptr(), x.raw_ptr(),
                static_cast<std::size_t>(_storage_elements) * static_cast<std::size_t>(esz));
}

Tensor::Tensor(Tensor&& o) noexcept
    : _offset_bytes(o._offset_bytes),
      _logical_elements(o._logical_elements),
      _storage_elements(o._storage_elements),
      _shape(std::move(o._shape)),
      _dtype(o._dtype) {
    o._offset_bytes = 0;
    o._logical_elements = 0;
    o._storage_elements = 0;
    o._dtype = TensorDType::Float32;
}

Tensor& Tensor::operator=(Tensor&& o) noexcept {
    if (this != &o) {
        _offset_bytes = o._offset_bytes;
        _logical_elements = o._logical_elements;
        _storage_elements = o._storage_elements;
        _shape = std::move(o._shape);
        _dtype = o._dtype;
        o._offset_bytes = 0;
        o._logical_elements = 0;
        o._storage_elements = 0;
        o._dtype = TensorDType::Float32;
    }
    return *this;
}

Tensor::Tensor(const std::vector<int>& shape_vec) {
    _shape = shape_vec;
    _dtype = TensorDType::Float32;
    _logical_elements = 1;
    for (int s : _shape) {
        _logical_elements *= s;
    }
    _storage_elements = static_cast<int>(align_storage_elements(_logical_elements));
    _offset_bytes = arena_alloc_perm(static_cast<std::size_t>(_storage_elements) * sizeof(float), 256);
}

Tensor::Tensor(const std::vector<int>& shape_vec, float constant) {
    _shape = shape_vec;
    _dtype = TensorDType::Float32;
    _logical_elements = 1;
    for (int s : _shape) {
        _logical_elements *= s;
    }
    _storage_elements = static_cast<int>(align_storage_elements(_logical_elements));
    _offset_bytes = arena_alloc_perm(static_cast<std::size_t>(_storage_elements) * sizeof(float), 256);
    std::fill_n(fp32(), _storage_elements, constant);
}

float& Tensor::operator[](int idx) {
    if (_dtype != TensorDType::Float32) {
        throw std::runtime_error("Tensor::operator[] Float32 host path only");
    }
    if (idx < 0 || idx >= _logical_elements) {
        throw std::runtime_error("Tensor::operator[] out of range");
    }
    return fp32()[idx];
}

const float& Tensor::operator[](int idx) const {
    if (_dtype != TensorDType::Float32) {
        throw std::runtime_error("Tensor::operator[] Float32 host path only");
    }
    if (idx < 0 || idx >= _logical_elements) {
        throw std::runtime_error("Tensor::operator[] out of range");
    }
    return fp32()[idx];
}

void* Tensor::raw_ptr() const {
    if (_storage_elements == 0) {
        return nullptr;
    }
    if (!g_arena) {
        throw std::runtime_error("Tensor arena not initialized");
    }
    return g_arena + _offset_bytes;
}

float* Tensor::fp32() const { return reinterpret_cast<float*>(raw_ptr()); }

int Tensor::dim() const { return static_cast<int>(_shape.size()); }

int Tensor::size() const { return _logical_elements; }

Tensor Tensor::from_shape(const std::vector<int>& shape_vec, TensorDType dt) {
    if (g_from_shape_scratch) {
        return scratch_from_shape(shape_vec, dt);
    }
    Tensor t;
    t._shape = shape_vec;
    t._dtype = dt;
    t._logical_elements = 1;
    for (int s : t._shape) {
        t._logical_elements *= s;
    }
    const int esz = (dt == TensorDType::Float16) ? 2 : 4;
    t._storage_elements = static_cast<int>(align_storage_elements(t._logical_elements));
    t._offset_bytes = arena_alloc_perm(static_cast<std::size_t>(t._storage_elements) * static_cast<std::size_t>(esz), 256);
    return t;
}

Tensor Tensor::scratch_from_shape(const std::vector<int>& shape_vec, TensorDType dt) {
    Tensor t;
    t._shape = shape_vec;
    t._dtype = dt;
    t._logical_elements = 1;
    for (int s : t._shape) {
        t._logical_elements *= s;
    }
    const int esz = (dt == TensorDType::Float16) ? 2 : 4;
    t._storage_elements = static_cast<int>(align_storage_elements(t._logical_elements));
    t._offset_bytes = arena_alloc_scratch(static_cast<std::size_t>(t._storage_elements) * static_cast<std::size_t>(esz), 256);
    return t;
}

Tensor Tensor::like(const Tensor& x) { return scratch_from_shape(x._shape, x._dtype); }

Tensor Tensor::copy(const Tensor& x) {
    Tensor t = scratch_from_shape(x._shape, x._dtype);
    std::memcpy(t.raw_ptr(), x.raw_ptr(),
                static_cast<std::size_t>(x._storage_elements) * static_cast<std::size_t>(x.elem_size_bytes()));
    return t;
}

void Tensor::copy_from(const Tensor& src) {
    if (_shape != src._shape) {
        throw std::runtime_error("Tensor::copy_from shape mismatch");
    }
    std::memcpy(raw_ptr(), src.raw_ptr(),
                static_cast<std::size_t>(_storage_elements) * static_cast<std::size_t>(elem_size_bytes()));
}

static std::mt19937 global_rng;
static bool rng_initialized = false;

void Tensor::set_random_seed(unsigned int seed) {
    global_rng.seed(seed);
    rng_initialized = true;
}

void Tensor::use_random_device() {
    std::random_device rd;
    global_rng.seed(rd());
    rng_initialized = true;
}

Tensor Tensor::randn(const std::vector<int>& shape) {
    Tensor out = from_shape(shape);
    int N = out.size();
    if (!rng_initialized) {
        std::random_device rd;
        global_rng.seed(rd());
        rng_initialized = true;
    }
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; i++) {
        out[i] = dist(global_rng);
    }
    return out;
}

Tensor Tensor::load_npy(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    char magic[6];
    file.read(magic, 6);
    if (!file || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("Invalid NPY file format");
    }
    unsigned char version_major, version_minor;
    file.read(reinterpret_cast<char*>(&version_major), 1);
    file.read(reinterpret_cast<char*>(&version_minor), 1);
    uint32_t header_len = 0;
    if (version_major == 1 && version_minor == 0) {
        uint16_t h;
        file.read(reinterpret_cast<char*>(&h), 2);
        header_len = h;
    } else if (version_major == 2 && version_minor == 0) {
        file.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("Unsupported NPY version");
    }
    std::string header(header_len, '\0');
    file.read(&header[0], header_len);
    std::regex shape_regex(R"('shape':\s*\(([^)]*)\))");
    std::smatch match;
    if (!std::regex_search(header, match, shape_regex)) {
        throw std::runtime_error("Could not parse shape from NPY header");
    }
    std::string shape_str = match[1].str();
    std::vector<int> shape_v;
    std::stringstream ss(shape_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(0, item.find_first_not_of(" \t\n\r"));
        item.erase(item.find_last_not_of(" \t\n\r") + 1);
        if (!item.empty()) {
            shape_v.push_back(std::stoi(item));
        }
    }
    Tensor tensor(shape_v);
    int total_size = tensor.size();
    file.read(reinterpret_cast<char*>(tensor.fp32()), total_size * sizeof(float));
    if (!file) {
        throw std::runtime_error("Failed to read data from NPY file");
    }
    return tensor;
}

void Tensor::dump_npy(const std::string& filename) const {
    if (_dtype != TensorDType::Float32) {
        throw std::runtime_error("dump_npy: Float32 only");
    }
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < _shape.size(); i++) {
        header += std::to_string(_shape[i]);
        if (i < _shape.size() - 1) {
            header += ", ";
        }
    }
    header += ",), }";
    while ((10 + header.size() + 1) % 64 != 0) {
        header += ' ';
    }
    header += '\n';
    uint16_t header_len = static_cast<uint16_t>(header.size());
    file.write("\x93NUMPY", 6);
    file.put(1);
    file.put(0);
    file.write(reinterpret_cast<const char*>(&header_len), sizeof(uint16_t));
    file.write(header.c_str(), header.size());
    file.write(reinterpret_cast<const char*>(fp32()), static_cast<std::streamsize>(size() * sizeof(float)));
}

void Tensor::reshape(int idx, int val) {
    if (idx < static_cast<int>(_shape.size())) {
        _shape[idx] = val;
    }
}

bool Tensor::allclose(const Tensor& other, float rtol, float atol) const {
    if (_shape != other._shape) {
        return false;
    }
    for (int i = 0; i < size(); i++) {
        float a = (*this)[i];
        float b = other[i];
        float diff = std::abs(a - b);
        if (diff > atol + rtol * std::abs(b)) {
            return false;
        }
    }
    return true;
}

Tensor Tensor::sin() const {
    Tensor out = like(*this);
    for (int i = 0; i < size(); i++) {
        out[i] = std::sin((*this)[i]);
    }
    return out;
}

Tensor Tensor::cos() const {
    Tensor out = like(*this);
    for (int i = 0; i < size(); i++) {
        out[i] = std::cos((*this)[i]);
    }
    return out;
}

Tensor Tensor::sqrt() const {
    Tensor out = like(*this);
    for (int i = 0; i < size(); i++) {
        out[i] = std::sqrt((*this)[i]);
    }
    return out;
}

Tensor Tensor::exp() const {
    Tensor out = like(*this);
    for (int i = 0; i < size(); i++) {
        out[i] = std::exp((*this)[i]);
    }
    return out;
}

Tensor Tensor::log() const {
    Tensor out = like(*this);
    for (int i = 0; i < size(); i++) {
        out[i] = std::log((*this)[i]);
    }
    return out;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "Tensor(shape=[";
    for (int i = 0; i < t.dim(); i++) {
        os << t.shape()[i];
        if (i + 1 < t.dim()) {
            os << ", ";
        }
    }
    os << "], data=[";
    int m = std::min(10, t.size());
    for (int i = 0; i < m; i++) {
        os << t.fp32()[i];
        if (i + 1 < m) {
            os << ", ";
        }
    }
    if (t.size() > m) {
        os << ", ...";
    }
    os << "])";
    return os;
}
