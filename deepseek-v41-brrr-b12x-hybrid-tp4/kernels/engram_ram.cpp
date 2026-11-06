#include <torch/extension.h>
#include <fcntl.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

constexpr int64_t ROW = 256;
constexpr size_t CHUNK = 8 * 1024 * 1024;
constexpr auto SPIN_WINDOW = std::chrono::milliseconds(50);
constexpr auto IDLE_POLL = std::chrono::microseconds(500);

struct Store {
    uint8_t* table = nullptr;
    int64_t count = 0, bytes = 0, capacity = 0;
    const int64_t* ids;
    uint8_t* rows;
    int64_t* ids_ready;
    int64_t* rows_ready;
    std::vector<torch::Tensor> keepalive;
};

// Pollers and their tensor buffers deliberately live until process exit.
static auto* stores = new std::vector<std::unique_ptr<Store>>;

static void copy_rows(const Store* store, const int64_t* ids, uint8_t* out, int64_t count) {
    for (int64_t row = 0; row < count; ++row) {
        const int64_t id = ids[row];
        if (id < 0) {
            std::memset(out + row * ROW, 0, ROW);
        } else {
            TORCH_CHECK(id < store->count, "RAM Engram row out of bounds: ", id);
            std::memcpy(out + row * ROW, store->table + id * ROW, ROW);
        }
    }
}

static void poller(Store* store) {
    int64_t seen = 0;
    auto last = std::chrono::steady_clock::now();
    for (;;) {
        const int64_t value = __atomic_load_n(store->ids_ready, __ATOMIC_ACQUIRE);
        if (value == seen) {
            if (std::chrono::steady_clock::now() - last > SPIN_WINDOW) std::this_thread::sleep_for(IDLE_POLL);
            else _mm_pause();
            continue;
        }
        seen = value;
        const int64_t count = value & 0xffffffff;
        TORCH_CHECK(count <= store->capacity, "RAM Engram staging capacity exceeded");
        copy_rows(store, store->ids, store->rows, count);
        __atomic_store_n(store->rows_ready, value >> 32, __ATOMIC_RELEASE);
        last = std::chrono::steady_clock::now();
    }
}

static void preload(Store* store, const std::string& path, int64_t base) {
    int allocation_lock = -1;
    if (const char* lock_path = std::getenv("BRRR_RAM_LOCKFILE")) {
        allocation_lock = open(lock_path, O_CREAT | O_RDWR, 0600);
        TORCH_CHECK(allocation_lock >= 0 && flock(allocation_lock, LOCK_EX) == 0, "Cannot acquire RAM allocation guard");
        std::ifstream meminfo("/proc/meminfo");
        std::string key, units;
        int64_t value, available = -1;
        while (meminfo >> key >> value >> units) if (key == "MemAvailable:") { available = value * 1024; break; }
        TORCH_CHECK(available >= store->bytes + (int64_t(2) << 30), "Not enough RAM for the next fully locked Engram shard");
    }
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    TORCH_CHECK(fd >= 0, "Cannot open RAM Engram source for direct preload: ", path);
    struct stat status;
    TORCH_CHECK(fstat(fd, &status) == 0 && base >= 0 && base + store->bytes <= status.st_size, "RAM Engram source range is invalid");
    store->table = static_cast<uint8_t*>(mmap(nullptr, store->bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    TORCH_CHECK(store->table != MAP_FAILED, "RAM Engram mmap failed: ", strerror(errno));
    TORCH_CHECK(mlock(store->table, store->bytes) == 0, "Cannot lock full Engram table in RAM: ", strerror(errno));
    madvise(store->table, store->bytes, MADV_DONTDUMP);
    void* allocation = nullptr;
    TORCH_CHECK(posix_memalign(&allocation, 4096, CHUNK) == 0, "RAM Engram preload buffer allocation failed");
    auto* buffer = static_cast<uint8_t*>(allocation);
    int64_t done = 0;
    while (done < store->bytes) {
        const int64_t offset = base + done;
        const int64_t aligned = offset & ~int64_t(4095);
        const int64_t skip = offset - aligned;
        const size_t payload = std::min<int64_t>(store->bytes - done, CHUNK - skip);
        const size_t requested = (payload + skip + 4095) & ~size_t(4095);
        ssize_t got;
        do { got = pread(fd, buffer, requested, aligned); } while (got < 0 && errno == EINTR);
        TORCH_CHECK(got >= int64_t(payload + skip), "Short direct RAM Engram preload read: ", got);
        std::memcpy(store->table + done, buffer + skip, payload);
        done += payload;
    }
    free(allocation);
    close(fd);
    TORCH_CHECK(mprotect(store->table, store->bytes, PROT_READ) == 0, "Cannot protect resident Engram table");
    if (allocation_lock >= 0) close(allocation_lock);
}

static int64_t resident_bytes(const Store* store) {
    const int64_t page = sysconf(_SC_PAGESIZE);
    const int64_t pages = (store->bytes + page - 1) / page;
    std::vector<unsigned char> state(pages);
    TORCH_CHECK(mincore(store->table, store->bytes, state.data()) == 0, "Cannot verify Engram residency");
    int64_t resident = 0;
    for (auto value : state) resident += (value & 1) != 0;
    return std::min(resident * page, store->bytes);
}

int64_t create_store(std::string path, int64_t base, int64_t count, torch::Tensor ids, torch::Tensor rows,
                     torch::Tensor ids_ready, torch::Tensor rows_ready) {
    TORCH_CHECK(count > 0 && count < INT64_MAX / ROW, "Invalid Engram row count");
    TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type() == torch::kInt64 && ids.is_contiguous());
    TORCH_CHECK(rows.device().is_cpu() && rows.scalar_type() == torch::kUInt8 && rows.is_contiguous());
    TORCH_CHECK(ids_ready.device().is_cpu() && rows_ready.device().is_cpu());
    TORCH_CHECK(ids_ready.scalar_type() == torch::kInt64 && rows_ready.scalar_type() == torch::kInt64);
    TORCH_CHECK(ids_ready.numel() == 1 && rows_ready.numel() == 1 && rows.numel() >= ids.numel() * ROW);
    auto store = std::make_unique<Store>();
    store->count = count;
    store->bytes = count * ROW;
    store->capacity = ids.numel();
    store->keepalive = {ids, rows, ids_ready, rows_ready};
    store->ids = ids.data_ptr<int64_t>();
    store->rows = rows.data_ptr<uint8_t>();
    store->ids_ready = ids_ready.data_ptr<int64_t>();
    store->rows_ready = rows_ready.data_ptr<int64_t>();
    preload(store.get(), path, base);
    TORCH_CHECK(resident_bytes(store.get()) == store->bytes, "Full Engram table is not resident");
    std::thread(poller, store.get()).detach();
    stores->push_back(std::move(store));
    return static_cast<int64_t>(stores->size() - 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("ROW") = ROW;
    m.def("create_store", &create_store);
    m.def("store_info", [](int64_t handle) {
        const auto& store = stores->at(handle);
        pybind11::dict value;
        value["bytes"] = store->bytes;
        value["resident_bytes"] = resident_bytes(store.get());
        value["rows"] = store->count;
        value["locked"] = true;
        value["source_fd_closed"] = true;
        return value;
    });
    m.def("test_rows", [](int64_t handle, torch::Tensor ids) {
        TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type() == torch::kInt64 && ids.is_contiguous());
        auto output = torch::empty({ids.numel(), ROW}, torch::TensorOptions().dtype(torch::kUInt8));
        copy_rows(stores->at(handle).get(), ids.data_ptr<int64_t>(), output.data_ptr<uint8_t>(), ids.numel());
        return output;
    });
}
