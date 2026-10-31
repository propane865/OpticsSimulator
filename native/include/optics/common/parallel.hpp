// Chunked parallel dispatch for native kernels: split ``[0, n)`` across a
// process-wide persistent worker pool and re-raise the first chunk failure
// on the caller. Workers are spawned once (lazily, on first use) and live
// for the process lifetime, so batch calls pay no per-call thread startup;
// the deliberately leaked singleton also keeps shutdown order irrelevant.
// Jobs are serialized (one dispatch at a time); chunks within a job fan out
// across all workers. Kernels run without the GIL and never re-enter
// ``run_chunks`` from inside a job.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace optics::native {

// Hardware thread count clamped to >= 1: the one sizing source every worker
// count derivation (pool, chunk fan-out, candidate stages) goes through.
inline std::size_t hardware_threads() {
    unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : static_cast<std::size_t>(hc);
}

namespace pool_detail {

// When true on the calling thread, ``run_chunks`` executes its kernel inline
// instead of dispatching to the shared pool. Set by ``ScopedSerialTrace`` so a
// candidate-concurrency worker thread (which is NOT a pool worker and must not
// re-enter the busy-serialized pool) traces its ray fan serially on itself.
inline bool& serial_mode() {
    static thread_local bool value = false;
    return value;
}

class WorkerPool {
  public:
    // Runs ``fn`` over ``[0, n)`` in chunks of ``chunk``.
    template <typename Fn>
    void execute(std::size_t n, std::size_t chunk, const Fn& fn) {
        if (n == 0) {
            return;
        }
        const std::size_t n_threads = hardware_threads();
        const std::size_t n_chunks = std::min<std::size_t>((n + chunk - 1) / chunk, n_threads);
        {
            std::unique_lock<std::mutex> lock(mutex_);
            idle_.wait(lock, [this] { return !busy_; });
            busy_ = true;
            fn_ = static_cast<const void*>(&fn);
            invoke_ = [](const void* cookie, std::size_t lo, std::size_t hi) {
                (*static_cast<const Fn*>(cookie))(lo, hi);
            };
            n_ = n;
            chunk_ = chunk;
            chunks_ = n_chunks;
            next_ = 0;
            completed_ = 0;
            errors_.assign(n_chunks, std::exception_ptr{});
            ensure_workers(n_threads);
        }
        cv_.notify_all();
        std::vector<std::exception_ptr> errors;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_.wait(lock, [this] { return completed_ >= chunks_; });
            errors.swap(errors_);
            busy_ = false;
        }
        idle_.notify_all();
        for (const std::exception_ptr& exc : errors) {
            if (exc) {
                std::rethrow_exception(exc);
            }
        }
    }

    static WorkerPool& instance() {
        // Leaked on purpose: worker threads outlive static destruction so
        // interpreter/module teardown can never race a joining worker.
        static WorkerPool* pool = new WorkerPool();
        return *pool;
    }

  private:
    void ensure_workers(std::size_t want) {
        while (workers_.size() < want) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    void worker_loop() {
        std::size_t take = 0;
        for (;;) {
            std::size_t n = 0;
            std::size_t chunk = 0;
            std::size_t chunks = 0;
            const void* cookie = nullptr;
            void (*invoke)(const void*, std::size_t, std::size_t) = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return next_ < chunks_; });
                take = next_++;
                n = n_;
                chunk = chunk_;
                chunks = chunks_;
                cookie = fn_;
                invoke = invoke_;
            }
            const std::size_t lo = take * chunk;
            const std::size_t hi = (n - lo < chunk) ? n : lo + chunk;
            std::exception_ptr exc;
            try {
                invoke(cookie, lo, hi);
            } catch (...) {
                exc = std::current_exception();
            }
            std::size_t done = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (exc) {
                    errors_[take] = exc;
                }
                done = ++completed_;
            }
            if (done >= chunks) {
                done_.notify_all();
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;     // work available (next_ < chunks_)
    std::condition_variable done_;   // job complete (completed_ >= chunks_)
    std::condition_variable idle_;   // pool free for the next job
    bool busy_ = false;
    std::size_t n_ = 0;
    std::size_t chunk_ = 1;
    std::size_t chunks_ = 0;  // upper bound; next_ >= chunks_ means no work
    std::size_t next_ = 0;
    std::size_t completed_ = 0;
    const void* fn_ = nullptr;
    void (*invoke_)(const void*, std::size_t, std::size_t) = nullptr;
    std::vector<std::exception_ptr> errors_;
    std::vector<std::thread> workers_;
};

}  // namespace pool_detail

// RAII: forces nested ``run_chunks`` calls on this thread to run inline rather
// than dispatch to the shared pool, saving/restoring the prior mode so it
// nests. Used around per-candidate ray tracing in candidate concurrency.
class ScopedSerialTrace {
  public:
    ScopedSerialTrace() : prev_(pool_detail::serial_mode()) { pool_detail::serial_mode() = true; }
    ~ScopedSerialTrace() { pool_detail::serial_mode() = prev_; }
    ScopedSerialTrace(const ScopedSerialTrace&) = delete;
    ScopedSerialTrace& operator=(const ScopedSerialTrace&) = delete;

  private:
    bool prev_;
};

template <typename Fn>
void run_chunks(std::size_t n, Fn&& fn) {
    if (n == 0) {
        return;
    }
    if (pool_detail::serial_mode()) {
        fn(std::size_t{0}, n);
        return;
    }
    const std::size_t nt = std::max<std::size_t>(1, std::min<std::size_t>(hardware_threads(), n));
    const std::size_t chunk = (n + nt - 1) / nt;
    pool_detail::WorkerPool::instance().execute(n, chunk, fn);
}

// One-shot fan-out that bypasses the shared pool and spawns fresh threads per
// call: ``[0, n)`` split across ``n_threads`` contiguous ranges, ``work(t, lo, hi)``
// run on each thread, all threads joined, and the first worker exception
// rethrown on the caller. For stages where every worker needs private scratch
// and the caller already sized the thread count (a busy pool would serialize
// these behind unrelated dispatch).
template <typename Work>
void run_chunk_threads(std::size_t n, std::size_t n_threads, Work&& work) {
    std::vector<std::exception_ptr> errors(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (std::size_t t = 0; t < n_threads; ++t) {
        const std::size_t lo = n * t / n_threads;
        const std::size_t hi = n * (t + 1) / n_threads;
        threads.emplace_back([&, t, lo, hi] {
            try {
                work(t, lo, hi);
            } catch (...) {
                errors[t] = std::current_exception();
            }
        });
    }
    for (std::thread& th : threads) {
        th.join();
    }
    for (const std::exception_ptr& exc : errors) {
        if (exc) {
            std::rethrow_exception(exc);
        }
    }
}

}  // namespace optics::native
