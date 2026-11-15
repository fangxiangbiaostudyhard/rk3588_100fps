#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <atomic>
#include <cstdio>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

#include "core/safe_queue.hpp"

namespace dpool {
using Task = std::function<void()>;

class ThreadPool {
   public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency(), size_t max_queue_size = 1000)
        : accepting_(true), queue_(max_queue_size) {
        num_threads = std::max<size_t>(1, num_threads);
        for (size_t i = 0; i < num_threads; ++i) threads_.emplace_back([this, i] { worker(i); });
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ~ThreadPool() { stop(); }

    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        using ReturnType = decltype(f(args...));
        if (!accepting_.load()) throw std::runtime_error("ThreadPool is stopped");
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            [func = std::forward<F>(f), tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
                return std::apply(std::move(func), std::move(tup));
            });
        auto result = task->get_future();
        if (!queue_.push([task] { (*task)(); })) throw std::runtime_error("ThreadPool queue is stopped");
        return result;
    }

    template <typename F, typename... Args>
    auto try_submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        using ReturnType = decltype(f(args...));
        if (!accepting_.load()) return {};
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            [func = std::forward<F>(f), tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
                return std::apply(std::move(func), std::move(tup));
            });
        auto result = task->get_future();
        if (!queue_.try_push([task] { (*task)(); })) return {};
        return result;
    }

    // Graceful drain: stop accepting, then workers consume every queued task before exiting.
    void stop() {
        bool expected = true;
        if (!accepting_.compare_exchange_strong(expected, false)) return;
        queue_.shutdown();
        for (auto& t : threads_) if (t.joinable()) t.join();
    }

    size_t queue_size() const { return queue_.size(); }
    size_t thread_count() const { return threads_.size(); }
    bool is_running() const { return accepting_.load(); }

   private:
    void worker(size_t id) {
        Task task;
        while (queue_.wait_pop(task)) {
            if (!task) continue;
            try { task(); }
            catch (const std::exception& e) { std::fprintf(stderr, "[ThreadPool] worker %zu: %s\n", id, e.what()); }
            catch (...) { std::fprintf(stderr, "[ThreadPool] worker %zu: unknown exception\n", id); }
        }
    }

    std::atomic<bool> accepting_;
    rknn_deploy::SafeQueue<Task> queue_;
    std::vector<std::thread> threads_;
};
}  // namespace dpool
#endif
