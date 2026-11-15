#ifndef SAFE_QUEUE_HPP
#define SAFE_QUEUE_HPP

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace rknn_deploy {

template <typename T>
class SafeQueue {
   public:
    explicit SafeQueue(size_t max_size = 1000) : max_size_(max_size) {}
    SafeQueue(const SafeQueue&) = delete;
    SafeQueue& operator=(const SafeQueue&) = delete;
    ~SafeQueue() { shutdown(); }

    bool push(T val) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (max_size_ > 0) {
            cond_not_full_.wait(lock, [this] { return shutdown_ || queue_.size() < max_size_; });
        }
        if (shutdown_) return false;
        queue_.push(std::move(val));
        lock.unlock();
        cond_not_empty_.notify_one();
        return true;
    }

    bool try_push(T val) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_ || (max_size_ > 0 && queue_.size() >= max_size_)) return false;
        queue_.push(std::move(val));
        cond_not_empty_.notify_one();
        return true;
    }

    // Returns false only when the queue is shut down AND completely drained.
    bool wait_pop(T& val) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_not_empty_.wait(lock, [this] { return shutdown_ || !queue_.empty(); });
        if (queue_.empty()) return false;
        val = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        if (max_size_ > 0) cond_not_full_.notify_one();
        return true;
    }

    bool try_pop(T& val) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        val = std::move(queue_.front());
        queue_.pop();
        if (max_size_ > 0) cond_not_full_.notify_one();
        return true;
    }

    bool pop(T& val, int timeout_ms = 0) {
        if (timeout_ms == 0) return try_pop(val);
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_not_empty_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                      [this] { return shutdown_ || !queue_.empty(); })) return false;
        if (queue_.empty()) return false;
        val = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        if (max_size_ > 0) cond_not_full_.notify_one();
        return true;
    }

    size_t size() const { std::lock_guard<std::mutex> lock(mutex_); return queue_.size(); }
    bool empty() const { std::lock_guard<std::mutex> lock(mutex_); return queue_.empty(); }
    bool full() const { std::lock_guard<std::mutex> lock(mutex_); return max_size_ > 0 && queue_.size() >= max_size_; }
    size_t max_size() const { return max_size_; }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        std::swap(queue_, empty);
        cond_not_full_.notify_all();
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cond_not_empty_.notify_all();
        cond_not_full_.notify_all();
    }
    bool is_shutdown() const { std::lock_guard<std::mutex> lock(mutex_); return shutdown_; }

   private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cond_not_empty_;
    std::condition_variable cond_not_full_;
    size_t max_size_;
    bool shutdown_ = false;
};

}  // namespace rknn_deploy
#endif
