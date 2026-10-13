#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

// Synchronous row-range jobs. Run() calls on one pool must be serialized.
// The caller participates; the other threads sleep between jobs.
class ThreadPool {
public:
    explicit ThreadPool(int num_threads = 1) : m_num_threads(num_threads) {
        if (num_threads < 1)
            throw std::invalid_argument("Thread count must be positive");
        m_workers.reserve(num_threads - 1);
        try {
            for (int id = 1; id < num_threads; ++id)
                m_workers.emplace_back([this, id] { Worker(id); });
        } catch (...) {
            Stop();
            throw;
        }
    }

    ~ThreadPool() { Stop(); }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int GetNumThreads() const { return m_num_threads; }

    // The callable stays alive until all ranges finish; no task allocation/copy.
    template<class Function>
    void Run(int rows, const Function& function) {
        if (rows <= 0)
            return;
        if (m_num_threads == 1 || rows == 1) {
            function(0, rows);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_rows = rows;
            m_context = &function;
            m_task = [](const void* context, int begin, int end) {
                (*static_cast<const Function*>(context))(begin, end);
            };
            m_error = nullptr;
            m_remaining = m_num_threads - 1;
            ++m_generation;
        }
        m_work_ready.notify_all();
        Execute(0);

        std::exception_ptr error;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_completed.wait(lock, [this] { return m_remaining == 0; });
            error = m_error;
            m_task = nullptr;
            m_context = nullptr;
        }
        if (error)
            std::rethrow_exception(error);
    }

private:
    void Execute(int id) {
        try {
            const int begin = static_cast<int>(std::int64_t(m_rows) * id / m_num_threads);
            const int end = static_cast<int>(std::int64_t(m_rows) * (id + 1) / m_num_threads);
            if (begin < end)
                m_task(m_context, begin, end);
        } catch (...) {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_error)
                m_error = std::current_exception();
        }
    }

    void Worker(int id) {
        std::uint64_t generation = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_work_ready.wait(lock, [&] { return m_stopping || generation != m_generation; });
                if (m_stopping)
                    return;
                generation = m_generation;
            }
            Execute(id);
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (--m_remaining == 0)
                    m_completed.notify_one();
            }
        }
    }

    void Stop() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stopping = true;
        }
        m_work_ready.notify_all();
        for (auto& worker : m_workers)
            worker.join();
    }

    const int m_num_threads;
    std::mutex m_mutex;
    std::condition_variable m_work_ready, m_completed;
    std::vector<std::thread> m_workers;
    std::uint64_t m_generation = 0;
    int m_rows = 0, m_remaining = 0;
    bool m_stopping = false;
    const void* m_context = nullptr;
    void (*m_task)(const void*, int, int) = nullptr;
    std::exception_ptr m_error;
};
