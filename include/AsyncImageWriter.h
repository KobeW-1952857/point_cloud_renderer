#pragma once

#include <condition_variable>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

struct ImageSaveTask {
  std::string filename;
  std::vector<uchar3> pixel_data;
  size_t width, height;
};

class AsyncImageWriter {
 public:
  AsyncImageWriter(size_t threads = 4) : m_running(true), m_num_threads(threads) {
    for (size_t i = 0; i < m_num_threads; ++i) m_writer_threads.emplace_back(&AsyncImageWriter::processQueue, this);
  }
  ~AsyncImageWriter() {
    {
      std::lock_guard<std::mutex> lock(m_queue_mutex);
      m_running = false;
      m_queue_cond_var.notify_one();
    }
    for (auto &t : m_writer_threads) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  void enqueue(ImageSaveTask task) {
    {
      std::lock_guard<std::mutex> lock(m_queue_mutex);
      m_save_queue.push(std::move(task));
    }
    m_queue_cond_var.notify_one();
  }

 private:
  std::queue<ImageSaveTask> m_save_queue;
  std::mutex m_queue_mutex;
  std::condition_variable m_queue_cond_var;
  std::vector<std::thread> m_writer_threads;
  size_t m_num_threads;
  bool m_running;

 private:
  void processQueue() {
    while (true) {
      ImageSaveTask task;
      {
        std::unique_lock<std::mutex> lock(m_queue_mutex);
        m_queue_cond_var.wait(lock, [this] { return !m_save_queue.empty() || !m_running; });
        if (m_save_queue.empty() && !m_running) {
          break;
        }
        if (!m_save_queue.empty()) {
          task = std::move(m_save_queue.front());
          m_save_queue.pop();
        } else {
          continue;
        }
      }
      savePPMImage(task.filename, task.pixel_data, task.width, task.height, false);
    }
  }

  void savePPMImage(const std::string &filename, const std::vector<uchar3> &pixel_data, int width, int height,
                    bool verbose = false) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
      return;
    }

    file << "P6\n";
    file << width << " " << height << "\n";
    file << "255\n";  // Max color value

    file.write(reinterpret_cast<const char *>(pixel_data.data()), pixel_data.size() * sizeof(uchar3));

    file.close();
    if (verbose) std::cout << "Image saved to " << filename << "\r" << std::flush;
  }
};