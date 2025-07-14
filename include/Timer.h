#pragma once

// https://gist.github.com/TheCherno/b2c71c9291a4a1a29c889e76173c8d14

#include <chrono>
#include <iostream>
#include <string_view>
#include <filesystem>
#include <fstream>

class Timer {
public:
  Timer() { Reset(); }
  void Reset() { m_Start = std::chrono::high_resolution_clock::now(); }
  float Elapsed() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - m_Start)
               .count() *
           0.001f * 0.001f;
  }
  float ElapsedMillis() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - m_Start)
               .count() *
           0.001f;
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
};

class ScopedTimer {
public:
  ScopedTimer(std::string_view name) : m_Name(name) {}
  ~ScopedTimer() {
    float time = m_Timer.ElapsedMillis();
    std::cout << m_Name << " - " << time << "ms\n";
  }

private:
  Timer m_Timer;
  std::string m_Name;
};


class SteadyTimer {
private:
  std::chrono::time_point<std::chrono::steady_clock> m_Start;
public:
  SteadyTimer() { Reset(); }
  void Reset() { m_Start = std::chrono::steady_clock::now(); }
  float ElapsedMillis() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - m_Start)
               .count() * 0.001f;
  }
};

class OptionalTimerWriter {
public:
  OptionalTimerWriter(bool enabled, const std::string& path) : enabled_(enabled) {
    if (enabled_) {
      file_.open(path);
    }
  }

  ~OptionalTimerWriter() {
    if (file_.is_open()) file_.close();
  }

  void write(double value) {
    if (enabled_) {
      file_ << value << "\n";
    }
  }

private:
  bool enabled_;
  std::ofstream file_;
};
