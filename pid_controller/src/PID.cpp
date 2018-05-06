#include "PID.h"

#include <algorithm>
#include <iostream>

constexpr double decay = 0.5;

PID::PID(double kp,
         double ki,
         double kd,
         double target,
         double min_correction,
         double max_correction)
    : kp_(kp),
      ki_(ki),
      kd_(kd),
      total_error_(0.0),
      last_error_(0.0),
      target_(target),
      correction_(0.0),
      min_correction_(min_correction),
      max_correction_(max_correction) {}

void PID::Report(double observed) {
  double error = target_ - observed;
  total_error_ += error;
  double smoothed_error = decay * error + (1.0 - decay) * last_error_;
  double delta_error = smoothed_error - last_error_;
  last_error_ = smoothed_error;
  correction_ = kp_ * error + ki_ * total_error_ + kd_ * delta_error;
  correction_ = std::min(max_correction_, std::max(min_correction_, correction_));
}

double PID::Correction() {
  return correction_;
}
