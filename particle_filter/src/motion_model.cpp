//
// Bicycle motion model implementation.
//

#include "motion_model.h"

MotionModel::MotionModel(double sdev_x, double sdev_y, double sdev_theta) {
  noise_x_ = std::normal_distribution<double>(0, sdev_x);
  noise_y_ = std::normal_distribution<double>(0, sdev_y);
  noise_theta_ = std::normal_distribution<double>(0, sdev_theta);
}

Eigen::VectorXd MotionModel::Predict(
    Eigen::VectorXd state, double velocity, double yaw_rate, double delta_t) {
  Eigen::VectorXd delta(3), noise(3);

  // Experiment with modeling acceleration and yaw_rate noise.
  /*
  double delta_t_2 = delta_t * delta_t;
  double k1 = 0.5 * acceleration_noise_(generator_) * delta_t_2;
  noise[0] = k1 * sin(state[2]);
  noise[1] = k1 * cos(state[2]);
  noise[2] = 0.5 * rho_dot_dot_noise_(generator_) * delta_t_2;
  */

  noise[0] = noise_x_(generator_);
  noise[1] = noise_y_(generator_);
  noise[2] = noise_theta_(generator_);


  if (fabs(yaw_rate) > std::numeric_limits<double>::epsilon()) {
    // Curved case.
    double k2 = velocity / yaw_rate + std::numeric_limits<double>::epsilon();
    double delta_yaw = yaw_rate * delta_t;
    delta[0] = k2 * (sin(state[2] + delta_yaw) - sin(state[2]));
    delta[1] = k2 * (cos(state[2]) - cos(state[2] + delta_yaw));
    delta[2] = delta_yaw;
  } else {
    // Straight line case.
    delta[0] = velocity * sin(state[2]) * delta_t;
    delta[1] = velocity * cos(state[2]) * delta_t;
    delta[2] = 0.0;
  }
  return state + delta + noise;
}
