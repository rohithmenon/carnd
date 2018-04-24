//
// Bicycle Motion Model
//

#ifndef PARTICLE_FILTER_MOTION_MODEL_H
#define PARTICLE_FILTER_MOTION_MODEL_H

#include <random>
#include "Eigen/Dense"


class MotionModel {
public:
  MotionModel(double sdev_x, double sdev_y, double sdev_theta);

  /**
   * Predict new state given current state, velocity and yaw_rate
   *
   * @param state Current state
   * @param velocity Velocity in meters/sec
   * @param yaw_rate Yaw rate in radians/sec
   * @param delta Delta time in seconds
   *
   * @return Predicted state
   */
  Eigen::VectorXd Predict(Eigen::VectorXd state, double velocity, double yaw_rate, double delta_t);

private:
  // Random engine
  std::default_random_engine generator_;
  // X poisition noise Guassian
  std::normal_distribution<double> noise_x_;
  // Y poisition noise Guassian
  std::normal_distribution<double> noise_y_;
  // Theta noise Guassian
  std::normal_distribution<double> noise_theta_;
};


#endif //PARTICLE_FILTER_MOTION_MODEL_H
