#include "Eigen/Dense"
#include <exception>
#include <limits>
#include <vector>

#include "tools.h"


Tools::Tools() {}

Tools::~Tools() {}

Eigen::VectorXd Tools::RMSE(const std::vector<Eigen::VectorXd> &estimations,
                            const std::vector<Eigen::VectorXd> &ground_truths) {
  size_t num_estimations = estimations.size();
  size_t num_ground_truth = ground_truths.size();

  if (num_estimations != num_ground_truth) {
    throw std::length_error("Vectors don't match in length");
  }

  if (num_estimations == 0) {
    throw std::invalid_argument("Empty vector");
  }

  Eigen::VectorXd sum_squared_error;
  for (size_t idx = 0; idx < num_estimations; ++idx) {
    const Eigen::VectorXd& estimation = estimations[idx];
    const Eigen::VectorXd& ground_truth = ground_truths[idx];
    const Eigen::VectorXd& error = (estimation - ground_truth);
    const Eigen::VectorXd& squared_error = error.array().square().matrix();
    if (idx == 0) {
      sum_squared_error = squared_error;
    } else {
      sum_squared_error = sum_squared_error + squared_error;
    }
  }
  Eigen::VectorXd mse = (sum_squared_error.array()
                         / (num_estimations + std::numeric_limits<double>::epsilon())).matrix();
  return mse.array().sqrt().matrix();
}
