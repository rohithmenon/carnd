//
// Created by Rohith Menon on 3/31/18.
//

#include <iostream>
#include "SigmaPoints.h"
#include "Eigen/Dense"

SigmaPoints::SigmaPoints() : state_size_(0) { }

SigmaPoints::SigmaPoints(
    const Eigen::VectorXd& state,
    const Eigen::MatrixXd& covariance,
    double lambda) : state_size_(state.rows()) {
  sigma_points_ = Eigen::MatrixXd(state_size_, 2 * state_size_ + 1);
  sigma_points_.col(0) = state;

  Eigen::MatrixXd A = covariance.llt().matrixL();
  for (int i = 0; i < covariance.cols(); ++i) {
    sigma_points_.col(i + 1) = state + sqrt(lambda + state_size_) * A.col(i);
    sigma_points_.col(state_size_ + i + 1) = state - sqrt(lambda + state_size_) * A.col(i);
  }
  weights_ = Eigen::VectorXd(sigma_points_.cols());
  weights_.fill(1 / (2 * (lambda + state_size_)));
  weights_[0] = lambda / (lambda + state_size_);
}

SigmaPoints::SigmaPoints(const SigmaPoints& from, Eigen::MatrixXd sigma_points)
    : sigma_points_(sigma_points),
      weights_(from.weights_),
      state_size_(sigma_points.rows()) { }

Eigen::MatrixXd SigmaPoints::PredictCovariance() const {
  Eigen::VectorXd pstate = PredictState();
  Eigen::MatrixXd pcov = Eigen::MatrixXd(state_size_, state_size_);
  pcov.fill(0);
  for (int i = 0; i < sigma_points_.cols(); ++i) {
    Eigen::VectorXd deviation = sigma_points_.col(i) - pstate;
    pcov += weights_[i] * deviation * deviation.transpose();
  }
  return pcov;
}

Eigen::VectorXd SigmaPoints::PredictState() const {
  Eigen::VectorXd pstate(state_size_);
  for (int i = 0; i < state_size_; ++i) {
    pstate[i] = weights_.dot(sigma_points_.row(i));
  }
  return pstate;
}

