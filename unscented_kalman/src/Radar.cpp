//
// Created by Rohith Menon on 4/2/18.
//

#include <iostream>
#include "Radar.h"
#include "Eigen/Dense"

Radar::Radar(): R(3, 3) {
  R << 0.3 * 0.3, 0, 0,
       0, 0.03 * 0.03, 0,
       0, 0, 0.3 * 0.3;
}

void Radar::Update(
    const Eigen::VectorXd &measurement,
    const Eigen::VectorXd& state,
    const Eigen::MatrixXd& cov,
    const SigmaPoints& sigma_points,
    Eigen::VectorXd *state_out,
    Eigen::MatrixXd *cov_out) {
  // Reuse already calculated Sigma points.
  Eigen::MatrixXd points = sigma_points.GetPoints();

  // Predicted Sigma points.
  Eigen::MatrixXd predicted_points(3, points.cols());
  for (int i = 0; i < points.cols(); ++i) {
    double px = points(0, i);
    double py = points(1, i);
    double v = points(2, i);
    double phi = points(3, i);

    predicted_points(0, i) = sqrt(px * px + py * py) + std::numeric_limits<double>::epsilon();
    predicted_points(1, i) = atan2(py, px + std::numeric_limits<double>::epsilon());
    predicted_points(2, i) = (px * v * cos(phi) + py * v * sin(phi)) / predicted_points(0, i);
    predicted_points(1, i) = fmod(predicted_points(1, i), 2 * M_PI);
  }
  SigmaPoints predicted_sigma_points(sigma_points, predicted_points);
  Eigen::VectorXd z = predicted_sigma_points.PredictState();

  Eigen::VectorXd weights = sigma_points.GetWeights();
  Eigen::MatrixXd T(points.rows(), predicted_points.rows());
  T.fill(0);
  for (int i = 0; i < points.cols(); ++i) {
    Eigen::VectorXd z_diff = predicted_points.col(i) - z;
    z_diff(1) = fmod(z_diff(1), 2 * M_PI);
    Eigen::VectorXd x_diff = points.col(i) - state;
    x_diff(3) = fmod(x_diff(3), 2* M_PI);
    T += weights(i) * x_diff * z_diff.transpose();
  }
  Eigen::MatrixXd S = predicted_sigma_points.PredictCovariance() + R;
  auto S_inv = S.inverse();
  MatrixXd K = T * S_inv;
  Eigen::VectorXd y = measurement - z;
  y(1) = fmod(y(1), 2.0 * M_PI);

  // Updated state and covariance.
  *state_out = state + K * y;
  *cov_out = cov - K * S * K.transpose();

  // Calculate and output NIS
  std::cout << "RADAR_NIS: " << y.transpose() * S_inv * y << std::endl;
}
