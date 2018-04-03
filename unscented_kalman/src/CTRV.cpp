//
// Created by Rohith Menon on 3/31/18.
//

#include <iostream>
#include "CTRV.h"

void CTRV::Predict(
    const Eigen::VectorXd& state,
    const Eigen::MatrixXd& cov,
    const Eigen::VectorXd& noise,
    const std::chrono::microseconds& micros,
    Eigen::VectorXd *state_out,
    Eigen::MatrixXd *cov_out,
    SigmaPoints *sigma_points_out) {
  // Augment UKF.
  long aug_state_size = state.rows() + noise.rows();
  Eigen::VectorXd aug_state(aug_state_size);
  aug_state.fill(0);
  aug_state.head(5) = state;
  Eigen::MatrixXd aug_cov(aug_state_size, aug_state_size);
  aug_cov.fill(0);
  aug_cov.topLeftCorner(5, 5) = cov;
  aug_cov(5, 5) = noise(0);
  aug_cov(6, 6) = noise(1);
  SigmaPoints sigma_points(aug_state, aug_cov, 3 - aug_state_size);

  // Predicted Sigma points.
  double dt = (double) micros.count() / 1000000.0;
  Eigen::MatrixXd points = sigma_points.GetPoints();
  Eigen::MatrixXd predicted_points(5, points.cols());
  for (int i = 0; i < points.cols(); ++i) {
    double px = points(0, i);
    double py = points(1, i);
    double v = points(2, i);
    double yaw = points(3, i);
    double yaw_rate = points(4, i);
    double nu_a = points(5, i);
    double nu_yaw_a = points(6, i);

    Eigen::VectorXd state_pt(5);
    state_pt << px, py, v, yaw, yaw_rate;
    Eigen::VectorXd model(5);
    if (fabs(yaw_rate) < 0.0001) {
      model << v * cos(yaw) * dt, v * sin(yaw) * dt, 0, yaw_rate * dt, 0;
    } else {
      model << v / yaw_rate * (sin (yaw + yaw_rate * dt) - sin(yaw)),
          v / yaw_rate * (cos(yaw) - cos(yaw + yaw_rate * dt)),
          0,
          yaw_rate * dt,
          0;
    }
    Eigen::VectorXd noise_t(5);
    double dt_2 = dt * dt;
    noise_t << 0.5 * dt_2 * cos(yaw) * nu_a,
        0.5 * dt_2 * sin(yaw) * nu_a,
        dt * nu_a,
        0.5 * dt_2 * nu_yaw_a,
        dt * nu_yaw_a;
    predicted_points.col(i) = state_pt + model + noise_t;
  }
  // Predict state and covariance.
  SigmaPoints predicted_sigma_points(sigma_points, predicted_points);
  *sigma_points_out = predicted_sigma_points;
  *state_out = predicted_sigma_points.PredictState();
  *cov_out = predicted_sigma_points.PredictCovariance();
}
