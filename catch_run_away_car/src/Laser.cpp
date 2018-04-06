//
// Laser measurement model implementation.
//

#include <iostream>
#include "Laser.h"

Laser::Laser(): R(2, 2), H(2, 5) {
  R << 0.15 * 0.15, 0,
       0, 0.15 * 0.15;
  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;
  Ht = H.transpose();
}

void Laser::Update(
    const Eigen::VectorXd& measurement,
    const Eigen::VectorXd& state,
    const Eigen::MatrixXd& cov,
    Eigen::VectorXd *state_out,
    Eigen::MatrixXd *cov_out) {
  Eigen::MatrixXd cov_ht = cov * Ht;
  Eigen::MatrixXd S = R + H * cov_ht;
  auto S_inv = S.inverse();
  Eigen::MatrixXd K = cov_ht * S_inv;
  Eigen::VectorXd y = measurement - H * state;
  *state_out = state + K * y;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(cov.rows(), cov.cols());
  *cov_out = (I - K * H) * cov;
}
