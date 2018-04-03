//
// Constant turn rate and velocity model (CTRV).
//

#ifndef UNSCENTEDKF_CTRV_H
#define UNSCENTEDKF_CTRV_H


#include <chrono>
#include "SigmaPoints.h"

class CTRV {
public:

  /**
   * Predict next state, and covariance
   *
   * @param state current state
   * @param covariance current cov
   * @param noise noise
   * @param micros duration in micros
   * @param [out] state_out new state
   * @param [out] cov_out new cov
   * @param [out] sigma_points_out predicated sigma points.
   * @return
   */
  void Predict(
      const Eigen::VectorXd& state,
      const Eigen::MatrixXd& cov,
      const Eigen::VectorXd& noise,
      const std::chrono::microseconds& micros,
      Eigen::VectorXd* state_out,
      Eigen::MatrixXd* cov_out,
      SigmaPoints* sigma_points_out
  );
};


#endif //UNSCENTEDKF_CTRV_H
