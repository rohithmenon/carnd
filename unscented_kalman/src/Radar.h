//
// Radar measurement model. Uses Unscented transformation.
//

#ifndef UNSCENTEDKF_RADAR_H
#define UNSCENTEDKF_RADAR_H


#include "Eigen/Dense"
#include "SigmaPoints.h"

class Radar {
public:
  Radar();
  /**
   * Update with Radar measurement
   *
   * @param measurement Radar measurement
   * @param state Predicted state
   * @param cov Predicted covariance
   * @param sigma_points Current predicted sigma points
   * @param [out] state_out Updated state
   * @param [out] cov_out Updated covariance
   */
  void Update(const Eigen::VectorXd& measurement,
              const Eigen::VectorXd& state,
              const Eigen::MatrixXd& cov,
              const SigmaPoints& sigma_points,
              Eigen::VectorXd *state_out,
              Eigen::MatrixXd *cov_out);
private:
  Eigen::MatrixXd R;
};


#endif //UNSCENTEDKF_RADAR_H
