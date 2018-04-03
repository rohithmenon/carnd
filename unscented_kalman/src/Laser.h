//
// Laser measurement model
//

#ifndef UNSCENTEDKF_LASER_H
#define UNSCENTEDKF_LASER_H


#include "Eigen/Dense"

class Laser {
public:
  Laser();
  /**
   * Update with Laser measurement
   *
   * @param measurement Laser measurement
   * @param state current_state
   * @param cov current_covariance
   * @param [out] state_out Updated state
   * @param [out] cov_out Updated covariance
   */
  void Update(const Eigen::VectorXd& measurement,
              const Eigen::VectorXd& state,
              const Eigen::MatrixXd& cov,
              Eigen::VectorXd *state_out,
              Eigen::MatrixXd *cov_out);
private:
  Eigen::MatrixXd R;
  Eigen::MatrixXd H;
  Eigen::MatrixXd Ht;
};


#endif //UNSCENTEDKF_LASER_H
