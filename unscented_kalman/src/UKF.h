//
// Unscented Kalman filter chain
//

#ifndef UNSCENTEDKF_UKF_H
#define UNSCENTEDKF_UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"
#include "CTRV.h"
#include "Laser.h"
#include "Radar.h"

class UKF {
public:
  UKF();

  UKF(const Eigen::VectorXd& state, const Eigen::MatrixXd& cov);


  /**
   * Report measurement.
   *
   * @param meas_package
   * @return
   */
  void ReportMeasurement(MeasurementPackage meas_package);

  /**
   * Predict state
   *
   * @param micros duration in micros
   *
   * @return Vector of predicted state
   */
  Eigen::VectorXd Predict(const std::chrono::microseconds& micros);
private:
  void UpdateStateFromMeasurement(const MeasurementPackage& measurement);

  CTRV ctrv_;
  Laser laser_;
  Radar radar_;
  Eigen::VectorXd state_;
  Eigen::MatrixXd cov_;
  Eigen::VectorXd noise_;
  std::atomic<std::chrono::microseconds> previous_timestamp_;
};


#endif //UNSCENTEDKF_UKF_H
