#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "Eigen/Dense"
#include <memory>

#include "kalman_filter.h"
#include "measurement_package.h"

class FusionEKF {
public:
  /**
   * Constructor.
   */
  FusionEKF();

  /**
   * Destructor.
   */
  virtual ~FusionEKF();

  /**
   * Report measurement
   */
  void ReportMeasurement(const MeasurementPackage &measurement_pack);

  /**
   * Predict the state variables from fused sensor estimation.
   */
  Eigen::VectorXd Predict();

private:
  std::atomic<std::chrono::microseconds> previous_timestamp_;
  std::shared_ptr<KalmanFilter> lidar_ekf_;
  std::shared_ptr<KalmanFilter> radar_ekf_;
};

#endif /* FusionEKF_H_ */
