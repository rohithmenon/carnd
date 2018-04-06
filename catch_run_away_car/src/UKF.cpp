//
// Unscented Kalman filter chain
//

#include <iostream>
#include "UKF.h"
#include "Eigen/Dense"
#include "SigmaPoints.h"

UKF::UKF()
    : state_(5),
      cov_(5, 5),
      noise_(2),
      previous_timestamp_(std::chrono::microseconds(0L)) {
  state_.fill(0);
  cov_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;
  noise_ << 9, 3.0;
}

UKF::UKF(const Eigen::VectorXd& state, const Eigen::MatrixXd& cov)
    : state_(state),
      cov_(cov) {}

/**
 * Predict state after micros microseconds.
 *
 * @param micros
 * @return Predicted state
 */
Eigen::VectorXd UKF::Predict(const std::chrono::microseconds& micros) {
  Eigen::VectorXd p_state;
  Eigen::MatrixXd p_cov;
  SigmaPoints p_sigma_pts;
  ctrv_.Predict(state_, cov_, noise_, micros, &p_state, &p_cov, &p_sigma_pts);
  return p_state;
}

/**
 * Compute state vector from measurement.
 *
 * @param measurement Measurement from either of lidar or radar
 */
void UKF::UpdateStateFromMeasurement(const MeasurementPackage& measurement) {
  const Eigen::VectorXd &raw_measurements = measurement.raw_measurements_;
  if (measurement.sensor_type_ == measurement.LASER) {
    state_ << raw_measurements(0), raw_measurements(1), 3.0, 2.3, 0;
  } else if (measurement.sensor_type_ == measurement.RADAR) {
    double rho = raw_measurements(0);
    double phi = raw_measurements(1);
    state_ << rho * cos(phi), rho * sin(phi), 3.0, 2.3, 0;
  }
}

/**
 * Report either Radar or Laser measurement
 *
 * @param meas_package
 */
void UKF::ReportMeasurement(MeasurementPackage meas_package) {
  auto current_timestamp = std::chrono::microseconds(meas_package.timestamp_);
  auto prev_timestamp = previous_timestamp_.load();
  auto elapsed_micros = current_timestamp - prev_timestamp;
  previous_timestamp_.store(current_timestamp);
  // First frame, initialize state and update the timestamp.
  if (prev_timestamp.count() == 0) {
    UpdateStateFromMeasurement(meas_package);
    return;
  }
  Eigen::VectorXd p_state;
  Eigen::MatrixXd p_cov;
  SigmaPoints p_sigma_pts;
  ctrv_.Predict(state_, cov_, noise_, elapsed_micros, &p_state, &p_cov, &p_sigma_pts);

  Eigen::VectorXd updated_state;
  Eigen::MatrixXd updated_cov;
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    laser_.Update(meas_package.raw_measurements_, p_state, p_cov, &updated_state, &updated_cov);
    state_ = updated_state;
    cov_ = updated_cov;
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    radar_.Update(meas_package.raw_measurements_, p_state, p_cov, p_sigma_pts, &updated_state, &updated_cov);
    state_ = updated_state;
    cov_ = updated_cov;
  }
}
