#include "FusionEKF.h"

#include <math.h>
#include <memory>
#include "Eigen/Dense"

/**
 * Radar projection matrix approximation using first term of the
 * Taylor series expansion of the radar projection function.
 *
 * @param state State of the filter
 *
 * @return Approximate radar projection matrix.
 */
static Eigen::MatrixXd RadarProjectionMatrix(const Eigen::VectorXd& state) {
  Eigen::MatrixXd Hj(3,4);

  double px = state(0);
  double py = state(1);
  double vx = state(2);
  double vy = state(3);

  double c1 = px * px + py * py + std::numeric_limits<double>::epsilon();
  double c2 = sqrt(c1);
  double c3 = (c1 * c2);

  // Compute the radar projection Jacobian matrix
  Hj << px / c2, py / c2, 0, 0,
       -py / c1, px / c1, 0, 0,
        py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2, py / c2;

  return Hj;
}

/**
 * Error in predicted radar state vs measurement.
 *
 * @param actual Measurement from radar
 * @param prediction Predicted state of radar
 *
 * @return Error vector.
 */
static Eigen::VectorXd RadarError(
    const Eigen::VectorXd& actual, const Eigen::VectorXd& prediction) {
  Eigen::VectorXd measurement(3);

  double px = prediction(0);
  double py = prediction(1);
  double vx = prediction(2);
  double vy = prediction(3);

  double c1 = sqrt(px * px + py * py) + std::numeric_limits<double>::epsilon();
  double c2 = atan2(py, px + std::numeric_limits<double>::epsilon());
  double c3 = (px * vx + py * vy) / c1;

  measurement << c1, c2, c3;
  Eigen::VectorXd error = actual - measurement;
  error(1) = fmod(error(1), 2 * M_PI);
  return error;
}

/**
 * Compute state vector from measurement.
 *
 * @param measurement Measurement from either of lidar or radar
 *
 * @return State vector.
 */
static Eigen::VectorXd GetStateFromMeasurement(const MeasurementPackage& measurement) {
  Eigen::VectorXd state(4);
  const Eigen::VectorXd &raw_measurements = measurement.raw_measurements_;
  if (measurement.sensor_type_ == measurement.LASER) {
    state << raw_measurements(0), raw_measurements(1), 1, 1;
  } else if (measurement.sensor_type_ == measurement.RADAR) {
    double rho = raw_measurements(0);
    double phi = raw_measurements(1);
    state << rho * cos(phi), rho * sin(phi), 1, 1;
  }
  return state;
}

/*
 * Constructor.
 */
FusionEKF::FusionEKF(): previous_timestamp_(std::chrono::microseconds(0L)) {
  // initializing matrices
  // state
  Eigen::VectorXd x(4);
  x << 1, 1, 1, 1;

  // covariance
  Eigen::MatrixXd P(4, 4);
  P << 1000, 0, 0, 0,
       0, 1000, 0, 0,
       0, 0, 1000, 0,
       0, 0, 0, 1000;

  Eigen::MatrixXd R_laser = Eigen::MatrixXd(2, 2);
  Eigen::MatrixXd R_radar = Eigen::MatrixXd(3, 3);

  // measurement covariance matrix - lidar
  R_laser << 0.0225, 0,
             0, 0.0225;
  // projection maxtrix - lidar
  Eigen::MatrixXd H_laser(2, 4);
  H_laser << 1, 0, 0, 0,
             0, 1, 0, 0;
  auto H_factory_laser = [=](const Eigen::VectorXd& state) { return H_laser; };
  // error function - lidar
  auto error_fn_laser = [=](
      const Eigen::VectorXd& actual, const Eigen::VectorXd& prediction) {
    Eigen::VectorXd error = actual - H_laser * prediction;
    return error;
  };

  // measurement covariance matrix - radar
  R_radar << 0.09, 0, 0,
             0, 0.0009, 0,
             0, 0, 0.09;

  auto F_factory = [](const std::chrono::microseconds& micros) {
    Eigen::MatrixXd F(4, 4);
    double seconds = (double) micros.count() / 1000000.0;
    F << 1.0, 0.0, seconds, 0.0,
         0.0, 1.0, 0.0, seconds,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0;
    return F;
  };

  // Process covariance matrix factory
  auto Q_factory = [](const std::chrono::microseconds& delta_micros) {
    // Process noise
    double ax = 5;
    double ay = 5;
    double delta_seconds = (double) delta_micros.count() / 1000000.0;

    double dt = delta_seconds;
    double dt_2 = dt * dt;
    double dt_3 = dt_2 * dt;
    double dt_4 = dt_3 * dt;

    Eigen::MatrixXd Q(4, 4);
    Q << dt_4 * ax / 4.0, 0, dt_3 * ax / 2.0, 0,
         0, dt_4 * ay / 4.0, 0, dt_3 * ay / 2.0,
         dt_3 * ax / 2.0, 0, dt_2 * ax, 0,
         0, dt_3 * ay / 2.0, 0, dt_2 * ay;
    return Q;
  };

  // Create kalman filters for lidar and radar
  lidar_ekf_.reset(
      new KalmanFilter(x, P, R_laser, error_fn_laser, H_factory_laser, Q_factory, F_factory));
  radar_ekf_.reset(
      new KalmanFilter(x, P, R_radar, RadarError, RadarProjectionMatrix, Q_factory, F_factory));
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ReportMeasurement(const MeasurementPackage &measurement_pack) {
  auto current_timestamp = std::chrono::microseconds(measurement_pack.timestamp_);
  auto prev_timestamp = previous_timestamp_.load();
  auto elapsed_micros = current_timestamp - prev_timestamp;
  previous_timestamp_.store(current_timestamp);
  // First frame, initialize state and update the timestamp.
  if (prev_timestamp.count() == 0) {
    Eigen::VectorXd state = GetStateFromMeasurement(measurement_pack);
    lidar_ekf_ = lidar_ekf_->FromState(state);
    radar_ekf_ = radar_ekf_->FromState(state);
    return;
  }
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Update radar filter with new filter returned from Update.
    radar_ekf_ = radar_ekf_->Update(measurement_pack.raw_measurements_, elapsed_micros);
    // Update lidar filter with state copied from new_radar_filter.
    lidar_ekf_ = lidar_ekf_->FromState(*radar_ekf_);
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    // Update lidar filter with new filter returned from Update.
    lidar_ekf_ = lidar_ekf_->Update(measurement_pack.raw_measurements_, elapsed_micros);
    // Update radar filter with state copied from new_lidar_filter.
    radar_ekf_ = radar_ekf_->FromState(*lidar_ekf_);
  }
}

Eigen::VectorXd FusionEKF::Predict() {
  // Here we can use either one of the Kalman filters because they share the state
  // and covariance matrix.
  return radar_ekf_->Predict(std::chrono::microseconds(0L));
}
