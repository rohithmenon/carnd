#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include <chrono>
#include "Eigen/Dense"

/**
 * This class implements KalmanFilter. Note that this class is immutable and hence
 * thread-safe.
 */
class KalmanFilter {
public:
  /**
   * Constructor
   */
  KalmanFilter(
      Eigen::VectorXd x,
      Eigen::MatrixXd P,
      Eigen::MatrixXd R,
      std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> error_fn,
      std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> H_factory,
      std::function<Eigen::MatrixXd(const std::chrono::microseconds&)> Q_factory,
      std::function<Eigen::MatrixXd(const std::chrono::microseconds&)> F_factory);

  /**
   * Predict the state for a time in future.
   *
   * @param micros Microseconds into future that the prediction must be made.
   *
   * @return Predicted state vector.
   */
  Eigen::VectorXd Predict(const std::chrono::microseconds& micros) const;

  /**
   * Update the state with the measurement, elapsed_micros after previous update.
   *
   * @param measurement Vector of measurements.
   * @param elapsed_micros Microseconds since the last update.
   *
   * @return Returns a new KalmanFilter with updated state
   */
  std::shared_ptr<KalmanFilter> Update(
      const Eigen::VectorXd &measurement,
      const std::chrono::microseconds& elapsed_micros) const;

  /**
   * Return a new Kalman filter with state vector and covariance matrix copied from the passed
   * in Kalman filter.
   *
   * @param other Kalman filter from which the state must be copied.
   *
   * @return A new Kalman filter
   */
  std::shared_ptr<KalmanFilter> FromState(const KalmanFilter& other) const;

  /**
   * Return a new Kalman filter with state vector copied from the passed in values.
   *
   * @param state State vector of the filter.
   *
   * @return A new Kalman filter
   */
  std::shared_ptr<KalmanFilter> FromState(const Eigen::VectorXd& state) const;
private:

  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // measurement covariance matrix
  Eigen::MatrixXd R_;

  // error function
  std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> error_fn_;

  // measurement matrix factory
  std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> H_factory_;

  // process covariance matrix factory
  std::function<Eigen::MatrixXd(const std::chrono::microseconds&)> Q_factory_;

  // state transition matrix factory
  std::function<Eigen::MatrixXd(const std::chrono::microseconds&)> F_factory_;
};

#endif /* KALMAN_FILTER_H_ */
