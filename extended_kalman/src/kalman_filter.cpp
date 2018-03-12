#include "kalman_filter.h"

#include <chrono>
#include <functional>
#include "Eigen/Dense"

KalmanFilter::KalmanFilter(
    Eigen::VectorXd x,
    Eigen::MatrixXd P,
    Eigen::MatrixXd R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> error_fn,
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> H_factory,
    std::function<Eigen::MatrixXd(const std::chrono::microseconds&)> Q_factory,
    std::function<Eigen::MatrixXd(const std::chrono::microseconds&)> F_factory
): x_(x),
   P_(P),
   R_(R),
   error_fn_(error_fn),
   H_factory_(H_factory),
   Q_factory_(Q_factory),
   F_factory_(F_factory) {}

std::shared_ptr<KalmanFilter> KalmanFilter::FromState(const KalmanFilter& other) const {
  return std::shared_ptr<KalmanFilter>(
      new KalmanFilter(other.x_, other.P_, R_, error_fn_, H_factory_, Q_factory_, F_factory_));
}

std::shared_ptr<KalmanFilter> KalmanFilter::FromState(const Eigen::VectorXd& state) const {
  return std::shared_ptr<KalmanFilter>(
      new KalmanFilter(state, P_, R_, error_fn_, H_factory_, Q_factory_, F_factory_));
}

Eigen::VectorXd KalmanFilter::Predict(const std::chrono::microseconds& micros) const {
  // x′ = Fx + u
  return F_factory_(micros) * x_;
}

std::shared_ptr<KalmanFilter> KalmanFilter::Update(
    const Eigen::VectorXd &z,
    const std::chrono::microseconds& elapsed_micros) const {
  // **Prediction**
  Eigen::MatrixXd F = F_factory_(elapsed_micros);
  // x′ = F*x + u
  Eigen::VectorXd x_p = F * x_;
  // P′= F*P*F_t + Q
  Eigen::MatrixXd P_p = Q_factory_(elapsed_micros) + F * P_ * F.transpose();

  // **Update**
  Eigen::MatrixXd H = H_factory_(x_p);
  Eigen::MatrixXd H_t = H.transpose();
  // y = z − Hx′
  Eigen::VectorXd y = error_fn_(z, x_p);

  // S = H*P′*H_t + R
  Eigen::MatrixXd S = R_ + H * P_p * H_t;
  // K = P′*H_t*S−1
  Eigen::MatrixXd K = P_p * H_t * S.inverse();

  // ** New state and covariance matrix**
  // x = x′ + Ky
  Eigen::VectorXd x_new = x_p + K * y;
  // P = (I − K*H) * P′
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P_.rows(), P_.cols());
  Eigen::MatrixXd P_new = (I - K * H) * P_p;

  return std::shared_ptr<KalmanFilter>(
      new KalmanFilter(x_new, P_new, R_, error_fn_, H_factory_, Q_factory_, F_factory_));
}
