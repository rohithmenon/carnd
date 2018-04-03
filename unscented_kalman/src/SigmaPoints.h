//
// Sigma points to approimate a non-linear tranformed guassian distribution.
//

#ifndef UNSCENTEDKF_SIGMAPOINTS_H
#define UNSCENTEDKF_SIGMAPOINTS_H


#include "Eigen/Dense"
#include "tools.h"

class SigmaPoints {
public:
  /**
   * Noop constructor
   */
  SigmaPoints();

  /**
   * Constructor
   *
   * @param state State vector
   * @param covariance Covariance matrix
   * @param lambda Parameter that decides the spread of sigma points
   */
  SigmaPoints(
      const Eigen::VectorXd& state,
      const Eigen::MatrixXd& covariance,
      double lambda);

  /**
   * Constructor
   *
   * @param from SigmaPoints instance to create from
   * @param sigma_points Sigma points
   */
  SigmaPoints(const SigmaPoints& from, Eigen::MatrixXd sigma_points);

  /**
   * Predict mean state from sigma points
   * @return VectorXd state
   */
  Eigen::VectorXd PredictState() const;

  /**
   * Predict covariance from sigma points
   * @return MatrixXd covariance
   */
  Eigen::MatrixXd PredictCovariance() const;

  /**
   * Get underlying sigma points
   *
   * @return Matrix of sigma points.
   */
  Eigen::MatrixXd GetPoints() const {
    return sigma_points_;
  }

  /**
   * Get weights vector of sigma points.
   *
   * @return Vector of weights
   */
  Eigen::MatrixXd GetWeights() const {
    return weights_;
  }
private:
  Eigen::MatrixXd sigma_points_;
  Eigen::VectorXd weights_;
  long state_size_;
};


#endif //UNSCENTEDKF_SIGMAPOINTS_H
