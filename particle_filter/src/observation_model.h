//
// Multi variate Gaussian for estimating probability of observation
//

#ifndef PARTICLE_FILTER_OBSERVATIONMODEL_H
#define PARTICLE_FILTER_OBSERVATIONMODEL_H

#include <unordered_map>

#include "Eigen/Dense"
#include "helper_functions.h"
#include <map>


class ObservationModel {
public:
  /**
   * Constructor
   *
   * @param x_sdev Standard deviation of position x
   * @param y_sdev Standard deviation of position y
   */
  ObservationModel(double x_sdev, double y_sdev): x_sdev_(x_sdev), y_sdev_(y_sdev) {}

  /**
   * Predicted weight for observation=>landmark mapping.
   * @param observations
   * @return
   */
  double Predict(const std::map<LandmarkObs, LandmarkObs> &observations) const;

private:
  double x_sdev_;
  double y_sdev_;
};


#endif //PARTICLE_FILTER_OBSERVATIONMODEL_H
