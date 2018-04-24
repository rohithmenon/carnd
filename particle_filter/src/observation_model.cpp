//
// Observation model implementation.
//

#include <map>
#include <iostream>
#include "Eigen/Dense"
#include "observation_model.h"


double ObservationModel::Predict(const std::map<LandmarkObs, LandmarkObs> &observations) const {
  double probability = 1.0;
  for (const auto &observation : observations) {
    const LandmarkObs& obs = observation.first;
    const LandmarkObs& landmark = observation.second;
    double exponent = -0.5 * (pow((obs.x - landmark.x) / x_sdev_, 2)
                              + pow((obs.y - landmark.y) / y_sdev_, 2));
    double current_prob = (0.5 / (M_PI * x_sdev_ * y_sdev_)) * exp(exponent);
    probability *= current_prob;
  }
  return probability;
}
