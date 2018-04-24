#include <iostream>
#include "particle_filter.h"


int main(int argc, char* argv[]) {
  double delta_t = 0.1; // Time elapsed between measurements [sec]
  double sensor_range = 50; // Sensor range [m]

  double sigma_pos [3] = {0.3, 0.3, 0.01}; // GPS measurement uncertainty [x [m], y [m], theta [rad]]
  double sigma_landmark[2] = {0.3, 0.3}; // Landmark measurement uncertainty [x [m], y [m]]

  Map map;
  map.emplace_back(LandmarkObs{1, 5, 3});
  map.emplace_back(LandmarkObs{2, 2, 1});
  map.emplace_back(LandmarkObs{3, 6, 1});
  map.emplace_back(LandmarkObs{4, 7, 4});
  map.emplace_back(LandmarkObs{5, 4, 7});

  ParticleFilter pf(sensor_range, sigma_landmark);

  Eigen::VectorXd p1_state(3);
  p1_state << 4, 5, -M_PI / 2;
  auto p1 = Particle(0, p1_state, 1.0);

  pf.SetParticles(std::vector<Particle>({p1}));
  std::vector<LandmarkObs> obs;
  obs.emplace_back(LandmarkObs{-1, 2, 2});
  obs.emplace_back(LandmarkObs{-1, 3, -2});
  obs.emplace_back(LandmarkObs{-1, 0, -4});

  pf.ReportObservation(obs, map);

}