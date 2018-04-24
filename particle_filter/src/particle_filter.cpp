/*
 * Implementation of particle filter
 */

#include "particle_filter.h"

#include "particle.h"
#include <iostream>
#include <iterator>
#include <numeric>
#include <set>
#include <sstream>

constexpr int NUM_PARTICLES = 100;

ParticleFilter::ParticleFilter(double sensor_range, double sigma_landmark[2], double sdev_pos[3])
    : motion_model_(sdev_pos[0], sdev_pos[1], sdev_pos[2]),
      observation_model_(sigma_landmark[0], sigma_landmark[1]),
      sensor_range_(sensor_range) { }

void ParticleFilter::Init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	// x, y, theta and their uncertainties from GPS) and all weights to 1.
  std::normal_distribution<double> x_dist(x, std[0]);
  std::normal_distribution<double> y_dist(y, std[1]);
  std::normal_distribution<double> theta_dist(theta, std[2]);
  for (int i = 0; i < NUM_PARTICLES; ++i) {
    Eigen::VectorXd state(3);
    state << x_dist(generator_), y_dist(generator_), theta_dist(generator_);
    particles_.emplace_back(Particle(i, state, 1.0));
  }
  initialized_ = true;
}

void ParticleFilter::ReportMotion(double delta_t, double velocity, double yaw_rate) {
  // Add measurements to each particle.
  for (auto &particle : particles_) {
    const Eigen::VectorXd& predicted_state =
        motion_model_.Predict(particle.GetState(), velocity, yaw_rate, delta_t);
    particle = Particle(particle.GetId(), predicted_state, particle.GetWeight());
  }
}

void ParticleFilter::ReportObservation(
    const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution.
  std::vector<double> weights;
  for (auto &particle : particles_) {
    const Eigen::VectorXd& state = particle.GetState();

    // Transform to map co-ordinates
    double theta = state[2];
    Eigen::MatrixXd universal_transform(3, 3);
    universal_transform << cos(theta), -sin(theta), state[0],
                           sin(theta), cos(theta), state[1],
                           0, 0, 1;
    std::vector<LandmarkObs> transformed_obs;
    for (LandmarkObs observation: observations) {
      Eigen::VectorXd observation_vector(3);
      observation_vector << observation.x, observation.y, 1;
      Eigen::VectorXd transformed_ob_vector = universal_transform * observation_vector;
      LandmarkObs transformed_ob {
          observation.id,
          transformed_ob_vector[0],
          transformed_ob_vector[1]
      };
      transformed_obs.emplace_back(transformed_ob);
    }

    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
    // Find nearest landmark corresponding to observations.
    std::map<LandmarkObs, LandmarkObs> nearest_landmarks;
    for (const LandmarkObs& transformed_ob : transformed_obs) {
      double best_distance = std::numeric_limits<double>::max();
      double sensor_distance = dist(
          state[0], state[1], transformed_ob.x, transformed_ob.y);
      if (sensor_distance > sensor_range_) {
        continue;
      }
      particle.sense_x.emplace_back(transformed_ob.x);
      particle.sense_y.emplace_back(transformed_ob.y);
      LandmarkObs best_match = map_landmarks[0];
      for (const LandmarkObs& map_landmark : map_landmarks) {
        double cur_distance = dist(
            map_landmark.x, map_landmark.y, transformed_ob.x, transformed_ob.y);
        if (cur_distance < best_distance && cur_distance < sensor_range_) {
          best_distance = cur_distance;
          best_match = map_landmark;
        }
      }
      nearest_landmarks.emplace(transformed_ob, best_match);
      particle.associations.emplace_back(best_match.id);
    }

    weights.emplace_back(observation_model_.Predict(nearest_landmarks));
  }

  // Resample particles.
  double normalizer = std::accumulate(weights.begin(), weights.end(), 0.0);
  for (double &weight : weights) {
    weight /= normalizer;
  }

  std::discrete_distribution<> particle_dist(weights.begin(), weights.end());
  std::vector<Particle> resampled;
  for (int i = 0; i < NUM_PARTICLES; ++i) {
    int sampled_idx = particle_dist(generator_);
    const auto& sampled = particles_[sampled_idx];
    Particle new_particle(sampled.GetId(), sampled.GetState(), weights[sampled_idx]);
    new_particle.associations = sampled.associations;
    new_particle.sense_x = sampled.sense_x;
    new_particle.sense_y = sampled.sense_y;
    resampled.emplace_back(new_particle);
  }
  particles_ = resampled;
}

const std::vector<Particle> &ParticleFilter::GetParticles() const {
  return particles_;
}

std::string ParticleFilter::getAssociations(Particle best)
{
  std::vector<int> v = best.associations;
  std::stringstream ss;
  std::copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
std::string ParticleFilter::getSenseX(Particle best)
{
  std::vector<double> v = best.sense_x;
  std::stringstream ss;
  std::copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
std::string ParticleFilter::getSenseY(Particle best)
{
  std::vector<double> v = best.sense_y;
  std::stringstream ss;
  std::copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

