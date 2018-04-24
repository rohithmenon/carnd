/*
 * 2D particle filter
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "helper_functions.h"
#include "motion_model.h"
#include "observation_model.h"
#include "particle.h"


class ParticleFilter {
public:

  /**
   * Constructor
   *
   * @param sensor_range Range for the sensor
   * @param sigma_landmark Standard deviation for landmark observation
   * @param sigma_position Standar deviation for position.
   */
	ParticleFilter(double sensor_range, double sigma_landmark[2], double sigma_position[3]);

	/**
	 * Initialize Particle filter with the first reading from GPS, heading and standard deviations
	 * of the position/theta.
	 *
	 * @param x
	 * @param y
	 * @param theta
	 * @param std
	 */
	void Init(double x, double y, double theta, double std[]);

	/**
	 * Report motion to the filter
	 *
	 * @param delta_t Time delta in seconds
	 * @param velocity Previous velocity
	 * @param yaw_rate Previous yaw-rate
	 */
	void ReportMotion(double delta_t, double velocity, double yaw_rate);

  /**
   * Returns whether the filter is initialized or not.
   *
   * @return boolean
   */
	bool Initialized() const {
		return initialized_;
	}

  /**
   * Report observation along with Map of landmarks.
   *
   * @param observations Noisy observations from sensors.
   * @param map_landmarks Map with landmarks.
   */
	void ReportObservation(const std::vector<LandmarkObs> &observations, const Map &map_landmarks);


  const std::vector<Particle> &GetParticles() const;
	std::string getAssociations(Particle best);
	std::string getSenseX(Particle best);
	std::string getSenseY(Particle best);

private:
  // Set of current particles
  std::vector<Particle> particles_;

  // Random engine
  std::default_random_engine generator_;

  // Motion model
  MotionModel motion_model_;

  // Observation model
  ObservationModel observation_model_;

  // Range for sensor.
  double sensor_range_;

	// Whether the filter is initialized.
	bool initialized_;
};



#endif /* PARTICLE_FILTER_H_ */
