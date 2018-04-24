//
// Created by Rohith Menon on 4/22/18.
//

#ifndef PARTICLE_FILTER_PARTICLE_H
#define PARTICLE_FILTER_PARTICLE_H


#include <iostream>
#include <vector>
#include "Eigen/Dense"

class Particle {
public:
  Particle(int id, const Eigen::VectorXd& state, double weight)
      : id_(id), weight_(weight), state_(state) { }

  int GetId() const {
    return id_;
  }

  double GetWeight() const {
    return weight_;
  }

  const Eigen::VectorXd &GetState() const {
    return state_;
  }

  bool operator <(const Particle& other) const {
    return id_ < other.id_;
  }

  bool operator ==(const Particle& other) const {
    return id_ == other.id_;
  }

  friend std::ostream &operator<<(std::ostream &os, const Particle& particle) {
    return os << "[" << particle.id_ << "," << particle.state_ << "," << particle.weight_ << "]";
  }

  std::vector<int> associations;
  std::vector<double> sense_x;
  std::vector<double> sense_y;

private:
  int id_;
  double weight_;
  Eigen::VectorXd state_;
};
#endif //PARTICLE_FILTER_PARTICLE_H
