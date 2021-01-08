#pragma once

#include "sarsa.hpp"

namespace reilly {

namespace agents {

Sarsa::Sarsa(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay)
    : TemporalDifference(states, actions, alpha, epsilon, gamma, epsilon_decay) {}

Sarsa::Sarsa(const Sarsa &other) : TemporalDifference(other) {}

Sarsa &Sarsa::operator=(const Sarsa &other) {
    if (this != &other) {
        Sarsa tmp(other);
        std::swap(tmp.states, states);
        std::swap(tmp.actions, actions);
        std::swap(tmp.Q, Q);
        std::swap(tmp.pi, pi);
        std::swap(tmp.alpha, alpha);
        std::swap(tmp.epsilon, epsilon);
        std::swap(tmp.gamma, gamma);
        std::swap(tmp.epsilon_decay, epsilon_decay);
        std::swap(tmp.state, state);
        std::swap(tmp.action, action);
    }
    return *this;
}

Sarsa::~Sarsa() {}

void Sarsa::update(size_t next_state, float reward, bool done, py::kwargs kwargs) {
    size_t next_action = select_action(pi, next_state);

    bool training = py::cast<bool>(kwargs["training"]);
    if (training) {
        Q(state, action) += alpha * (reward + gamma * Q(next_state, next_action) - Q(state, action));
        policy_update(Q, pi, state);
    }

    state = next_state;
    action = next_action;

    if (done) epsilon *= epsilon_decay;
}

}  // namespace agents

}  // namespace reilly
