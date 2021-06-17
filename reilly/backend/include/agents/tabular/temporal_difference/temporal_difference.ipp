#pragma once

#include "temporal_difference.hpp"

namespace reilly {

namespace agents {

TemporalDifference::TemporalDifference(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay)
    : TabularAgent(states, actions, alpha, epsilon, gamma, epsilon_decay) {}

TemporalDifference::TemporalDifference(const TemporalDifference &other) : TabularAgent(other) {}

TemporalDifference::~TemporalDifference() {}

void TemporalDifference::reset(size_t init_state) {
    state = init_state;
    action = select_action(pi, init_state);
}

}  // namespace agents

}  // namespace reilly
