#pragma once

#include "montecarlo.hpp"

namespace reilly {

namespace agents {

MonteCarlo::MonteCarlo(size_t states, size_t actions, float epsilon, float gamma, float epsilon_decay)
    : TabularAgent(states, actions, 0, epsilon, gamma, epsilon_decay) {
    returns = xt::zeros<float>({states, actions});
}

MonteCarlo::MonteCarlo(const MonteCarlo &other) : TabularAgent(other), trajectory(other.trajectory), returns(other.returns) {}

MonteCarlo::~MonteCarlo() {}

void MonteCarlo::reset(size_t init_state) {
    state = init_state;
    action = select_action(pi, init_state);
    trajectory.clear();
}

void MonteCarlo::update(size_t next_state, float reward, bool done, py::kwargs kwargs) {
    trajectory.push_back({state, action, reward});
    
    bool training = py::cast<bool>(kwargs["training"]);
    if (done && training) control();

    state = next_state;
    action = select_action(pi, next_state);

    if (done) epsilon *= epsilon_decay;
}

std::string MonteCarlo::__repr__() {
    int status;
    std::stringstream out;
    char *demangled = abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status);
    out << "<" << demangled << "(epsilon=" << epsilon;
    out << ", gamma=" << gamma << ", epsilon_decay=" << epsilon_decay << ")>";
    return out.str();
}

}  // namespace agents

}  // namespace reilly
