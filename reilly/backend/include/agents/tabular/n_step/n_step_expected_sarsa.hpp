#pragma once

#include "n_step.ipp"

namespace reilly {

namespace agents {

class NStepExpectedSarsa : public NStep {
   public:
    NStepExpectedSarsa(size_t states, size_t actions, float alpha, float epsilon, float gamma, size_t n_step, float epsilon_decay = 1);
    NStepExpectedSarsa(const NStepExpectedSarsa &other);
    NStepExpectedSarsa &operator=(const NStepExpectedSarsa &other);
    virtual ~NStepExpectedSarsa();

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs);
};

}  // namespace agents

}  // namespace reilly
