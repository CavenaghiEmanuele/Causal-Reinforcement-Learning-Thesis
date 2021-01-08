#pragma once

#include "double_temporal_difference.ipp"

namespace reilly {

namespace agents {

class DoubleSarsa : public DoubleTemporalDifference {
   public:
    DoubleSarsa(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay = 1);
    DoubleSarsa(const DoubleSarsa &other);
    DoubleSarsa &operator=(const DoubleSarsa &other);
    virtual ~DoubleSarsa();

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs);
};

}  // namespace agents

}  // namespace reilly
