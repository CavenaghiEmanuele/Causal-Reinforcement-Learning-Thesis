#pragma once

#include "../temporal_difference.ipp"

namespace reilly {

namespace agents {

class DoubleTemporalDifference : public TemporalDifference {
   protected:
    ActionValue Q2;
    Policy pi2;

   public:
    DoubleTemporalDifference(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay);
    DoubleTemporalDifference(const DoubleTemporalDifference &other);
    virtual ~DoubleTemporalDifference();

    void reset(size_t init_state);
};

}  // namespace agents

}  // namespace reilly
