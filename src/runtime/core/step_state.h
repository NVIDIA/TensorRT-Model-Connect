#pragma once

namespace trtmc {

// Opaque base for per-step state during autoregressive generation.
// Recurrent and hybrid models own their concrete step state in family-local
// runtime code.
class IStepState {
  public:
    virtual ~IStepState() = default;
};

} // namespace trtmc
