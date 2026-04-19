#ifndef SCHDULES_H
#define SCHDULES_H

#include <functional>
#include <tensor.h>

namespace sch {
    // linear beta scheduler
    inline Tensor linear_beta_schedule(int timesteps) {
        float scale = 1000.0f / timesteps;
        float beta_start = scale * 0.0001f;
        float beta_end = scale * 0.02f;
        Tensor beta_schedule = Tensor::from_shape({timesteps});
        for (int i = 1; i <= timesteps; i++) {
            beta_schedule[i-1] = beta_start + i * (beta_end - beta_start) / timesteps;
        }
        return beta_schedule;
    }
}

#endif