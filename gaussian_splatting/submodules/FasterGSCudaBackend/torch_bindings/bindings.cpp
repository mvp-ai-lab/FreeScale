#include <torch/extension.h>
#include "adam.h"

namespace adam_api = faster_gs::adam;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adam_step", &adam_api::adam_step_wrapper);
}
