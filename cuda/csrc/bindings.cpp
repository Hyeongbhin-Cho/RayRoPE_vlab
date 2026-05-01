// cuda/csrc/bindings.cpp
#include <torch/extension.h>
#include "rayrope.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rayrope_coeffs_fwd", &rayrope::fused_rayrope_coeffs_fwd, "Fused RayRoPE Forward");
    m.def("fused_rayrope_coeffs_bwd", &rayrope::fused_rayrope_coeffs_bwd, "Fused RayRoPE Backward");
    
    m.def("rope2D_coeffs_fwd", &rayrope::rope2D_coeffs_fwd, "RoPE 2D Forward");
    m.def("rope2D_coeffs_bwd", &rayrope::rope2D_coeffs_bwd, "RoPE 2D Backward");
}