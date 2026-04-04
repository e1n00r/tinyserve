#include <torch/extension.h>

namespace {

// Convert Python int (from _DTYPE_TO_INT) to c10::ScalarType
inline c10::ScalarType to_scalar_type(int64_t v) {
  return static_cast<c10::ScalarType>(v);
}

torch::Tensor silu_gate_forward(
    const torch::Tensor& hidden_states,
    const torch::Tensor& w_gu,
    const torch::Tensor& w_dn,
    const c10::optional<torch::Tensor>& b_gu,
    const c10::optional<torch::Tensor>& b_dn) {
  auto gate_up = torch::linear(hidden_states, w_gu,
                                b_gu.has_value() ? *b_gu : torch::Tensor());
  auto chunks = gate_up.chunk(2, /*dim=*/-1);
  auto gated = torch::silu(chunks[0]) * chunks[1];
  return torch::linear(gated, w_dn, b_dn.has_value() ? *b_dn : torch::Tensor());
}

torch::Tensor swiglu_forward(
    const torch::Tensor& hidden_states,
    const torch::Tensor& w_gu,
    const torch::Tensor& w_dn,
    const c10::optional<torch::Tensor>& b_gu,
    const c10::optional<torch::Tensor>& b_dn) {
  auto gate_up = torch::linear(hidden_states, w_gu,
                                b_gu.has_value() ? *b_gu : torch::Tensor());
  // GPT-OSS interleaved SwiGLU: even indices = gate, odd = up
  using namespace torch::indexing;
  auto gate = gate_up.index({"...", Slice(None, None, 2)}).clamp_max(7.0);
  auto up = gate_up.index({"...", Slice(1, None, 2)}).clamp(-7.0, 7.0);
  auto gated = (up + 1) * gate * torch::sigmoid(gate * 1.702);  // Quick GELU coefficient (matches _QUICK_GELU_COEFF in expert_forward.py)
  return torch::linear(gated, w_dn, b_dn.has_value() ? *b_dn : torch::Tensor());
}

} // namespace

torch::Tensor fast_expert_forward(
    torch::Tensor hidden_states,
    torch::Tensor expert_slots,
    torch::Tensor routing_weights,
    torch::Tensor cache_packed,
    int64_t gu_offset,
    int64_t gu_size,
    std::vector<int64_t> gu_shape,
    int64_t gu_dtype_int,
    bool gu_needs_transpose,
    int64_t dn_offset,
    int64_t dn_size,
    std::vector<int64_t> dn_shape,
    int64_t dn_dtype_int,
    bool dn_needs_transpose,
    bool has_bias,
    int64_t gub_offset,
    int64_t gub_size,
    std::vector<int64_t> gub_shape,
    int64_t gub_dtype_int,
    int64_t dnb_offset,
    int64_t dnb_size,
    std::vector<int64_t> dnb_shape,
    int64_t dnb_dtype_int,
    const std::string& activation) {

  auto output = torch::zeros_like(hidden_states);
  int top_k = expert_slots.size(0);

  auto gu_dtype = to_scalar_type(gu_dtype_int);
  auto dn_dtype = to_scalar_type(dn_dtype_int);
  auto gub_dtype = to_scalar_type(gub_dtype_int);
  auto dnb_dtype = to_scalar_type(dnb_dtype_int);

  // Pre-move slots/weights to CPU for scalar access (avoid per-iteration sync)
  auto slots_cpu = expert_slots.to(torch::kCPU);
  auto weights_cpu = routing_weights.to(torch::kCPU);
  auto slots_a = slots_cpu.accessor<int32_t, 1>();
  auto weights_a = weights_cpu.accessor<float, 1>();

  bool use_silu = (activation == "silu");

  for (int i = 0; i < top_k; i++) {
    int slot = slots_a[i];
    if (slot < 0) continue;

    float weight = weights_a[i];
    auto packed = cache_packed[slot];

    // Extract gate_up weight (zero-copy view into cache)
    auto w_gu = packed.slice(0, gu_offset, gu_offset + gu_size)
                    .view(gu_dtype)
                    .reshape(gu_shape);
    if (gu_needs_transpose) w_gu = w_gu.t();

    // Extract down_proj weight
    auto w_dn = packed.slice(0, dn_offset, dn_offset + dn_size)
                    .view(dn_dtype)
                    .reshape(dn_shape);
    if (dn_needs_transpose) w_dn = w_dn.t();

    // Extract biases if present
    c10::optional<torch::Tensor> b_gu = c10::nullopt;
    c10::optional<torch::Tensor> b_dn = c10::nullopt;
    if (has_bias) {
      b_gu = packed.slice(0, gub_offset, gub_offset + gub_size)
                 .view(gub_dtype)
                 .reshape(gub_shape);
      b_dn = packed.slice(0, dnb_offset, dnb_offset + dnb_size)
                 .view(dnb_dtype)
                 .reshape(dnb_shape);
    }

    torch::Tensor out;
    if (use_silu) {
      out = silu_gate_forward(hidden_states, w_gu, w_dn, b_gu, b_dn);
    } else {
      out = swiglu_forward(hidden_states, w_gu, w_dn, b_gu, b_dn);
    }
    output += weight * out;
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_expert_forward", &fast_expert_forward,
        "Fast expert forward loop — eliminates Python dispatch between experts");
}
