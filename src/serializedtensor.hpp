#pragma once
#ifndef SERIALIZED_TENSOR
#define SERIALIZED_TENSOR

#include <boost/mpi.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <torch/torch.h>

class SerializedTensorCPU_impl {
  bool has_grad = false;
  int64_t num_bytes = 0;
  int8_t scalar_type{};
  std::vector<int64_t> sizes;

  torch::Tensor cont_tensor;

public:
  SerializedTensorCPU_impl() = default;

  explicit SerializedTensorCPU_impl(const torch::Tensor &t) {
    cont_tensor = t.contiguous();

    sizes = cont_tensor.sizes().vec();
    num_bytes = cont_tensor.numel() * cont_tensor.element_size();
    scalar_type = static_cast<int8_t>(cont_tensor.scalar_type());


    if (cont_tensor.mutable_grad().numel() != 0)
      has_grad = true;
  }

  void toTensor(torch::Tensor &outTensor) {
    outTensor = std::move(cont_tensor);
  }

private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & has_grad;
    ar & num_bytes;
    ar & sizes;
    ar & scalar_type;
    ar & boost::serialization::make_array(static_cast<char *>(cont_tensor.data_ptr()), num_bytes);
  }
};


#endif
