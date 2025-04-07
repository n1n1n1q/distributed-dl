#pragma once
#ifndef SERIALIZED_TENSOR
#define SERIALIZED_TENSOR

#include <boost/mpi.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <torch/torch.h>

class SerializedTensorCPU_impl
{
    bool has_grad = false;
    int64_t num_bytes = 0;
    int8_t scalar_type;
    std::vector<int64_t> sizes;
    std::vector<char> tensor_data;
    std::vector<char> grad_data;

public:
    SerializedTensorCPU_impl() = default;

    SerializedTensorCPU_impl(const torch::Tensor &t)
    {
        auto cont_tensor = t.contiguous();

        sizes = cont_tensor.sizes().vec();
        num_bytes = cont_tensor.numel() * cont_tensor.element_size();
        scalar_type = static_cast<int8_t>(cont_tensor.scalar_type());

        tensor_data.resize(num_bytes);
        std::memcpy(tensor_data.data(), cont_tensor.data_ptr(), num_bytes);

        if (cont_tensor.mutable_grad().numel() != 0)
        {
            has_grad = true;
            grad_data.resize(num_bytes);
            std::memcpy(grad_data.data(), cont_tensor.grad().data_ptr(), num_bytes);
        }
    }

    void toTensor(torch::Tensor &outTensor)
    {
        outTensor = torch::from_blob(
                        tensor_data.data(),
                        sizes,
                        static_cast<torch::ScalarType>(scalar_type))
                        .clone();
        if (has_grad)
        {
            outTensor.mutable_grad() = torch::from_blob(
                                           grad_data.data(),
                                           sizes,
                                           static_cast<torch::ScalarType>(scalar_type))
                                           .clone();
        }
    }

private:
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & has_grad;
        ar & num_bytes;
        ar & sizes;
        ar & tensor_data;
        ar & grad_data;
        ar & scalar_type;
    }
};

#endif
