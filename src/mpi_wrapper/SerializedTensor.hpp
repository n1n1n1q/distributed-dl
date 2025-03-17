#pragma once
#ifndef SERIALIZED_TENSOR
#define SERIALIZED_TENSOR

#include <boost/mpi.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <torch/torch.h>


class SerializedTensorBase {
public:
    virtual ~SerializedTensorBase() = default;

    virtual void toTensor(torch::Tensor &outTensor) const = 0;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & /* ar */, const unsigned int /* version */) {
    }
};


class SerializedTensorCPU_impl : public SerializedTensorBase {
    bool has_grad = false;
    int64_t num_bytes = 0;
    std::vector<int64_t> sizes;
    std::vector<char> tensor_data;
    std::vector<char> grad_data;

public:
    SerializedTensorCPU_impl() = default;

    SerializedTensorCPU_impl(const torch::Tensor &t) {
        auto cont_tensor = t.contiguous();

        sizes = cont_tensor.sizes().vec();
        num_bytes = cont_tensor.numel() * cont_tensor.element_size();

        tensor_data.resize(num_bytes);
        std::memcpy(tensor_data.data(), cont_tensor.data_ptr(), num_bytes);

        if (cont_tensor.grad().numel() != 0) {
            has_grad = true;
            grad_data.resize(num_bytes);
            std::memcpy(grad_data.data(), cont_tensor.grad().data_ptr(), num_bytes);
        }
    }

    void toTensor(torch::Tensor &outTensor) {
        outTensor = torch::from_blob(
            tensor_data.data(),
            sizes,
            outTensor.scalar_type()
        ).clone();
        if (has_grad) {
            outTensor.mutable_grad() = torch::from_blob(
                grad_data.data(),
                sizes,
                outTensor.scalar_type()
            ).clone();
        }
    }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & has_grad;
        ar & num_bytes;
        ar & sizes;
        ar & tensor_data;
        ar & grad_data;
    }
};


class SerializedTensor {
private:
    std::unique_ptr<SerializedTensorCPU_impl> tensor_impl;

public:
    SerializedTensor() = default;


    SerializedTensor(torch::Tensor &t) {
        if (t.is_cpu()) tensor_impl = std::make_unique<SerializedTensorCPU_impl>(t);
    }

    void toTensor(torch::Tensor &outTensor) {
        tensor_impl->toTensor(outTensor);
    }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int /* version */) {
        ar & tensor_impl;
    }
};

#endif
