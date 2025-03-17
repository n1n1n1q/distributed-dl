#pragma once
#ifndef SERIALIZED_TENSOR
#define SERIALIZED_TENSOR

#include <boost/mpi.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <torch/torch.h>

class SerializedTensorBase {
};

class SerializedTensorCPU_impl : public SerializedTensorBase {
public:
    int64_t num_bytes;
    std::vector<char> data;

    SerializedTensorCPU_impl() = default;

    SerializedTensorCPU_impl(const torch::Tensor &t) {
        auto cont_tensor = t.contiguous();

        if (cont_tensor.retains_grad())std::cout << "gradient is:\n" << cont_tensor.grad() << std::endl;

        num_bytes = cont_tensor.numel() * cont_tensor.element_size();
        data.resize(num_bytes);
        std::memcpy(data.data(), cont_tensor.data_ptr(), num_bytes);
    }

    void toTensor(torch::Tensor &outTensor) const {
        std::memcpy(outTensor.data_ptr(), data.data(), num_bytes);
    }

private:
    friend class boost::serialization::access;

    template<typename T>
    void save(T &ar, const unsigned int version) const {
        ar & num_bytes;
        ar & data;
    }

    template<typename Archive>
    void load(Archive &ar, const unsigned int version) {
        data.resize(num_bytes);

        ar & num_bytes;
        ar & data;
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

namespace boost::mpi {
    template<>
    struct is_mpi_datatype<SerializedTensorCPU_impl> : mpl::false_ {
    };
}

class SerializedTensor {
    SerializedTensorBase tensor;

    SerializedTensor(torch::Tensor &t) {
        if (t.is_cpu()) tensor = SerializedTensorCPU_impl(t);
    }
};

#endif
