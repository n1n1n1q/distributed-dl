#include <boost/mpi.hpp>
#include <boost/serialization/split_member.hpp>
#include <torch/torch.h>

class TensorWrapper {
public:
    torch::Tensor tensor;

    TensorWrapper() = default;
    explicit TensorWrapper(const torch::Tensor& t) 
        : tensor(t.device().is_cpu() ? t.contiguous() : t.to(torch::kCPU).contiguous()) {}

private:
    friend class boost::serialization::access;

    template <typename Archive>
    void save(Archive& ar, const unsigned int version) const {

        const torch::Tensor& cont_tensor = tensor;
        const auto sizes = cont_tensor.sizes().vec();
        const int64_t ndims = static_cast<int64_t>(sizes.size());
        const int dtype_int = static_cast<int>(cont_tensor.scalar_type());
        const size_t num_bytes = cont_tensor.numel() * cont_tensor.element_size();

        ar & ndims;
        ar & boost::serialization::make_array(sizes.data(), ndims);
        ar & dtype_int;
        ar & num_bytes;
        ar & boost::serialization::make_array(
            static_cast<const char*>(cont_tensor.data_ptr()), 
            num_bytes
        );
    }

    template <typename Archive>
    void load(Archive& ar, const unsigned int version) {
        int64_t ndims;
        ar & ndims;

        std::vector<int64_t> sizes(ndims);
        ar & boost::serialization::make_array(sizes.data(), ndims);

        int dtype_int;
        ar & dtype_int;
        const auto dtype = static_cast<torch::ScalarType>(dtype_int);

        size_t num_bytes;
        ar & num_bytes;

        tensor = torch::empty(sizes, torch::dtype(dtype));
        const size_t expected_bytes = tensor.numel() * tensor.element_size();

        if (num_bytes != expected_bytes) {
            throw std::runtime_error("Tensor data size mismatch during deserialization");
        }

        ar & boost::serialization::make_array(
            static_cast<char*>(tensor.data_ptr()), 
            num_bytes
        );
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

namespace boost::mpi {
    template<> struct is_mpi_datatype<TensorWrapper> : mpl::false_ {};
}

int main(int argc, char** argv) {
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    if (world.rank() == 0) {
        torch::Tensor t = torch::randn({3, 4});
        TensorWrapper wrapper(t);
        world.send(1, 0, wrapper);
    } else {
        TensorWrapper wrapper;
        world.recv(0, 0, wrapper);
        std::cout << "Received tensor:\n" << wrapper.tensor << std::endl;
    }

    return 0;
}