#include "mpi_wrapper/mpi_send_tensor.hpp"
// tmp file
int main(int argc, char **argv) {
    std::cout << "Hello, World!" << std::endl;

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    if (world.rank() == 0) {
        torch::Tensor t = torch::randn({3, 4});
        SerializedTensor wrapper(t);
        world.send(1, 0, wrapper);
    } else {
        SerializedTensor wrapper;
        world.recv(0, 0, wrapper);
        std::cout << "Received tensor:\n" << wrapper.tensor << std::endl;
    }

    return 0;
}
