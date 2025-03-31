#include "serializedtensor.hpp"
#include "distributed.hpp"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <torch/torch.h>


int main(int argc, char** argv) {
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    const int root = 0;
    std::vector<torch::Tensor> gathered_tensors;
    torch::Tensor local_tensor;

    local_tensor = torch::full({2, 2}, world.rank(), torch::kFloat32);

    gather(world, local_tensor, gathered_tensors, root);

    if (world.rank() == root) {

        for (int i = 0; i < world.size(); ++i) {
            auto expected = torch::full({2, 2}, i, torch::kFloat32);
            std::cout << "Expected: " << '\n' << expected << std::endl;
            std::cout << "Got: " << '\n' << gathered_tensors[i] <<std::endl;
        }
        std::cout << "Gather test passed on root." << std::endl;
    }

    return 0;
}