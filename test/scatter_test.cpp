#include "serializedtensor.hpp"
#include "distributed.hpp"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <torch/torch.h>

int main(int argc, char** argv) {
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    const int root = 0;
    torch::Tensor local_tensor;

    if (world.rank() == root) {
        local_tensor = torch::full({8, 8}, world.rank(), torch::kFloat32).cpu();
    } else {
        local_tensor = torch::empty({1, 8}, torch::kFloat32).cpu();
    }

    scatter(world, local_tensor, root);

    if (world.rank() == root) {
        std::cout << "Rank " << world.rank() << " Tensor:\n" << local_tensor << std::endl;
        if (world.size() > 1) {
            world.send(1, 0, 0);
        }
    } else {
        int token;
        world.recv(world.rank() - 1, 0, token);
        std::cout << "Rank " << world.rank() << " Tensor:\n" << local_tensor << std::endl;
        if (world.rank() < world.size() - 1) {
            world.send(world.rank() + 1, 0, 0);
        }
    }

    return 0;
}