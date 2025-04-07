#include "distributed.hpp"
#include <torch/torch.h>
#include <iostream>

void reduceTest(ProcessGroupMPI &pg, int root = 0)
{
    int rank = pg.rank();
    int size = pg.size();
    torch::Tensor tensor = torch::full({2, 2}, rank, torch::kFloat32);
    pg.reduce(tensor, root);
    float expected_val = (size * (size - 1)) / 2.0f;
    torch::Tensor expected = torch::full({2, 2}, expected_val, torch::kFloat32);
    if (rank == 0)
    {
        if (rank == root)
        {
            std::cout << "Rank " << rank << " Reduced Tensor:\n"
                      << tensor << std::endl;
            std::cout << "Expected:\n"
                      << expected << "\n"
                      << std::endl;
        }
        else
        {
            std::cout << "Rank " << rank << " Tensor (unchanged):\n"
                      << tensor << "\n"
                      << std::endl;
        }
        if (size > 1)
            pg.world_ptr->send(1, 0, 0);
    }
    else
    {
        int token;
        pg.world_ptr->recv(rank - 1, 0, token);

        if (rank == root)
        {
            std::cout << "Rank " << rank << " Reduced Tensor:\n"
                      << tensor << std::endl;
            std::cout << "Expected:\n"
                      << expected << "\n"
                      << std::endl;
        }
        else
        {
            std::cout << "Rank " << rank << " Tensor (unchanged):\n"
                      << tensor << "\n"
                      << std::endl;
        }

        if (rank < size - 1)
            pg.world_ptr->send(rank + 1, 0, 0);
    }
}

int main(int argc, char **argv)
{
    ProcessGroupMPI pg(argc, argv);
    reduceTest(pg);
    return 0;
}
