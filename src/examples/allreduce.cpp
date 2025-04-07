#include "distributed.hpp"
#include <torch/torch.h>
#include <iostream>

void all_reduceTest(ProcessGroupMPI &pg, int root = 0)
{
    int rank = pg.rank();
    int size = pg.size();

    torch::Tensor tensor = torch::full({2, 2}, rank, torch::kFloat32);
    pg.all_reduce(tensor, root);

    float expected_value = (size * (size - 1)) / 2.0f;
    torch::Tensor expected = torch::full({2, 2}, expected_value, torch::kFloat32);

    if (rank == root)
    {
        std::cout << "Rank " << rank << " All-Reduced Tensor:\n"
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

    if (rank == root)
    {
        if (size > 1)
        {
            pg.send(tensor, 1);
        }
    }
    else
    {
        torch::Tensor dummy_tensor = torch::zeros_like(tensor);
        pg.recv(dummy_tensor, rank - 1);
        if (rank < size - 1)
        {
            pg.send(tensor, rank + 1);
        }
    }
}

int main(int argc, char **argv)
{
    ProcessGroupMPI pg(argc, argv);
    all_reduceTest(pg);
    return 0;
}
