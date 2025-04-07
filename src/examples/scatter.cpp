#include "distributed.hpp"
#include <torch/torch.h>
#include <iostream>

void scatterTest(ProcessGroupMPI &pg, int root = 0)
{
    torch::Tensor local_tensor;

    if (pg.rank() == root)
    {
        local_tensor = torch::full({8, 8}, pg.rank(), torch::kFloat32).cpu();
    }
    else
    {
        local_tensor = torch::empty({1, 8}, torch::kFloat32).cpu();
    }

    pg.scatter(local_tensor, root);

    if (pg.rank() == root)
    {
        std::cout << "Rank " << pg.rank() << " Tensor:\n"
                  << local_tensor << std::endl;
    }
    else
    {
        std::cout << "Rank " << pg.rank() << " Tensor:\n"
                  << local_tensor << std::endl;
    }
}

int main(int argc, char **argv)
{
    ProcessGroupMPI pg(argc, argv);

    scatterTest(pg);

    return 0;
}
