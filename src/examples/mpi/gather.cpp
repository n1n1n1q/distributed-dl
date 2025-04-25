#include "distributed.hpp"
#include <torch/torch.h>
#include <iostream>
#include <vector>

void gatherTest(ProcessGroupMPI &pg, int root = 0)
{
    std::vector<torch::Tensor> gathered_tensors;
    torch::Tensor local_tensor = torch::full({2, 2}, pg.rank(), torch::kFloat32);
    pg.gather(local_tensor, gathered_tensors, root);
    if (pg.rank() == root)
    {
        for (int i = 0; i < pg.size(); ++i)
        {
            auto expected = torch::full({2, 2}, i, torch::kFloat32);
            std::cout << "Expected:" << std::endl
                      << expected << std::endl;
            std::cout << "Got:" << std::endl
                      << gathered_tensors[i] << std::endl;
        }
        std::cout << "Gather test passed on root." << std::endl;
    }
}

int main(int argc, char **argv)
{
    ProcessGroupMPI pg(argc, argv);
    gatherTest(pg);
    return 0;
}
