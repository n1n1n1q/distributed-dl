#include "../../nccl_distr.cuh"
#include <torch/torch.h>
#include <iostream>

void reduce_test(ProcessGroupNCCL &pg, int root = 0)
{
    int rank = pg.rank();
    int size = pg.size();

    torch::Tensor tensor = torch::full({2, 2}, rank, torch::kFloat32).to(torch::kCUDA);
    std::cout << "Rank " << rank << " before reduce tensor:\n"
              << tensor << std::endl;
    
    pg.reduce(tensor, root);

    if (rank == root) {
        float expected_value = (size * (size - 1)) / 2.0f;
        torch::Tensor expected = torch::full({2, 2}, expected_value, torch::kFloat32).to(torch::kCUDA);
        
        std::cout << "Rank " << rank << " after reduce tensor:\n"
                  << tensor << std::endl;
        std::cout << "Expected:\n"
                  << expected << std::endl;
    } else {
        std::cout << "Rank " << rank << " after reduce (should be unchanged locally):\n"
                  << tensor << std::endl;
    }
    
    pg.barrier();
}

int main(int argc, char **argv)
{
    ProcessGroupNCCL pg(argc, argv);
    reduce_test(pg);
    return 0;
}