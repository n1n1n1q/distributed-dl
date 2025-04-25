#include "../../nccl_distr.cuh"
#include <torch/torch.h>
#include <iostream>

void all_reduce_test(ProcessGroupNCCL &pg)
{
    int rank = pg.rank();
    int size = pg.size();

    torch::Tensor tensor = torch::full({2, 2}, rank, torch::kFloat32).to(torch::kCUDA);
    std::cout << "Rank " << rank << " before all-reduce tensor:\n"
              << tensor << std::endl;
    
    pg.all_reduce(tensor);

    float expected_value = (size * (size - 1)) / 2.0f;
    torch::Tensor expected = torch::full({2, 2}, expected_value, torch::kFloat32).to(torch::kCUDA);

    std::cout << "Rank " << rank << " after all-reduce tensor:\n"
              << tensor << std::endl;
    std::cout << "Expected:\n"
              << expected << std::endl;
    
    pg.barrier();
}

int main(int argc, char **argv)
{
    ProcessGroupNCCL pg(argc, argv);
    all_reduce_test(pg);
    return 0;
}