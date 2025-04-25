#include "../../nccl_distr.cuh"
#include <torch/torch.h>
#include <iostream>

void scatter_test(ProcessGroupNCCL &pg, int root = 0)
{
    int rank = pg.rank();
    int size = pg.size();
    
    torch::Tensor send_tensor;
    if (rank == root) {
        send_tensor = torch::zeros({size, 2, 2}, torch::kFloat32).to(torch::kCUDA);
        for (int i = 0; i < size; i++) {
            send_tensor[i].fill_(i * 10);
        }
        std::cout << "Rank " << rank << " sending tensor:\n" << send_tensor << std::endl;
    } else {
        send_tensor = torch::zeros({1, 2, 2}, torch::kFloat32).to(torch::kCUDA);
    }
    
    torch::Tensor recv_tensor = torch::zeros({2, 2}, torch::kFloat32).to(torch::kCUDA);
    
    pg.scatter(recv_tensor, send_tensor, root);
    
    std::cout << "Rank " << rank << " received tensor:\n" << recv_tensor << std::endl;
    
    torch::Tensor expected = torch::full({2, 2}, rank * 10, torch::kFloat32).to(torch::kCUDA);
    if (torch::allclose(recv_tensor, expected)) {
        std::cout << "Rank " << rank << " received correct data" << std::endl;
    } else {
        std::cout << "Rank " << rank << " error: expected\n" << expected 
                  << "\nbut got\n" << recv_tensor << std::endl;
    }
    
    pg.barrier();
}

int main(int argc, char **argv)
{
    ProcessGroupNCCL pg(argc, argv);
    scatter_test(pg);
    return 0;
}