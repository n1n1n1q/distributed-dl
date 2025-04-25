#include "../../nccl_distr.cuh"
#include <torch/torch.h>
#include <iostream>

void gather_test(ProcessGroupNCCL &pg, int root = 0)
{
    int rank = pg.rank();
    int size = pg.size();

    torch::Tensor send_tensor = torch::full({2, 2}, rank, torch::kFloat32).to(torch::kCUDA);
    std::cout << "Rank " << rank << " send tensor:\n" << send_tensor << std::endl;
    
    torch::Tensor recv_tensor;
    if (rank == root) {
        recv_tensor = torch::zeros({size, 2, 2}, torch::kFloat32).to(torch::kCUDA);
    } else {
        recv_tensor = torch::zeros({1, 2, 2}, torch::kFloat32).to(torch::kCUDA);
    }
    
    pg.gather(send_tensor, recv_tensor, root);
    
    if (rank == root) {
        std::cout << "Rank " << rank << " gathered tensor:\n" << recv_tensor << std::endl;
        
        bool correct = true;
        for (int i = 0; i < size; i++) {
            torch::Tensor expected = torch::full({2, 2}, i, torch::kFloat32).to(torch::kCUDA);
            torch::Tensor gathered = recv_tensor[i];
            if (!torch::allclose(gathered, expected)) {
                correct = false;
                std::cout << "Error at index " << i << ": expected\n" << expected 
                          << "\nbut got\n" << gathered << std::endl;
            }
        }
        
        if (correct) {
            std::cout << "Gather operation successful!" << std::endl;
        }
    }
    
    pg.barrier();
}

int main(int argc, char **argv)
{
    ProcessGroupNCCL pg(argc, argv);
    gather_test(pg);
    return 0;
}