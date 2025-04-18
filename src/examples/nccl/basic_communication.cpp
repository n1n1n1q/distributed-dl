#include "../../nccl_distr.cuh"
#include <torch/torch.h>
#include <iostream>
#include <vector>

void test_send_recv(ProcessGroupNCCL &pg)
{
    const int world_size = pg.size();
    const int rank = pg.rank();

    if (world_size < 2)
    {
        if (rank == 0)
            std::cerr << "Send/Recv test requires at least 2 processes." << std::endl;
        pg.barrier();
        return;
    }

    if (rank == 0)
    {
        torch::Tensor tensor = torch::ones({2, 2}, torch::kFloat32).to(torch::kCUDA);
        std::cout << "Rank 0 sending tensor:\n"
                  << tensor << std::endl;
        pg.send(tensor, 1);
    }
    else if (rank == 1)
    {
        torch::Tensor tensor = torch::zeros({2, 2}, torch::kFloat32).to(torch::kCUDA);
        pg.recv(tensor, 0);
        std::cout << "Rank 1 received tensor:\n"
                  << tensor << std::endl;
    }
    else
    {
        std::cout << "Rank " << rank << " didn't receive or send anything" << std::endl;
    }
    pg.barrier();
}

void test_broadcast(ProcessGroupNCCL &pg)
{
    const int rank = pg.rank();
    torch::Tensor tensor;
    if (rank == 0)
    {
        tensor = torch::arange(0, 4, torch::kFloat32).reshape({2, 2}).to(torch::kCUDA);
        std::cout << "Rank 0 broadcasting tensor:\n"
                  << tensor << std::endl;
    }
    else
    {
        tensor = torch::zeros({2, 2}, torch::kFloat32).to(torch::kCUDA);
    }

    pg.broadcast(tensor, 0);

    std::cout << "Rank " << rank << " tensor after broadcast:\n"
              << tensor << std::endl;
    pg.barrier();
}

int main(int argc, char **argv)
{
    ProcessGroupNCCL pg(argc, argv);

    test_send_recv(pg);
    test_broadcast(pg);

    return 0;
}