#include "serializedtensor.hpp"
#include "distributed.hpp"
#include <torch/torch.h>
#include <iostream>
#include <vector>

void test_send_recv(ProcessGroupMPI &pg)
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
        torch::Tensor tensor = torch::ones({2, 2}, torch::kFloat32);
        std::cout << "Rank 0 sending tensor:\n"
                  << tensor << std::endl;
        pg.send(tensor, 1);
    }
    else if (rank == 1)
    {
        torch::Tensor tensor = torch::zeros({2, 2}, torch::kFloat32);
        pg.recv(tensor, 0);
        std::cout << "Rank 1 received tensor:\n"
                  << tensor << std::endl;
    }
    else
    {
        std::cout << "Rank " << rank << " didn't recieve or send anything" << std::endl;
    }
    pg.barrier();
}

void test_broadcast(ProcessGroupMPI &pg)
{
    const int rank = pg.rank();
    torch::Tensor tensor;
    if (rank == 0)
    {
        tensor = torch::arange(0, 4, torch::kFloat32).reshape({2, 2});
        std::cout << "Rank 0 broadcasting tensor:\n"
                  << tensor << std::endl;
    }
    else
    {
        tensor = torch::zeros({2, 2}, torch::kFloat32);
    }

    pg.broadcast(tensor, 0);

    std::cout << "Rank " << rank << " tensor after broadcast:\n"
              << tensor << std::endl;
    pg.barrier();
}

void test_async_send_recv(ProcessGroupMPI &pg)
{
    const int world_size = pg.size();
    const int rank = pg.rank();

    if (world_size < 2)
    {
        if (rank == 0)
            std::cerr << "Async send/recv test requires at least 2 processes." << std::endl;
        pg.barrier();
        return;
    }

    if (rank == 0)
    {
        torch::Tensor tensor = torch::full({2, 2}, 5, torch::kFloat32);
        std::cout << "Rank 0 asynchronously sending tensor:\n"
                  << tensor << std::endl;
        auto request = pg.isend(tensor, 1);
        request.wait();
    }
    else if (rank == 1)
    {
        torch::Tensor tensor = torch::zeros({2, 2}, torch::kFloat32);
        auto request = pg.irecv(tensor, 0);
        request.wait();
        std::cout << "Rank 1 asynchronously received tensor:\n"
                  << tensor << std::endl;
    }
    pg.barrier();
}

int main(int argc, char **argv)
{
    ProcessGroupMPI pg(argc, argv);

    test_send_recv(pg);
    test_broadcast(pg);
    test_async_send_recv(pg);

    return 0;
}
