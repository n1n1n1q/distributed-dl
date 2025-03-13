#include "mpi_wrapper/SerializedTensor.hpp"
#include "distributed.hpp"
// tmp file
int test(int argc, char **argv) {
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    if (world.rank() == 0) {
        torch::Tensor t = torch::randn({2, 2});
        std::cout << "sending\n" << t << "\n------------------------\n";
        SerializedTensor wrapper(t);
        world.send(1, 0, wrapper);
    } else {
        torch::Tensor t = torch::zeros({2, 2,});

        SerializedTensor wrapper;
        world.recv(0, 0, wrapper);

        wrapper.toTensor(t);
        std::cout << "received\n" << t << "\n----------------------------\n" << std::endl;
    }

    return 0;
}

int main(int argc, char **argv) {
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    torch::Tensor t = torch::randn({2, 2});

    if (world.rank() == 0) {
        std::cout << "sending\n" << t << "\n----------------------------\n";
        t *= 2;
        send(world, t, 1);
    } else if (world.rank() == 1) {
        recv(world, t, 0);
        std::cout << "received\n" << t << "\n----------------------------\n";
    }
}
