#include "mpi_wrapper/SerializedTensor.hpp"
// #include "distributed.hpp"
// tmp file
int test_mpi_serialize(int argc, char **argv) {
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    if (world.rank() == 0) {
        torch::Tensor a = torch::randn({2, 2}, torch::requires_grad(true));

        auto z = (a + 1) * 2;
        auto b = z.mean();

        b.backward();

        std::cout << "Our tensor\n" << a << "\nits gradient\n" << a.grad() << std::endl;

        SerializedTensor wrapper(a);
        world.send(1, 0, wrapper);
    } else {
        torch::Tensor t = torch::zeros({2, 2,});

        SerializedTensor wrapper;
        world.recv(0, 0, wrapper);

        std::cout << "everything is fine there\n";

        wrapper.toTensor(t);
        std::cout << "received\n" << t << "\ntogether with (if it had) gradient\n" << t.grad() << std::endl;
    }

    return 0;
}

int main(int argc, char **argv) {
    test_mpi_serialize(argc, argv);

    // return 0;
    //
    // boost::mpi::environment env(argc, argv);
    // boost::mpi::communicator world;
    //
    // torch::Tensor t = torch::randn({2, 2});
    //
    // if (world.rank() == 0) {
    //     std::cout << "sending\n" << t << "\n----------------------------\n";
    //     t *= 2;
    //     send(world, t, 1);
    // } else if (world.rank() == 1) {
    //     recv(world, t, 0);
    //     std::cout << "received\n" << t << "\n----------------------------\n";
    //     t *= 8;
    //     send(world, t, 2);
    // } else if (world.rank() == 2) {
    //     recv(world, t, 1);
    //     std::cout << "I am changed to\n" << t << "\n";
    // }
}
