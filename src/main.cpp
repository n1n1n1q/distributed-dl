#include "serializedtensor.hpp"
#include "distributed.hpp"
// tmp file
int test(int argc, char **argv) {
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    if (world.rank() == 0) {
        torch::Tensor a = torch::randn({2, 2}, torch::requires_grad(true));

        auto z = (a + 1) * 2;
        auto b = z.mean();

        b.backward();

        std::cout << "Our tensor\n" << a << "\nits gradient\n" << a.grad() << std::endl;

        SerializedTensorCPU_impl wrapper(a);
        world.send(1, 0, wrapper);
    } else {
        torch::Tensor t = torch::zeros({2, 2,});

        SerializedTensorCPU_impl wrapper;
        world.recv(0, 0, wrapper);

        std::cout << "everything is fine there\n";

        wrapper.toTensor(t);
        std::cout << "received\n" << t << "\ntogether with (if it had) gradient\n" << t.grad() << std::endl;
    }

    return 0;
}

int main(int argc, char **argv) {
    // test_mpi_serialize(argc, argv);


    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    torch::Tensor t = torch::randn({2, 2}, torch::requires_grad(true));

    if (world.rank() == 0) {
        auto out = (t * 2).mean();
        out.backward();
        std::cout << "sending\n" << t << "\nmy grad\n" << t.grad() << std::endl;

        send(world, t, 1);
    } else if (world.rank() == 1) {
        recv(world, t, 0);
        std::cout << "received\n" << t << "\ntogether with (if it had) grad\n" << t.grad() << std::endl;
        t *= 8;
        send(world, t, 2);
    } else if (world.rank() == 2) {
        recv(world, t, 1);
        std::cout << "I am changed to\n" << t << "\n";
    }
}
