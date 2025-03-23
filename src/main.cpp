#include "serializedtensor.hpp"
#include "distributed.hpp"
// tmp file
int test(int argc, char **argv) {
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    torch::Tensor a = torch::randn({2, 2}, torch::requires_grad(true));
    if (world.rank() == 0) {
        auto z = (a + 1) * 2;
        auto b = z.mean();

        b.backward();

        std::cout << "Our tensor\n" << a << "\nits gradient\n" << a.grad() << std::endl;

        SerializedTensorCPU_impl wrapper(a.mutable_grad());
        world.send(1, 0, wrapper);
    } else {
        SerializedTensorCPU_impl wrapper;
        world.recv(0, 0, wrapper);


        wrapper.toTensor(a.mutable_grad());
        std::cout << "received\n" << a << "\ntogether with (if it had) gradient\n" << a.grad() << std::endl;
    }

    return 0;
}

int main(int argc, char **argv) {
    // test(argc, argv);


    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    torch::Tensor t = torch::randn({2, 2}, torch::requires_grad(true));


    if (world.rank() == 0) {
        auto out = (t * 2).mean();
        out.backward();

        std::cout << "sending\n" << t << "\nmy grad\n" << t.grad() << std::endl;
        send(world, t.mutable_grad(), 1);
    } else if (world.rank() == 1) {
        recv(world, t.mutable_grad(), 0);

        std::cout << "received\n" << t << "\ntogether with (if it had) grad\n" << t.grad() << std::endl;

        t = t * 8;

        std::cout << "sosi\n" << std::endl;
        send(world, t.mutable_grad(), 2);
    } else if (world.rank() == 2) {
        recv(world, t.mutable_grad(), 1);
        std::cout << "I am changed to\n" << t << "\n";
    }
}
