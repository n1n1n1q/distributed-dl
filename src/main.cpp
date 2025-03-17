#include "mpi_wrapper/SerializedTensor.hpp"
// #include "distributed.hpp"
// tmp file
// int test_mpi_serialize(int argc, char **argv) {
//     boost::mpi::environment env(argc, argv);
//     boost::mpi::communicator world;
//
//     if (world.rank() == 0) {
//         torch::Tensor t = torch::randn({2, 2}, torch::requires_grad(true));
//         t.retain_grad();
//         auto b = ((t + 1) * 2).sigmoid();
//         b.backward();
//         // std::cout << t.retains_grad() << std::endl;
//         // std::cout << "sending\n" << t << "\n------------------------\n";
//         // if (t.retains_grad()) std::cout << "gradient is:\n" << t.grad() << std::endl;
//
//         SerializedTensor wrapper(t);
//         // world.send(1, 0, wrapper);
//     } else {
//         torch::Tensor t = torch::zeros({2, 2,});
//
//         SerializedTensor wrapper;
//         world.recv(0, 0, wrapper);
//
//         wrapper.toTensor(t);
//         std::cout << "received\n" << t << "\n----------------------------\n" << std::endl;
//     }
//
//     return 0;
// }

int main(int argc, char **argv) {
    torch::Tensor a = torch::randn({2, 2}, torch::requires_grad(true));

    auto z = (a + 1) * 2;
    auto b = z.mean();

    // std::cout << b.grad_fn()->name() << std::endl;
    b.backward();
    std::cout << a.grad().numel() << std::endl;


    // test_mpi_serialize(argc, argv);

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
