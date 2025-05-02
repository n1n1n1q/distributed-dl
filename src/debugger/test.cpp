#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);                 // Initialize MPI execution environment:contentReference[oaicite:2]{index=2}
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Get this process’s rank in MPI_COMM_WORLD
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // Get total number of processes
    std::cout << "Hello from rank " << rank
              << " of " << size << std::endl;  // Print a message
    MPI_Finalize();                         // Clean up MPI
    return 0;
}
