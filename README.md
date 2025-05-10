# distributed-dl
Reimplementation of torch distributed model on C++.
## Installation 
Requirements:
* libtorch
* libboost (with MPI support)

If compiling with CUDA support, cuda development toolkit must be installed

```bash
git clone https://github.com/n1n1n1q/distributed-dl/
cd distributed-dl
mkdir build
cd build
cmake .. # path cuda architecture flag if compiling with CUDA support
make # or make install
```

## Overview
Torch distributed is a very powerful tool for distributed deep learning. Although python interface is well done and is easy to use, it's hard to say the same about libtorch.  

We offer the following functionality implementation for MPI and NCCL backends.
| Operation   | MPI      | NCCL     |
|------------|---------|---------|
| send       | ✅       | ✅       |
| recv       | ✅       | ✅       |
| broadcast  | ✅       | ✅       |
| all_reduce | ✅       | ✅       |
| reduce     | ✅       | ✅       |
| scatter    | ✅       | ✅       |
| gather     | ✅       | ✅       |
| barrier    | ✅       | ✅       |
| isend      | ✅       | ❌       |
| irecv      | ✅       | ❌       |
| Debugging  | ✅ | ❌ |
| High-level interface| ✅ | ✅ |

## License
![MIT License](./LICENSE)
