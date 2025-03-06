import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def separator():
    print("".join(["=" for _ in range(30)]))

def run(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size, store=dist.TCPStore("127.0.0.1", 29500, world_size, is_master=(rank == 0), use_libuv=False))
    tensor = torch.rand(3)
    print(f"[all_reduce] rank = {rank} before: {tensor}")
    dist.all_reduce(tensor)
    print(f"[all_reduce] rank = {rank} after: {tensor}")

    separator()

    if rank == 0:
        dist.send(torch.tensor([1.48]), dst = 1)
        print(f"[send] master sent magic number")
    elif rank == 1:
        tensor = torch.zeros(1)
        dist.recv(tensor, src = 0)
        print(f"[recieve] rank = {rank} recieved {tensor}")
    else:
        print(f"[recieve] rank = {rank} unable to recieve")

    separator()

    tensor_brd = torch.tensor([3.14808]) if rank == 0 else torch.zeros(1)
    dist.broadcast(tensor_brd, src = 0)
    print(f"[broadcast] rank = {rank} {'sent ' + str(tensor_brd) if rank == 0 else 'got ' + str(tensor_brd)}")

    separator()
    tensor_rdc = torch.rand(1)
    print(f"[reduce] before rank = {rank} {tensor_rdc=}")
    dist.reduce(tensor_rdc, dst=0)
    print(f"[reduce] after rank = {rank} {tensor_rdc=}")
    
    separator()

    tensor_gthr = torch.rand(1)
    print(f"[gather] before {rank = } {tensor_gthr = }")
    gathered = list(torch.zeros(1) for _ in range(world_size)) if rank == 0 else None
    dist.gather(tensor_gthr, gathered)
    print(f"[gather] after {rank = } {gathered}" if rank == 0 else "")

    separator()
    scattered = [torch.rand(1) for _ in range(world_size)] if rank == 0 else None
    print(f"[scatter] before {rank = } { scattered}")
    tensor_sct = torch.zeros(1)
    dist.scatter(tensor_sct, scattered, src = 0)
    print(f"[scatter] after {rank = } {tensor_sct = }")

    separator()

    dist.barrier()
    print(f"[barrier] {rank =} reached the barrier")
    print("Sample operation to check the barrier")

    tensor = torch.rand(3)
    print(f"[all_reduce] rank = {rank} before: {tensor}")
    dist.all_reduce(tensor)
    print(f"[all_reduce] rank = {rank} after: {tensor}")

    dist.destroy_process_group()
if __name__ == '__main__':
    world_size = 4
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)