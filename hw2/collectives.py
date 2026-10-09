import torch
import torch.distributed as dist


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]

    # Кольцо: на шаге s отправляем чанк (rank - 1 - s) следующему процессу,
    # принимаем частичную сумму чанка (rank - 2 - s) от предыдущего и прибавляем
    # её к своей копии. Чанк j стартует с процесса j + 1, за world_size - 1 шагов
    # он проходит по кольцу, собирая вклады всех процессов, кроме j,
    # и на последнем шаге приходит процессу j, который прибавляет свой вклад.
    # После этого chunks[rank] содержит сумму по всем процессам.
    for step in range(world_size - 1):
        send_idx = (rank - 1 - step) % world_size
        recv_idx = (rank - 2 - step) % world_size
        send_req = dist.isend(chunks[send_idx], dst=(rank + 1) % world_size)
        recv_buf = torch.empty_like(chunks[recv_idx])
        recv_req = dist.irecv(recv_buf, src=(rank - 1) % world_size)
        send_req.wait()
        recv_req.wait()
        chunks[recv_idx] += recv_buf
    return chunks[rank]


def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)

    # Кольцо: на шаге s отправляем чанк (rank - s) следующему процессу,
    # принимаем чанк (rank - s - 1) от предыдущего. Чанк j стартует
    # с процесса j и за world_size - 1 шагов доходит до процесса j - 1,
    # то есть заполняются все слоты torch.cat(chunks).
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size
        send_req = dist.isend(chunks[send_idx], dst=(rank + 1) % world_size)
        recv_req = dist.irecv(chunks[recv_idx], src=(rank - 1) % world_size)
        send_req.wait()
        recv_req.wait()
    return torch.cat(chunks)


def ring_all_reduce(tensor):
    # Сначала reduce-scatter: каждый процесс получает сумму своего шарда.
    # Затем all-gather: собираем полные суммы ко всем шардам.
    # Возвращаем новую сумму, исходный тензор не меняем.
    return ring_all_gather(ring_reduce_scatter(tensor))
