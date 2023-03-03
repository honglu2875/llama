# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import functools
from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import gradio as gr
import queue
import multiprocessing as mp
import torch.distributed as dist


def setup_model_parallel(local_rank, world_size) -> Tuple[int, int]:
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return


def load(
        ckpt_dir: str,
        tokenizer_path: str,
        local_rank: int,
        world_size: int,
        max_seq_len: int,
        max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def inference(prompt, temperature, top_p, max_len, generator):
    results = generator.generate(
        [prompt], max_gen_len=max_len, temperature=temperature, top_p=top_p
    )
    return results[0]


def server(local_rank, world_size, msg_queue, ret_queue, ckpt_dir, tokenizer_path):
    setup_model_parallel(local_rank, world_size)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    while True:
        if not msg_queue.empty():
            args = msg_queue.get()
            result = inference(*args, generator)
            if local_rank == 0:
                ret_queue.put(result)
        sleep(0.1)


def req_dist(prompt, temperature: float, top_p: float, max_len: int, msg_queue, ret_queue):
    max_len = int(max_len)

    while not msg_queue.full():
        msg_queue.put((prompt, temperature, top_p, max_len))
    while True:
        if not ret_queue.empty():
            result = ret_queue.pop()
            return result
        sleep(0.1)


def main(ckpt_dir: str,
         tokenizer_path: str,
         world_size: int, ):
    # init codes here

    processes = []
    msg_queue = mp.Queue(world_size)
    ret_queue = mp.Queue(1)
    for i in range(world_size):
        processes.append(
            mp.Process(target=server, args=(i, world_size, msg_queue, ret_queue, ckpt_dir, tokenizer_path)))
        processes[-1].start()

    app = gr.Interface(
        functools.partial(req_dist, msg_queue=msg_queue, ret_queue=ret_queue),
        [
            "textbox",
            "number",
            "number",
            "number"
        ],
        "text",
    )

    app.launch(share=True)


if __name__ == '__main__':
    fire.Fire(main)


#######################

def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_seq_len: int = 512,
        max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
    ]
    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
