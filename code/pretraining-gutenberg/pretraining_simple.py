# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Script for pretraining a small GPT-2 124M parameter model
on books from Project Gutenberg.

Before running this script, make sure you downloaded and
processed the dataset as described in the README.md.
"""

import argparse
import os
from pathlib import Path
import time
import tiktoken
import torch
import wandb

from utils.gpt import create_dataloader_v1, create_dataloader_v2, GPTModel, generate_and_print_sample, calc_loss_batch, evaluate_model, plot_losses


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data


def create_dataloaders(text_data, train_ratio, batch_size, max_length, stride, num_workers=0):
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader_v2(
        text_data[:split_idx],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = create_dataloader_v2(
        text_data[split_idx:],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader


def convert_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def print_eta(start_time, batch_start_time, batch_index, total_batches):
    batch_end_time = time.time()  # End time of processing this batch
    elapsed_time = batch_end_time - batch_start_time
    total_elapsed_time = batch_end_time - start_time
    batches_remaining = total_batches - batch_index
    
    # Calculate average time per batch based on batches processed so far
    average_time_per_batch = total_elapsed_time / batch_index
    eta = average_time_per_batch * batches_remaining

    batch_h, batch_m, batch_s = convert_time(elapsed_time)
    total_h, total_m, total_s = convert_time(total_elapsed_time)
    eta_h, eta_m, eta_s = convert_time(eta)

    print(f"Batch {batch_index}/{total_batches} processed in {batch_h}h {batch_m}m {batch_s}s"
          f"\nTotal time elapsed: {total_h}h {total_m}m {total_s}s"
          f"\nETA for remaining batches: {eta_h}h {eta_m}m {eta_s}s"
          f"\nCompletion: {(batch_index/total_batches)*100:.2f}%")


def train_model_simple(model, optimizer, device, n_epochs,
                       eval_freq, eval_iter, print_sample_iter, start_context,
                       output_dir, save_ckpt_freq, tokenizer, run,
                       batch_size=1024, train_ratio=0.90):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    start_time = time.time()

    try:
        for epoch in range(n_epochs):

            # Iterate over the books in the training corpus
            for index, file_path in enumerate(preprocessed_files, 1):
                book_start_time = time.time()
                text_data = read_text_file(file_path) + " <|endoftext|> "
                print(f"Tokenizing file {index}: {file_path}")

                # Initialize new data loaders for each book
                train_loader, val_loader = create_dataloaders(
                    text_data,
                    train_ratio=train_ratio,
                    batch_size=batch_size,
                    max_length=GPT_CONFIG_124M["context_length"],
                    stride=GPT_CONFIG_124M["context_length"],
                    num_workers=2
                )
                print("Training ...")
                model.train()
                
                total_batches = len(train_loader)
                batch_start_time = time.time()
                
                for batch_idx, (input_batch, target_batch) in enumerate(train_loader, 1):
                    if batch_idx % 100 == 0:
                        print_eta(start_time, batch_start_time, batch_idx, total_batches)
                        batch_start_time = time.time()  # Reset batch start time
                        
                    optimizer.zero_grad()
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    loss.backward()
                    optimizer.step()
                    tokens_seen += input_batch.numel()
                    global_step += 1

                    # Optional evaluation step
                    if global_step % eval_freq == 0:
                        train_loss, val_loss = evaluate_model(
                            model, train_loader, val_loader, device, eval_iter)
                        # log to WandB
                        run.log({"train_loss": train_loss, "val_loss": val_loss})
                        
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)
                        print(f"Ep {epoch+1} (Step {global_step}): "
                              f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                    # Generate text passage
                    if global_step % print_sample_iter == 0:
                        generate_and_print_sample(
                            model, tokenizer, device, start_context
                        )

                if global_step % save_ckpt_freq:
                    file_name = output_dir / f"model_pg_{global_step}.pth"
                    torch.save(model.state_dict(), file_name)
                    print(f"Saved {file_name}")

    except KeyboardInterrupt:
        file_name = output_dir / f"model_pg_{global_step}_interrupted.pth"
        torch.save(model.state_dict(), file_name)
        print(f"Saved {file_name}")

    run.finish()
    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GPT Model Training Configuration')

    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Directory containing the raw data')
    parser.add_argument('--preprocessed_dir', type=str, default='data/preprocessed',
                        help='Directory containing the preprocessed data')
    parser.add_argument('--output_dir', type=str, default='model_checkpoints',
                        help='Directory where the model checkpoints will be saved')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs to train the model')
    parser.add_argument('--print_sample_iter', type=int, default=1000,
                        help='Iterations between printing sample outputs')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency of evaluations during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=100_000,
                        help='Frequency of saving model checkpoints during training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Uses a very small model for debugging purposes')

    args = parser.parse_args()

    if args.debug:
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 10,    # Context length
            "emb_dim": 12,           # Embedding dimension
            "n_heads": 2,            # Number of attention heads
            "n_layers": 2,           # Number of layers
            "drop_rate": 0.0,        # Dropout rate, deactivated via 0.0 as dropout in LLMs is not recommended anymore
            "qkv_bias": False        # Query-key-value bias
        }

    else:
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "emb_dim": 768,          # Embedding dimension
            "n_heads": 12,           # Number of attention heads
            "n_layers": 12,          # Number of layers
            "drop_rate": 0.1,        # Dropout rate
            "qkv_bias": False        # Query-key-value bias
        }

    if torch.cuda.is_available():
        device = "cuda"
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print("Device: ", device)
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    tokenizer = tiktoken.get_encoding("gpt2")

    data_dir = args.data_dir
    all_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(data_dir) for name in files if name.endswith((".txt"))]
    total_files = len(all_files)
    
    preprocessed_dir = args.preprocessed_dir
    preprocessed_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(preprocessed_dir) for name in files if name.endswith((".txt"))]
   

    if total_files == 0:
        print("No training text files found. Make sure you "
              "selected the correct input directory")
        quit()
    print("Total files:", total_files)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="mswaringen",
        # Set the wandb project where this run will be logged.
        project="Pretraining-Gutenberg",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": args.lr,
            "architecture": "GPT2",
            "dataset": "Gutenberg",
            "book_count": total_files,
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "device": device
        },
    )

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, optimizer, device,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        eval_iter=1,
        print_sample_iter=args.print_sample_iter,
        output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        run=run
    )

    epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, output_dir)

    torch.save(model.state_dict(), output_dir / "model_pg_final.pth")
    
