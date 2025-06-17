import os
import torch


def save_checkpoint(epoch, G, D, g_opt, d_opt, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")

    checkpoint = {
        "epoch": epoch + 1,
        "generator_state_dict": G.state_dict(),
        "discriminator_state_dict": D.state_dict(),
        "generator_optimizer_state_dict": g_opt.state_dict(),
        "discriminator_optimizer_state_dict": d_opt.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint, G, D, g_opt, d_opt, device, checkpoint_dir="checkpoints"):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{checkpoint}.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        G.load_state_dict(checkpoint["generator_state_dict"])
        D.load_state_dict(checkpoint["discriminator_state_dict"])
        g_opt.load_state_dict(checkpoint["generator_optimizer_state_dict"])
        d_opt.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])

        return checkpoint["epoch"]
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0
