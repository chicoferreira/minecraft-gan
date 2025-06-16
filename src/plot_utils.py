import torch
import matplotlib.pyplot as plt


def denormalize(tensor):
    """Denormalize tensor from [-1, 1] to [0, 1]"""
    return (tensor + 1.0) / 2.0


def show_image_pairs(dataloader, num_pairs=4, figsize=(12, 8)):
    """Show pairs of vanilla and shader images from the dataloader"""
    plt.figure(figsize=figsize)

    # Get a batch of data
    batch = next(iter(dataloader))

    vanilla_images = batch["vanilla"]
    shader_images = batch["shader"]

    # Display pairs of images
    for i in range(min(num_pairs, vanilla_images.size(0))):
        # Vanilla image
        plt.subplot(2, num_pairs, i + 1)
        vanilla_img = denormalize(vanilla_images[i]).permute(1, 2, 0)
        plt.imshow(vanilla_img.cpu().numpy())
        plt.title(f"Vanilla {i + 1}")
        plt.axis("off")

        # Shader image
        plt.subplot(2, num_pairs, i + num_pairs + 1)
        shader_img = denormalize(shader_images[i]).permute(1, 2, 0)
        plt.imshow(shader_img.cpu().numpy())
        plt.title(f"Shader {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_training_metrics(train_metrics, test_metrics, figsize=(15, 10)):
    """Plot training and test metrics over epochs"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    epochs = range(1, len(train_metrics) + 1)

    # Generator Loss
    axes[0, 0].plot(epochs, [m["generator_loss"] for m in train_metrics], "b-", label="Train")
    axes[0, 0].plot(epochs, [m["test_generator_loss"] for m in test_metrics], "r-", label="Test")
    axes[0, 0].set_title("Generator Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Discriminator Loss
    axes[0, 1].plot(epochs, [m["discriminator_loss"] for m in train_metrics], "b-", label="Train")
    axes[0, 1].plot(epochs, [m["test_discriminator_loss"] for m in test_metrics], "r-", label="Test")
    axes[0, 1].set_title("Discriminator Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # GAN Loss
    axes[1, 0].plot(epochs, [m["gan_loss"] for m in train_metrics], "b-", label="Train")
    axes[1, 0].plot(epochs, [m["test_gan_loss"] for m in test_metrics], "r-", label="Test")
    axes[1, 0].set_title("GAN Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # L1 Loss
    axes[1, 1].plot(epochs, [m["l1_loss"] for m in train_metrics], "b-", label="Train")
    axes[1, 1].plot(epochs, [m["test_l1_loss"] for m in test_metrics], "r-", label="Test")
    axes[1, 1].set_title("L1 Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def show_test_results(generator, test_dataloader, device, num_samples=4):
    """Show results on test set"""
    print("Results on Test Set:")
    
    # Get random samples from test dataset
    import random
    dataset_size = len(test_dataloader.dataset)
    random_indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
    
    x_samples = []
    y_samples = []
    
    for idx in random_indices:
        sample = test_dataloader.dataset[idx]
        x_samples.append(sample["vanilla"])
        y_samples.append(sample["shader"])
    
    x = torch.stack(x_samples).to(device)
    y = torch.stack(y_samples).to(device)

    with torch.no_grad():
        generator.eval()
        y_hat = generator(x)

    show_results(x, y, y_hat, n=num_samples)


def show_results(x, y, y_hat, n=4):
    """Show comparison of vanilla, generated, and shader images"""
    x = (x + 1) / 2
    y = (y + 1) / 2
    y_hat = (y_hat + 1) / 2

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    for i in range(n):
        axes[i, 0].imshow(x[i].permute(1, 2, 0).cpu())
        axes[i, 0].set_title("Vanilla")
        axes[i, 1].imshow(y_hat[i].permute(1, 2, 0).cpu())
        axes[i, 1].set_title("Generated")
        axes[i, 2].imshow(y[i].permute(1, 2, 0).cpu())
        axes[i, 2].set_title("Shader")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()
