import torch
import matplotlib.pyplot as plt
import random


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


def show_test_results(generator, test_dataloader, device, num_samples=4):
    """Show results on test set"""
    print("Results on Test Set:")

    # Get random samples from test dataset
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


def show_combined_results(models_dict, test_dataloader, device, num_samples=3):
    """Show results from all models in a combined visualization"""
    print("Comparing results from all loss configurations:")
    
    # Get test samples
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
    
    # Generate results for all models
    model_results = {}
    for config_name, model in models_dict.items():
        with torch.no_grad():
            model.eval()
            y_hat = model(x)
            model_results[config_name] = y_hat.cpu()
    
    # Create combined visualization
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    
    # Denormalize all images
    x_display = (x_cpu + 1) / 2
    y_display = (y_cpu + 1) / 2
    
    num_models = len(models_dict)
    num_cols = num_models + 2  # vanilla + all models + target
    
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(4 * num_cols, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Show vanilla (input)
        axes[i, 0].imshow(x_display[i].permute(1, 2, 0))
        axes[i, 0].set_title("Vanilla (Input)")
        axes[i, 0].axis("off")
        
        # Show generated results for each model
        for j, (config_name, result) in enumerate(model_results.items()):
            result_display = (result[i] + 1) / 2
            axes[i, j + 1].imshow(result_display.permute(1, 2, 0))
            axes[i, j + 1].set_title(f"Generated ({config_name})")
            axes[i, j + 1].axis("off")
        
        # Show target (shader)
        axes[i, -1].imshow(y_display[i].permute(1, 2, 0))
        axes[i, -1].set_title("Target (Shader)")
        axes[i, -1].axis("off")
    
    plt.tight_layout()
    plt.show()
