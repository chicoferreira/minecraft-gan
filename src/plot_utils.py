import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
import os


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


def show_epoch_evolution(loss_configs, epochs_to_show, test_dl, device, checkpoint_base_dir, create_model_func, load_checkpoint_func, num_samples=2):
    """
    Show how model outputs evolve across different epochs for each loss function
    """
    # Get random samples from the test dataset for consistent comparison

    # Get all available samples from test dataset
    all_samples = []
    for batch in test_dl:
        for i in range(batch["vanilla"].size(0)):
            all_samples.append({"vanilla": batch["vanilla"][i], "shader": batch["shader"][i]})

    # Randomly select samples
    random_indices = random.sample(range(len(all_samples)), min(num_samples, len(all_samples)))
    selected_samples = [all_samples[i] for i in random_indices]

    # Convert to tensors
    x_test = torch.stack([sample["vanilla"] for sample in selected_samples]).to(device)
    y_test = torch.stack([sample["shader"] for sample in selected_samples])

    print(f"Selected random samples at indices: {random_indices}")

    num_configs = len(loss_configs)
    num_epochs = len(epochs_to_show)

    # Create a large figure with subplots
    fig = plt.figure(figsize=(4 * (num_epochs + 2), 4 * num_configs * num_samples))
    gs = GridSpec(num_configs * num_samples, num_epochs + 2, figure=fig)

    # First, create the input and target columns for all samples and loss functions
    for sample_idx in range(num_samples):
        for config_idx, config_name in enumerate(loss_configs.keys()):
            row = sample_idx * num_configs + config_idx

            # Show input (vanilla)
            ax_input = fig.add_subplot(gs[row, 0])
            input_img = (x_test[sample_idx].cpu() * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1)
            ax_input.imshow(input_img)
            ax_input.set_title(f"{config_name}\nSample {sample_idx + 1}\nVanilla Input")
            ax_input.axis("off")

            # Show target (shader)
            ax_target = fig.add_subplot(gs[row, 1])
            target_img = (y_test[sample_idx] * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1)
            ax_target.imshow(target_img)
            ax_target.set_title("Target\n(Shader)")
            ax_target.axis("off")

    # Now load each checkpoint once and generate predictions for all samples
    for config_idx, config_name in enumerate(loss_configs.keys()):
        print(f"Loading evolution for {config_name}...")

        for epoch_idx, epoch in enumerate(epochs_to_show):
            # Create and load model once per epoch
            G_temp, D_temp, g_opt_temp, d_opt_temp = create_model_func(device)
            checkpoint_dir = os.path.join(checkpoint_base_dir, config_name)
            load_checkpoint_func(epoch, G_temp, D_temp, g_opt_temp, d_opt_temp, device, checkpoint_dir)

            # Generate predictions for all samples at once
            G_temp.eval()
            with torch.no_grad():
                y_pred_all = G_temp(x_test)  # Generate for all samples at once

            # Display results for each sample
            for sample_idx in range(num_samples):
                row = sample_idx * num_configs + config_idx
                ax = fig.add_subplot(gs[row, epoch_idx + 2])

                pred_img = (y_pred_all[sample_idx].cpu() * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1)
                ax.imshow(pred_img)
                ax.set_title(f"Epoch {epoch}")
                ax.axis("off")

            # Clean up memory
            del G_temp, D_temp, g_opt_temp, d_opt_temp
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    plt.tight_layout()
    plt.show()
