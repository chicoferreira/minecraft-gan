import torch
import numpy as np
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def denormalize_to_01(tensor):
    return (tensor + 1.0) / 2.0


class ImageQualityMetrics:
    def __init__(self, device):
        self.device = device
        self.psnr_metric = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device)
        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    def evaluate_batch(self, real_images, fake_images):
        real_denorm = denormalize_to_01(real_images)
        fake_denorm = denormalize_to_01(fake_images)

        # Compute metrics
        psnr_value = self.psnr_metric(fake_denorm, real_denorm)
        ssim_value = self.ssim_metric(fake_denorm, real_denorm)
        lpips_value = self.lpips_metric(fake_denorm, real_denorm)

        return {"psnr": psnr_value.item(), "ssim": ssim_value.item(), "lpips": lpips_value.item()}

    def update_fid(self, real_images, fake_images):
        real_denorm = denormalize_to_01(real_images)
        fake_denorm = denormalize_to_01(fake_images)

        self.fid_metric.update(real_denorm, real=True)
        self.fid_metric.update(fake_denorm, real=False)

    def compute_fid(self):
        return self.fid_metric.compute().item()

    def reset_fid(self):
        self.fid_metric.reset()


def evaluate_generator_metrics(generator, dataloader, device, max_batches=None):
    generator.eval()
    metrics_calculator = ImageQualityMetrics(device)

    psnr_values = []
    ssim_values = []
    lpips_values = []

    metrics_calculator.reset_fid()

    print("Computing image quality metrics...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating metrics")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            vanilla_images = batch["vanilla"].to(device)
            shader_images = batch["shader"].to(device)

            # Generate fake images
            generated_images = generator(vanilla_images)

            # Compute batch metrics
            batch_metrics = metrics_calculator.evaluate_batch(shader_images, generated_images)

            psnr_values.append(batch_metrics["psnr"])
            ssim_values.append(batch_metrics["ssim"])
            lpips_values.append(batch_metrics["lpips"])

            # Update FID
            metrics_calculator.update_fid(shader_images, generated_images)

    # Compute final metrics
    results = {
        "psnr_mean": np.mean(psnr_values),
        "psnr_std": np.std(psnr_values),
        "ssim_mean": np.mean(ssim_values),
        "ssim_std": np.std(ssim_values),
        "lpips_mean": np.mean(lpips_values),
        "lpips_std": np.std(lpips_values),
        "fid": metrics_calculator.compute_fid(),
    }

    return results


def print_metrics_summary(metrics):
    print(f"PSNR (higher is better):  {metrics['psnr_mean']:.4f} ± {metrics['psnr_std']:.4f} dB")
    print(f"SSIM (higher is better):  {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
    print(f"FID (lower is better):    {metrics['fid']:.4f}")
    print(f"LPIPS (lower is better):  {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}")
