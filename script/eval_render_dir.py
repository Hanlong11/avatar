from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import lpips
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def crop_image(gt_mask: np.ndarray, patch_size: int, *images: np.ndarray):
    mask_uv = np.argwhere(gt_mask > 0.0)
    min_v, min_u = mask_uv.min(0)
    max_v, max_u = mask_uv.max(0)

    pad_size = patch_size // 2
    min_v = (min_v - pad_size).clip(0, gt_mask.shape[0])
    min_u = (min_u - pad_size).clip(0, gt_mask.shape[1])
    max_v = (max_v + pad_size).clip(0, gt_mask.shape[0])
    max_u = (max_u + pad_size).clip(0, gt_mask.shape[1])

    cropped = []
    for image in images:
        cropped.append(image[min_v:max_v, min_u:max_u])
    return cropped


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0


def main():
    data_dir = Path("/home/hanlong/project/mmlphuman/render/subject02_trainview")
    result_dir = data_dir / "result"
    gt_dir = data_dir / "gt"
    mask_dir = data_dir / "mask"
    out_path = data_dir / "metrics.json"

    filenames = sorted(p.name for p in gt_dir.glob("*.png"))
    if not filenames:
        raise FileNotFoundError(f"No PNG files found in {gt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr_model = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_model = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    psnr_list = []
    ssim_list = []
    lpips_list = []

    with torch.no_grad():
        for idx, name in enumerate(filenames, start=1):
            im_gt = iio.imread(gt_dir / name)
            im_result = iio.imread(result_dir / name)
            mask = iio.imread(mask_dir / name)

            mask = mask > 128
            im_gt_crop, im_result_crop = crop_image(mask, 512, im_gt, im_result)

            gt_tensor = to_tensor(im_gt, device)
            result_tensor = to_tensor(im_result, device)
            gt_crop_tensor = to_tensor(im_gt_crop, device) * 2 - 1
            result_crop_tensor = to_tensor(im_result_crop, device) * 2 - 1

            psnr_list.append(psnr_model(result_tensor, gt_tensor).item())
            ssim_list.append(ssim_model(result_tensor, gt_tensor).item())
            lpips_list.append(lpips_model(result_crop_tensor, gt_crop_tensor).item())

            if idx % 50 == 0 or idx == len(filenames):
                print(f"{idx}/{len(filenames)}")

    metrics = {
        "num_frames": len(filenames),
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "lpips": float(np.mean(lpips_list)),
    }

    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
