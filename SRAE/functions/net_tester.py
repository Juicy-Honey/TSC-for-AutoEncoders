import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric
import os
from PIL import Image

def test_set(testLoader, device, model, save_path=None):
    model.eval() 
    total_psnr = 0
    total_ssim = 0
    count = 0

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for step, patches in enumerate(testLoader):
            for p_step, ((hr_tensors, lr_tensors), gs, filenames) in enumerate(patches):
                hr_tensors = hr_tensors.to(device)
                lr_tensors = lr_tensors.to(device)

                scale_factor = 2 ** (model.depth-1)
                _, h, w = lr_tensors.size()
                crop_h = (h // scale_factor) * scale_factor
                crop_w = (w // scale_factor) * scale_factor
                lr_cropped = lr_tensors[:, :crop_h, :crop_w]

                g2= gs[0].to(device)
                if model.scale == 4:
                  g4= gs[1].to(device)
                  
                guides = [g2[:, :crop_h * 2, :crop_w * 2].unsqueeze(0).to(device)]
                if model.scale == 4:
                    guides.append(g4[:, :crop_h * 4, :crop_w * 4].unsqueeze(0).to(device))
                sr_cropped = model(lr_cropped.unsqueeze(0), guides).squeeze(0)

                if crop_h < h or crop_w < w:
                    lr_right_patch = None
                    if crop_w < w:
                        right_patch_w = w - crop_w
                        right_patch_w = scale_factor  
                        lr_right_patch = lr_tensors[:, :crop_h, w-crop_w:]

                    lr_bottom_patch = None
                    if crop_h < h:
                        bottom_patch_h = h - crop_h
                        bottom_patch_h = scale_factor  
                        lr_bottom_patch = lr_tensors[:, h-crop_h:, :crop_w]

                    lr_corner_patch = None
                    if crop_h < h and crop_w < w:
                        lr_corner_patch = lr_tensors[:, h-crop_h:, w-crop_w:]

                    sr_right_patch, sr_bottom_patch, sr_corner_patch = None, None, None

                    if lr_right_patch is not None:
                        guides_right = [g2[:, :crop_h*2, (w-crop_w)*2:].unsqueeze(0).to(device)]
                        if model.scale == 4:
                            guides_right.append(g4[:, :crop_h*4, (w-crop_w)*4:].unsqueeze(0).to(device))
                        sr_right_patch = model(lr_right_patch.unsqueeze(0), guides_right).squeeze(0)

                    if lr_bottom_patch is not None:
                        guides_bottom = [g2[:, (h-crop_h)*2:, :crop_w*2].unsqueeze(0).to(device)]
                        if model.scale == 4:
                            guides_bottom.append(g4[:, (h-crop_h)*4:, :(crop_w)*4].unsqueeze(0).to(device))
                        sr_bottom_patch = model(lr_bottom_patch.unsqueeze(0), guides_bottom).squeeze(0)

                    if lr_corner_patch is not None:
                        guides_corner = [g2[:, (h-crop_h)*2:, (w-crop_w)*2:].unsqueeze(0).to(device)]
                        if model.scale == 4:
                            guides_corner.append(g4[:, (h-crop_h)*4:, (w-crop_w)*4:].unsqueeze(0).to(device))
                        sr_corner_patch = model(lr_corner_patch.unsqueeze(0), guides_corner).squeeze(0)

                    sr_tensors = torch.zeros_like(hr_tensors)
                    sr_tensors[:, :crop_h*model.scale, :crop_w*model.scale] = sr_cropped
                    if sr_right_patch is not None:
                        sr_tensors[:, :crop_h*model.scale, (w - scale_factor)*model.scale:] = sr_right_patch[:, :, (crop_w-scale_factor)*model.scale:]
                    if sr_bottom_patch is not None:
                        sr_tensors[:, (h - scale_factor)*model.scale:, :(crop_w)*model.scale] = sr_bottom_patch[:, (crop_h-scale_factor)*model.scale:, :]
                    if sr_corner_patch is not None:
                        sr_tensors[:, (h - scale_factor)*model.scale:, (w - scale_factor)*model.scale:] = sr_corner_patch[:, (crop_h-scale_factor)*model.scale:, (crop_w-scale_factor)*model.scale:]
                else:
                    sr_tensors = sr_cropped

                hr_y = rgb_to_y_channel(hr_tensors)
                sr_y = rgb_to_y_channel(sr_tensors)

                hr_img = hr_y.cpu().numpy()
                sr_img = sr_y.cpu().numpy()

                psnr = psnr_metric(hr_img, sr_img, data_range=1)
                ssim = ssim_metric(hr_img, sr_img, data_range=1, channel_axis=0)

                total_psnr += psnr
                total_ssim += ssim
                count += 1

                if save_path is not None:
                    sr_image = sr_tensors.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                    sr_image = (sr_image * 255).clip(0, 255).astype('uint8')
                    sr_pil = Image.fromarray(sr_image)
                    sr_pil.save(os.path.join(save_path, f"{filenames}"))

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    print(f'Average PSNR: {avg_psnr:.4f} dB')
    print(f'Average SSIM: {avg_ssim:.4f}')

def rgb_to_y_channel(tensor):
    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Lambda(lambda x: 0.299 * x[0, :, :] + 0.587 * x[1, :, :] + 0.114 * x[2, :, :]),
    ])
    return transform(tensor).unsqueeze(0)

# test_set(testLoader, device, model, save_path='model_weights.pth')
