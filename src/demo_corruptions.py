# Demo: compress a clean and a corrupted Kodak image with a learned codec, JPEG and JPEG2000.
import io

import numpy as np
import torch
import torch.nn.functional as F
from compressai.zoo import bmshj2018_hyperprior
from imagecorruptions import corrupt
from PIL import Image

from seed import set_seed

IMAGE_PATH = "data/kodak/kodim01.png"
CORRUPTIONS = ["gaussian_noise", "defocus_blur", "elastic_transform"]
SEVERITIES = [1, 3, 5]


def psnr(original, decoded):
    mse = np.mean((original.astype(float) - decoded.astype(float)) ** 2)
    return 10 * np.log10(255**2 / mse)


def compress_learned(model, img):
    """Returns (decoded image, bits per pixel)."""
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255
    x = x.unsqueeze(0)
    h, w = x.shape[2], x.shape[3]

    # the model needs height and width to be multiples of 64
    x = F.pad(x, (0, (-w) % 64, 0, (-h) % 64), mode="replicate")

    with torch.no_grad():
        out = model.compress(x)
        x_hat = model.decompress(out["strings"], out["shape"])["x_hat"]

    x_hat = x_hat.clamp(0, 1)[:, :, :h, :w]
    decoded = (x_hat[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

    n_bytes = sum(len(s) for strings in out["strings"] for s in strings)
    return decoded, n_bytes * 8 / (h * w)


def compress_pil(img, file_format, **options):
    """JPEG / JPEG2000 through Pillow. Returns (decoded image, bits per pixel)."""
    buffer = io.BytesIO()
    Image.fromarray(img).save(buffer, format=file_format, **options)
    n_bytes = buffer.tell()
    buffer.seek(0)
    decoded = np.array(Image.open(buffer).convert("RGB"))
    return decoded, n_bytes * 8 / (img.shape[0] * img.shape[1])


clean = np.array(Image.open(IMAGE_PATH).convert("RGB"))

model = bmshj2018_hyperprior(quality=3, pretrained=True).eval()
model.update(force=True)  # needed once before compress()

# every (corruption, severity) pair, plus the clean image as the reference
conditions = [("clean", 0)]
for name in CORRUPTIONS:
    for severity in SEVERITIES:
        conditions.append((name, severity))

clean_bpp = {}  # bpp of each codec on the clean image, to measure the rate spike

print("corruption         sev  codec        bpp    psnr  rate_ratio")
for name, severity in conditions:
    if name == "clean":
        img = clean
    else:
        set_seed(42)  # same noise every run
        img = corrupt(clean, corruption_name=name, severity=severity)
        img = np.uint8(np.clip(img, 0, 255))

    results = {
        "learned": compress_learned(model, img),
        "jpeg": compress_pil(img, "JPEG", quality=50),
        # fixed 24:1 ratio = 1 bpp, so this codec's bpp never changes
        "jpeg2000": compress_pil(
            img, "JPEG2000", quality_mode="rates", quality_layers=[24], irreversible=True
        ),
    }

    for codec, (decoded, bpp) in results.items():
        if name == "clean":
            clean_bpp[codec] = bpp
        ratio = bpp / clean_bpp[codec]  # above 1 means a rate spike
        print(f"{name:18} {severity:<4} {codec:9} {bpp:6.3f} {psnr(img, decoded):7.2f} {ratio:8.2f}")
