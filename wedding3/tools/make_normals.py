"""Build relief maps for the embossed frame images.

For every bg-*.jpg this writes bg-*-relief.jpg:
  R, G = surface slope (x, y) of the embossing, 128 = flat
  B    = relief mask, 0 on the flat paper and the sage flower texture, 255 on the
         embossed cream frame

The embossing is only visible as light/dark edges, so the image's own luminance
(band-passed to drop the paper grain and the slow lighting falloff) is treated as a
height map, and its gradient gives the slope. Run from wedding2/:

    python tools/make_normals.py
"""
import glob

import numpy as np
from PIL import Image
from scipy import ndimage

FINE, COARSE = 0.9, 10      # px: denoise / remove large-scale shading
MASK_WIN = 9                # px: window for local relief energy

for path in sorted(glob.glob("bg-[0-9]*x[0-9]*.jpg")):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    lum = img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    height = ndimage.gaussian_filter(lum, FINE) - ndimage.gaussian_filter(lum, COARSE)
    gx = ndimage.sobel(height, axis=1)
    gy = ndimage.sobel(height, axis=0)

    # slope scale: the strongest 1% of edges map to the full range
    scale = np.percentile(np.hypot(gx, gy), 99) or 1
    sx = np.clip(gx / scale, -1, 1)
    sy = np.clip(gy / scale, -1, 1)

    # mask: where there is real relief, not paper grain
    energy = np.sqrt(ndimage.uniform_filter(height ** 2, MASK_WIN))
    lo, hi = np.percentile(energy, 45), np.percentile(energy, 85)
    mask = np.clip((energy - lo) / (hi - lo), 0, 1)
    mask = ndimage.gaussian_filter(mask * mask * (3 - 2 * mask), 2)

    # frame only: the cream frame is clearly lighter than the sage flowers once blurred
    # (flowers top out near 0.72, the frame starts around 0.74)
    region = ndimage.gaussian_filter(lum, 5)
    frame = np.clip((region - 0.712) / (0.745 - 0.712), 0, 1)
    mask *= ndimage.gaussian_filter(frame * frame * (3 - 2 * frame), 1.5)

    out = np.stack([sx * 0.5 + 0.5, sy * 0.5 + 0.5, mask], axis=-1)
    name = path.replace(".jpg", "-relief.jpg")
    Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(name, quality=88, subsampling=0)
    print(name, Image.open(name).size)
