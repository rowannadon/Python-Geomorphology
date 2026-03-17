
from PIL import Image
import numpy as np
import os
import sys
from typing import Optional, Tuple

__all__ = [
    "load_heightmap",
    "ensure_outdir",
    "load_scalar_texture",
    "try_load_npy",
    "save_png_scalar",
    "save_png_rgb",
    "save_png_normal"
]

def load_heightmap(path: str, z_min: float, z_max: float) -> Tuple[np.ndarray, int]:
    """
    Load a grayscale PNG heightmap as float elevations, and detect input bit depth.
    Returns (elev_m, bit_depth), where bit_depth is 8 or 16.
    """
    img = Image.open(path)
    # Don't force-convert yet; inspect mode and dtype
    arr = np.array(img)
    # If someone accidentally passes RGB(A), take the first channel
    if arr.ndim == 3:
        arr = arr[..., 0]

    # Detect 16-bit robustly: any unsigned 2-byte dtype (handles '>u2'/'<u2')
    if arr.dtype.kind == 'u' and arr.dtype.itemsize == 2:
        # Ensure native endianness for consistency
        if arr.dtype.byteorder == '>' or (arr.dtype.byteorder == '=' and sys.byteorder == 'big'):
            arr = arr.byteswap().newbyteorder()
        bit_depth = 16
        maxv = 65535.0
    else:
        # Some PIL versions load 16-bit grayscale as 32-bit 'I'. If so, and
        # the value range fits in 16 bits, treat as 16-bit instead of clipping.
        if (arr.dtype.kind in ('i', 'u') and arr.dtype.itemsize >= 4 and np.max(arr) <= 65535):
            arr = arr.astype(np.uint16, copy=False)
            bit_depth = 16
            maxv = 65535.0
        else:
            # Normalize all other cases to 8-bit grayscale
            if img.mode != 'L':
                img = img.convert('L')
                arr = np.array(img)
            bit_depth = 8
            maxv = 255.0

    elev = z_min + (arr.astype(np.float32) / maxv) * (z_max - z_min)
    return elev, bit_depth

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def load_scalar_texture(path: str, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load a scalar texture from an image file as float32.
    - Accepts 8-bit or 16-bit grayscale; if RGB(A), uses the first channel.
    - Returns raw pixel values as float32 (not normalized to 0..1), so callers
      can decide the interpretation. If you need 0..1, divide by 255 or 65535.
    - If target_shape is provided and differs, resizes using nearest neighbor.
    """
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    # Accept 8/16-bit grayscale robustly; if other, convert to 8-bit L
    if arr.dtype.kind == 'u' and arr.dtype.itemsize == 2:
        # Normalize endianness
        if arr.dtype.byteorder == '>' or (arr.dtype.byteorder == '=' and sys.byteorder == 'big'):
            arr = arr.byteswap().newbyteorder()
        data = arr.astype(np.float32)
    elif arr.dtype == np.uint8:
        data = arr.astype(np.float32)
    else:
        # Convert to 8-bit grayscale then to float
        img = img.convert('L')
        data = np.array(img).astype(np.float32)

    if target_shape is not None and tuple(data.shape) != tuple(target_shape):
        # Resize with pure NumPy nearest-neighbor to preserve value range exactly
        src_h, src_w = data.shape
        dst_h, dst_w = target_shape
        y_idx = np.floor(np.arange(dst_h, dtype=np.float64) * src_h / dst_h).astype(np.int64)
        x_idx = np.floor(np.arange(dst_w, dtype=np.float64) * src_w / dst_w).astype(np.int64)
        y_idx = np.clip(y_idx, 0, src_h - 1)
        x_idx = np.clip(x_idx, 0, src_w - 1)
        data = data[y_idx][:, x_idx].astype(np.float32)

    return data.astype(np.float32)

def try_load_npy(filepath: str, name: str, load_previous: bool) -> Optional[np.ndarray]:
    """Try to load a numpy array from file if load_previous is True and file exists."""
    if not load_previous:
        return None
    
    if os.path.exists(filepath):
        try:
            data = np.load(filepath)
            print(f"    Loaded {name} from {filepath}")
            return data
        except Exception as e:
            print(f"    Warning: Could not load {filepath}: {e}")
            return None
    return None

def save_png_scalar(arr: np.ndarray, path: str, bit_depth: int, clip_lo: float = None, clip_hi: float = None):
    a = np.array(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    if clip_lo is None or clip_hi is None:
        lo, hi = np.percentile(a, [2, 98])
    else:
        lo, hi = clip_lo, clip_hi

    if hi <= lo:
        hi = lo + 1e-6

    a = (a - lo) / (hi - lo)
    a = np.clip(a, 0.0, 1.0)

    if bit_depth == 8:
        im = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), mode='L')
    elif bit_depth == 16:
        im = Image.fromarray((a * 65535.0 + 0.5).astype(np.uint16), mode='I;16')
    else:
        raise ValueError("bit-depth must be 8 or 16")

    im.save(path)

def save_png_normal(normal: np.ndarray, path: str, bit_depth: int):
    n = np.nan_to_num(normal, nan=0.0)
    n = (n * 0.5 + 0.5)
    n = np.clip(n, 0.0, 1.0)
    out8 = (n * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(out8, mode='RGB').save(path)

def save_png_rgb(arr_rgb: np.ndarray, path: str):
    arr_rgb = np.asarray(arr_rgb, dtype=np.uint8)
    Image.fromarray(arr_rgb, mode='RGB').save(path)