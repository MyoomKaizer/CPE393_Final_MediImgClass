"""
view_hdr.py

Simple utility to view 3D medical imaging volumes (.hdr/.img format).
Display scrolls through slices.
"""

import nibabel as nib
import matplotlib.pyplot as plt

def view_hdr_volume(path, step=5):
    """
    Load and display a HDR volume.
    
    Args:
        path: Path to .hdr file
        step: Display every Nth slice (default 5)
    """
    img = nib.load(path)
    vol = img.get_fdata()

    # Handle 4D (H, W, Z, 1)
    if vol.ndim == 4:
        vol = vol[..., 0]

    print("Shape:", vol.shape)

    # scroll through slices
    for z in range(0, vol.shape[2], step):  
        plt.imshow(vol[:, :, z].T, cmap='gray', origin='lower')
        plt.title(f"Slice {z}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    # Example usage
    path = "iSeg-2017-Training/subject-1-T2.hdr"
    view_hdr_volume(path)
