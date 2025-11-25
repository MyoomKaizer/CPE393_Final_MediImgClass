import nibabel as nib
import matplotlib.pyplot as plt

path = "iSeg-2017-Training/subject-1-T2.hdr"

img = nib.load(path)
vol = img.get_fdata()

# Handle 4D (H, W, Z, 1)
if vol.ndim == 4:
    vol = vol[..., 0]

print("Shape:", vol.shape)

# scroll through slices
for z in range(0, vol.shape[2], 5):  
    plt.imshow(vol[:, :, z].T, cmap='gray', origin='lower')
    plt.title(f"Slice {z}")
    plt.axis('off')
    plt.show()
