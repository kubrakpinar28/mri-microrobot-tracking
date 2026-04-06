import tarfile
import nibabel as nib
import numpy as np
import cv2
import os
import tempfile

TAR_PATH = r"C:\Users\kmfm2\Downloads\Task02_Heart.tar"
OUTPUT_FOLDER = r"C:\Users\kmfm2\Downloads\heart_png"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with tarfile.open(TAR_PATH, 'r') as tar:
    members = [m for m in tar.getmembers()
               if 'imagesTr' in m.name and 
               (m.name.endswith('.nii.gz') or m.name.endswith('.nii'))]
    print(f"{len(members)} adet dosya bulundu")
    for member in members:
        f = tar.extractfile(member)
        if f is None:
            continue
        suffix = '.nii.gz' if member.name.endswith('.nii.gz') else '.nii'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        try:
            img = nib.load(tmp_path)
            data = img.get_fdata()
        except Exception as e:
            print(f"  Atlandı: {member.name} ({e})")
            os.unlink(tmp_path)
            continue
        os.unlink(tmp_path)
        base_name = os.path.basename(member.name).replace('.nii.gz', '').replace('.nii', '')
        for i in range(data.shape[2]):
            slice_2d = data[:, :, i]
            mn, mx = slice_2d.min(), slice_2d.max()
            if mx > mn:
                slice_norm = ((slice_2d - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                slice_norm = np.zeros_like(slice_2d, dtype=np.uint8)
            out_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_slice{i:03d}.png")
            cv2.imwrite(out_path, slice_norm)
        print(f"  {base_name} tamamlandi")

print(f"Bitti! PNG klasoru: {OUTPUT_FOLDER}")