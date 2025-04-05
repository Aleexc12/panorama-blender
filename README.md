# MULTIBAND & POISSON PANORAMA STITCHING



![Panorama Result](panorama_poisson_final.jpg)

This project builds high-quality image panoramas using cylindrical warping, SIFT-based alignment, and two advanced blending techniques: **GraphCut + MultiBand** and **GraphCut + Poisson** blending.

## FEATURES

- Cylindrical warping based on focal length
- Automatic alignment using SIFT and FLANN
- GraphCut for optimal seam selection
- MultiBand blending (OpenCV) for seamless transitions
- Poisson blending (Numba-accelerated) for photometric consistency
- Command-line interface for flexible usage
- Intermediate visualizations saved at each step

## INSTALLATION

### 1. Clone the repository
```bash
git clone https://github.com/tu_usuario/panorama-blending.git
cd panorama-blending
```

### 2. (Optional) Create a virtual environment
```bash
python -m venv venv
venv\Scripts\Activate.ps1  # On Windows powershell
source venv/bin/activate   # On macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## USAGE

### Prepare your input
1. Place your images in a folder (default: `pano_images/`)
2. Create an `img_list.txt` file inside that folder with the following format:

```
1700.0
image1.jpg
image2.jpg
image3.jpg
...
```

(The first line is the focal length, and the rest are image filenames in stitching order)

### Run the stitching pipeline

You can simply run the script with default parameters:
```bash
python pano.py
```

Or customize them:
```bash
python pano.py --input_dir ./your_folder --processing_dir ./output_folder --focal 1500 --blender p
```

#### Parameters:
- `--input_dir`: Folder with input images and `img_list.txt` (default: `./pano_images`)
- `--processing_dir`: Folder for storing intermediate outputs (default: `./temp`)
- `--focal`: Focal length for cylindrical projection (default: `1700.0`)
- `--blender`: Blending method: `mb` (MultiBand) or `p` (Poisson) (default: `mb`)

### Output
- `panorama_multiband_final.jpg` or `panorama_poisson_final.jpg`
- Intermediate outputs in `temp/`

## PROJECT STRUCTURE

```
/panorama-blending
│── pano.py                         # Main stitching pipeline
│── pano_images/                   # Input images + img_list.txt (default input)
│── temp/                          # Intermediate processed images (default output)
│── panorama_multiband_final.jpg  # Final output (if using multiband)
│── panorama_poisson_final.jpg    # Final output (if using poisson)
│── README.md                      # Documentation
```

## TODO / FUTURE IMPROVEMENTS

- Interactive visualization of stitching steps
- Auto focal-length estimation
- Horizontal/vertical blending improvements

## LICENSE

This project is licensed under the MIT License.