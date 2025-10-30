from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from process import Process

# Scale factor for resizing input images
scale_factor = 0.5

# IMAGE PRE-PROCESSING
def pre_process(image_path, scale_factor=None):
    # Load, convert, and optionally resize an image before processing.
    print(f"Processing {image_path}...")
    im = Process(image_path)
    original_width, original_height = im.image.size

    if scale_factor is not None:
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        im.resize_image(new_width, new_height)
    else:
        new_width, new_height = original_width, original_height

    array = im.get_array()
    print(f"Pre-processing for {image_path} complete.")
    im.get_info()
    return array, new_width, new_height

# Read camera parameters (focal length, baseline, disparity offset) from calibration file.
def parse_calibration_file(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    calib_data = {}
    for line in lines:
        if line.startswith("cam0"):
            cam0_line = line.strip().split('=')[1]
            cam0_matrix = eval(cam0_line.replace(';', '],['))
            focal_length = cam0_matrix[0][0]
            calib_data['focal_length'] = focal_length
        elif line.startswith("doffs"):
            calib_data['doffs'] = float(line.split('=')[1])
        elif line.startswith("baseline"):
            calib_data['baseline'] = float(line.split('=')[1])
    
    return calib_data

# DISPARITY COMPUTATION
def calc_disparity(array_left, array_right, window_size, search_range):
    start_time = time.time()
    height, width = array_left.shape

    def compute_row_disparity(row):
        disps = []
        for col1 in range(width - window_size):
            win1 = array_left[row:row + window_size, col1:col1 + window_size].flatten()
            init = max(0, col1 - search_range)
            sads = [np.sum(np.abs(win1 - array_right[row:row + window_size, col2:col2 + window_size].flatten()))
                    for col2 in range(col1, init - 1, -1)]
            disps.append(np.argmin(sads))
        return disps

    disp_matrix = np.array(
        Parallel(n_jobs=-1)(
            delayed(compute_row_disparity)(row) for row in tqdm(range(height - window_size), desc="Computing Disparity")
        )
    )

    print(f"Disparity calculations complete. Time elapsed: {time.time() - start_time:.2f}s")
    return disp_matrix

# OUTLIER REMOVAL
def remove_outliers(disp_matrix, window_size, threshold):
    print("Removing outliers from disparity map...")
    median_disp = ndimage.median_filter(disp_matrix, size=window_size)
    diff = np.abs(disp_matrix - median_disp)
    outliers = diff > threshold
    num_outliers = np.sum(outliers)
    print(f"Detected {num_outliers} outlier pixels.")
    cleaned_disp = np.copy(disp_matrix)
    cleaned_disp[outliers] = median_disp[outliers]
    print("Outlier removal complete.")
    return cleaned_disp

# POST-PROCESSING
def post_process(disp_matrix):
    print(f"Post-processing disparity matrix of shape {disp_matrix.shape}...")
    disp_matrix = ndimage.median_filter(disp_matrix, size=5)
    disp_matrix = ndimage.gaussian_filter(disp_matrix, sigma=1)
    print("Post-processing complete.")
    return disp_matrix

# DEPTH CONVERSION
def disparity_to_depth(disp_matrix, focal_length, baseline, doffs):
    print("Converting disparity map to depth map...")
    with np.errstate(divide='ignore'):
        depth_map = (baseline * focal_length) / (disp_matrix/ scale_factor + doffs)
    depth_map[~np.isfinite(depth_map)] = 0
    print("Depth map generation complete.")
    return depth_map

def main():
    image_path_left = './data/set1/im0.png'
    image_path_right = './data/set1/im1.png'
    calib_path = './data/set1/calib.txt'
    output_dir = './data/output/'

    print("\n=== Pre-Processing Images ===")
    array_left, width, height = pre_process(image_path_left, scale_factor)
    array_right, _, _ = pre_process(image_path_right, scale_factor)

    window_size = max(5, int(90 * scale_factor))
    search_range = max(10, int(300 * scale_factor))

    print("\n=== Computing Disparity Map ===")
    disp_matrix = calc_disparity(array_left, array_right, window_size, search_range)

    print("\n=== Removing Outliers ===")
    disp_matrix = remove_outliers(disp_matrix, window_size=int(150*scale_factor), threshold=int(50*scale_factor))

    print("\n=== Post-Processing Disparity Map ===")
    disp_matrix = post_process(disp_matrix)

    print("\n=== Parsing Calibration ===")
    calib = parse_calibration_file(calib_path)

    print("\n=== Generating Depth Map ===")
    depth_map = disparity_to_depth(disp_matrix, calib['focal_length'], calib['baseline'], calib['doffs'])

    print("\n=== Saving Images ===")
    plt.imsave(f"{output_dir}/disparity_{width}x{height}.png", disp_matrix, cmap='gray')
    plt.imsave(f"{output_dir}/depth_map_{width}x{height}.png", depth_map, cmap='inferno')

    # Save colorized depth map for visualization
    plt.figure(figsize=(10, 6))
    im = plt.imshow(depth_map, cmap='inferno')
    plt.colorbar(im, label='Depth (mm)')
    plt.title('Depth Map with Color Scale')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/depth_map_colored_{width}x{height}.png")
    plt.close()

    print("\n=== Processing Complete! ===")

if __name__ == "__main__":
    main()
