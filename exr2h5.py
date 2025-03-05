import OpenEXR
import Imath
import numpy as np
import h5py
import os
from tqdm import tqdm
import argparse

def exr_to_h5(exr_path, h5_path, channel='R'):
#convert exr to h5
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()

    data_window = header['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    channel_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
    depth_array = np.frombuffer(channel_data, dtype=np.float32).reshape(height, width)

    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset(
            name='depth',
            data=depth_array,
            dtype='float32',
            compression='gzip'
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert EXR files to HDF5 format")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing EXR files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save HDF5 files")
    parser.add_argument("--channel", type=str, default="R", help="EXR channel to convert (default: 'R')")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    channel = args.channel


    os.makedirs(output_dir, exist_ok=True)

    #process with tqdm
    for filename in tqdm(os.listdir(input_dir), desc="Processing files"):
        if filename.endswith(".exr"):
            exr_path = os.path.join(input_dir, filename)
            h5_filename = os.path.splitext(filename)[0] + ".h5"
            h5_path = os.path.join(output_dir, h5_filename)
            exr_to_h5(exr_path, h5_path, channel=channel)

    print("Conversion complete.")