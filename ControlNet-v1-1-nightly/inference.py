import argparse
import glob
from pathlib import Path
from annotator.dwpose import DWposeDetector
import cv2 as cv
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_image", nargs="+", required=True)
args = parser.parse_args()

dwprocessor = DWposeDetector()

for path in args.input_image:
    print(f"Processing {path}")
    
    img = cv.imread(path)[:, :, ::-1]
    detected_map = dwprocessor(img)

    path = Path(path)

    os.makedirs(f"{path.parent.parent}/pose_image", exist_ok=True)
    output_path = f"{path.parent.parent}/pose_image/{path.name}"
    cv.imwrite(output_path, detected_map[:, :, ::-1])
