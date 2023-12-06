import argparse
import glob
from pathlib import Path
from annotator.dwpose import DWposeDetector
import cv2 as cv
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_image", required=True)
args = parser.parse_args()

if os.path.isdir(args.input_image):
    input_images = sorted(glob.glob(f"{args.input_image}/*"))
    dirpath = f"{Path(args.input_image).parent}/pose_img"
    os.makedirs(dirpath, exist_ok=True)
else:
    input_images = [args.input_image]

dwprocessor = DWposeDetector()

for input_image in input_images:
    img = cv.imread(input_image)[:, :, ::-1]
    detected_map = dwprocessor(img)

    path = Path(input_image)
    if os.path.isdir(args.input_image):
        output_path = f"{dirpath}/{path.stem}.png"
    else:
        output_path = f"{path.parent}/{path.stem}_pose.png"
    cv.imwrite(output_path, detected_map[:, :, ::-1])
