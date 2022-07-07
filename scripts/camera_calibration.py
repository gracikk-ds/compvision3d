import os
import cv2
import click
import numpy as np
from tqdm import tqdm
from typing import List
from pathlib import Path

# start point for images extraction
START_POINT = 60
# termination criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


@click.group()
def main():
    """Entrypoint for scripts"""
    pass


def save_coefficients(width: int, height: int, mtx: List[List], dist: List, path: str):
    """
    Save the camera matrix and the distortion coefficients to given path/file.
    :param width: image width
    :param height: image height
    :param mtx: camera matrix
    :param dist: distortion coefficients
    :param path:path to save camera.txt file
    :return: None
    """

    list_of_lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, fl_x, fl_y, cx, cy, k1, k2, p1, p2",
    ]
    os.makedirs(path, exist_ok=True)
    fl_x, fl_y, cx, cy = mtx[0][0], mtx[1][1], mtx[0][2], mtx[1][2]
    k1, k2, p1, p2, k3 = dist

    # we are doing crop procedure
    # so we have to adjust principal point parameters respectively
    cx_new = cx - 1390
    cy_new = cy - 250

    # then we are doing resize procedures
    # so we have to adjust such parameters as fl_x, fl_y, cx, cy
    fl_x_new = fl_x * (800 / 1060)
    fl_y_new = fl_y * (800 / 1060)
    cx_new = cx_new * (800 / 1060)
    cy_new = cy_new * (800 / 1060)

    main_line = " ".join([
        "1", "OPENCV", str(800), str(800),
        str(fl_x_new), str(fl_y_new), str(cx_new), str(cy_new),
        str(k1), str(k2), str(p1), str(p2)
    ])

    list_of_lines.append(main_line)

    with open(str(Path(path) / "cameras.txt"), "w") as text_file:
        for line in list_of_lines:
            text_file.write(line + "\n")


@main.command()
@click.option("--video_dir", type=str, required=True, help="video directory path")
@click.option(
    "--amount_of_frames",
    type=int,
    required=True,
    help="amount of frames to extract"
)
@click.option(
    "--path_to_images_folder",
    type=str,
    required=True,
    help="path to save frames"
)
def extract_calibration_images(
        video_dir: str,
        amount_of_frames: int,
        path_to_images_folder: str
):
    """
    Extract frames from video
    :param path_to_video: path to video file
    :param amount_of_frames: amount of frames to extract
    :param path_to_images_folder: path to save frames
    :return: None
    """
    os.makedirs(path_to_images_folder, exist_ok=True)
    # Read the video from specified path
    cam = cv2.VideoCapture(video_dir)

    frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    reducer = (frame_count - START_POINT) // amount_of_frames

    # frame
    frame_number = 0
    frame_to_write_number = 0

    with tqdm(total=frame_count - START_POINT) as pbar:
        while True:
            # reading from frame
            ret, frame = cam.read()

            if not ret:
                break
            if not START_POINT <= frame_number:
                frame_number += 1
                continue
            if (frame_number - START_POINT) % reducer == 0:

                name = Path(path_to_images_folder) / f"{frame_to_write_number:03d}.jpg"
                cv2.imwrite(str(name), frame)

                frame_to_write_number += 1
            frame_number += 1
            pbar.update()


@main.command()
@click.option("--image_dir", type=str, required=True, help="image directory path")
@click.option("--image_format", type=str, required=True, help="image format, png/jpg")
@click.option("--square_size", type=float, required=True, help="chessboard square size")
@click.option(
    "--width",
    type=int,
    required=True,
    default=9,
    help="Number of intersection points of squares in the long side",
)
@click.option(
    "--height",
    type=int,
    required=True,
    default=6,
    help="Number of intersection points of squares in the short side",
)
@click.option(
    "--output_path",
    type=str,
    required=True,
    default="data/processed/colmap_db/colmap_text",
    help="YML file to save calibration matrices",
)
def calibrate(
        image_dir,
        image_format,
        square_size,
        width,
        height,
        output_path
):
    """Apply camera calibration operation for images in the given directory path."""
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = [x.as_posix() for x in Path(image_dir).glob("*." + image_format)]

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            height, width = img.shape[:2]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print("ret:", ret)
    print("mtx:", mtx)
    print("dist:", dist)
    print("rvecs:", rvecs)
    print("tvecs:", tvecs)
    save_coefficients(width, height, mtx, dist[0], output_path)
    print("Calibration is finished. RMS: ", ret)


if __name__ == "__main__":
    main()
