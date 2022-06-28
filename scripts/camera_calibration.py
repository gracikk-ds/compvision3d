import os
import cv2
import click
import numpy as np
from pathlib import Path

# termination criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def save_coefficients(width, height, mtx, dist, path):
    """Save the camera matrix and the distortion coefficients to given path/file."""
    list_of_lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, fl_x, fl_y, k1, k2, p1, p2, cx, cy",
    ]
    os.makedirs(path, exist_ok=True)
    fl_x, fl_y, cx, cy = mtx[0][0], mtx[1][1], mtx[0][2], mtx[1][2]
    k1, k2, p1, p2, k3 = dist

    main_line = " ".join([
        "1", "OPENCV", str(width), str(height),
        str(fl_x), str(fl_y), str(cx), str(cy),
        str(k1), str(k2), str(p1), str(p2)
    ])

    list_of_lines.append(main_line)

    with open(str(Path(path) / "cameras.txt"), "w") as text_file:
        for line in list_of_lines:
            text_file.write(line + "\n")


@click.command()
@click.option("--image_dir", type=str, required=True, help="image directory path")
@click.option("--image_format", type=str, required=True, help="image format, png/jpg")
@click.option("--square_size", type=str, required=True, help="chessboard square size")
@click.option(
    "--width",
    type=str,
    required=True,
    default=9,
    help="Number of intersection points of squares in the long side",
)
@click.option(
    "--height",
    type=str,
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
def calibrate(image_dir, image_format, square_size, width, height, output_path):
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
            height, width = img.shape
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    ret, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    save_coefficients(width, height, mtx, dist, output_path)
    print("Calibration is finished. RMS: ", ret)


if __name__ == "__main__":
    calibrate()
