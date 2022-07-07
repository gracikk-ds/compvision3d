import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import cv2
import click
import subprocess
from tqdm import tqdm
from pathlib import Path
# from segmentation.inference.inference import segmentation

import numpy as np
import quaternion
import json


@click.group()
def main():
    """Entrypoint for scripts"""
    pass


CAMERA_ID = 0


def parse_meta(meta_):
    """
    In-place parser
    :param meta_: metadata dict
    :return: None
    """

    meta_["l"] = float(meta_["camera_distance"])
    meta_["h"] = float(meta_["camera_height"]) - float(meta_["stand_height"])
    meta_["l"] /= meta_["h"]
    meta_["h"] /= meta_["h"]

    meta_['theta_direction'] = int(meta_.get('theta_direction', 1))
    trim_ = meta_.get('trim', [0, 99999999999])
    meta_['trim'] = [int(trim_[0]), int(trim_[1])]


def get_theta(n_fr, lookup=None, meta_=None):
    """
    Main callable to get theta rotation angle from the frame number
    :param n_fr: int number of frame
    :param lookup: a lookup table. if not None the thetas will be sampled from there
    :param meta_: a metadata dict
    :return: theta angle in radians preserving direction sign
    """
    if lookup is not None:
        return lookup[n_fr]

    if n_fr < meta_['trim'][0]:
        return 0
    elif n_fr > meta_['trim'][1]:
        return meta_['theta_direction'] * 2 * np.pi
    else:
        return meta_['theta_direction'] * 2 * np.pi * (n_fr - meta_['trim'][0]) / (meta_['trim'][1] - meta_['trim'][0])


def get_camera_init_qt(meta_):
    """
    Return two initial camera quaternion parameters (R|T)
    :param meta_: parsed metadata
    :return: tuple of (R|T) quaternions
    """
    R_acute = quaternion.from_rotation_vector(
        [-(np.pi / 2 + np.arcsin(meta_["h"] / meta_["l"])), 0, 0]
    )

    T = quaternion.from_vector_part(
        [0, -np.sqrt(meta_["l"] ** 2 - meta_["h"] ** 2), meta_["h"]]
    )
    return R_acute, T


def rotate_by_theta(theta_, camera_position):
    """
    Return new camera position as a tuple of quaternions (R|T)
    :param theta_: scalar angle in radians with preserved sign
    :param camera_position: tuple of position quaternions (R|T)
    :return: new tuple of position quaternions at the angle theta
    """
    theta_rot = quaternion.from_rotation_vector([0, 0, theta_])
    R = theta_rot * camera_position[0]
    T = theta_rot * camera_position[1] * theta_rot.conjugate()
    return R, T


@main.command()
@click.option("--path_to_video", default="video/video_blue.MP4", type=str)
@click.option("--path_to_images_folder", default="images/", type=str)
@click.option("--amount_of_frames", default=150, type=int)
@click.option("--metadata", default="data/raw/meta/meta.json", type=str)
@click.option("--theta_path", default="None")
@click.option("--colmap_text_folder", default="data/processed/colmap_db/colmap_text")
def extract_images_from_video(
        path_to_video: str,
        path_to_images_folder: str,
        amount_of_frames: int,
        metadata: str,
        theta_path: str,
        colmap_text_folder: str,
) -> None:
    """
    Extract predefined number of images from video
    @param path_to_video: path to video file
    @param path_to_images_folder: path to image folder
    @param amount_of_frames: desirable amount of images to extract
    @param metadata: metadata of the video
    @param theta_path: path to json file containing theta angles per frame
    @param colmap_text_folder: folder to save colmap images
    @return: None
    """
    os.makedirs(colmap_text_folder, exist_ok=True)
    if not os.path.exists(metadata):
        raise FileNotFoundError("A meta.json file is required")
    with open(metadata) as j:
        meta = json.load(j)
    parse_meta(meta)

    if os.path.exists(theta_path):
        theta_lookup = {}
        with open(theta_path) as th:
            theta_f = json.load(th)
        for k, v in theta_f.items():
            theta_lookup[int(k)] = float(v)
    else:
        theta_lookup = None

    print("start extract_images_from_video")
    os.makedirs(path_to_images_folder, exist_ok=True)
    path_to_images_folder = Path(path_to_images_folder)
    # Read the video from specified path
    cam = cv2.VideoCapture(path_to_video)

    frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    reducer = (meta['trim'][1] - meta['trim'][0]) // amount_of_frames

    # frame
    frame_number = 0
    frame_to_write_number = 0

    camera_init_pose = get_camera_init_qt(meta)
    with open(os.path.join(colmap_text_folder, "images.txt"), "w") as out:
        pass

    with tqdm(total=meta['trim'][1] - meta['trim'][0]) as pbar:
        while True:
            # reading from frame
            ret, frame = cam.read()
            # frame = cv2.rotate(frame, cv2.ROTATE_180)
            if not ret:
                break
            if not meta['trim'][0] <= frame_number <= meta['trim'][1]:
                frame_number += 1
                continue

            if (frame_number - meta['trim'][0]) % reducer == 0:
                name = path_to_images_folder / f"{frame_to_write_number:03d}.jpg"
                cv2.imwrite(str(name), frame)

                theta = get_theta(frame_number, lookup=theta_lookup, meta_=meta)
                r, t = rotate_by_theta(theta, camera_init_pose)
                r = r.conjugate()  # make it a world to camera transform
                t = -r * t * r.conjugate()
                with open(
                        os.path.join(colmap_text_folder, "images.txt"), "a"
                ) as out:
                    out.write(
                        f"{frame_to_write_number} {r.w} {r.x} {r.y} {r.z} {t.x} "
                        f"{t.y} {t.z} {CAMERA_ID} "
                        f"{frame_to_write_number:03d}.png\n0 0 -1\n"
                    )  # 0 0 -1 is a placeholder
                frame_to_write_number += 1
            frame_number += 1
            pbar.update()


@main.command()
@click.option("--path_to_images_folder", default="images/", type=str)
@click.option("--path_to_cropped_images_folder", default="cropped_images/", type=str)
def crop_resize_images(
        path_to_images_folder: str,
        path_to_cropped_images_folder: str,
) -> None:
    """
    Crop and resize images
    @param path_to_images_folder: path to extracted images
    @param path_to_cropped_images_folder: path to save cropped images
    @return: None
    """
    print("start crop_resize_images")
    os.makedirs(path_to_cropped_images_folder, exist_ok=True)
    path_to_cropped_images_folder = Path(path_to_cropped_images_folder)
    images = [x for x in Path(path_to_images_folder).glob("*.jpg")]

    h, w, _ = cv2.imread(str(images[0])).shape

    for image_path in tqdm(images, total=len(images)):
        image = cv2.imread(str(image_path))

        image = image[250: h - 850, 1390: w - 1390]

        width = 800
        height = 800
        dim = (width, height)

        # resize image
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        cv2.imwrite(
            str(path_to_cropped_images_folder / (image_path.stem + ".png")), image
        )


@main.command()
@click.option("--path_to_cropped_images_folder", default="cropped_images/", type=str)
@click.option("--images_no_background", default="images_no_background/", type=str)
@click.option("--model_type", default="new", type=click.Choice(["new", "old"]))
def remove_background(
        path_to_cropped_images_folder: str, images_no_background: str, model_type: str
) -> None:
    """
    Run background removal net.
    !pip install rembg
    @param path_to_cropped_images_folder: path to images with bg
    @param images_no_background: path to save processed images
    @return: None
    """
    print("start remove_background")
    os.makedirs(images_no_background, exist_ok=True)

    if model_type == "new":
        subprocess.run(
            [
                "rembg",
                "p",
                # "-a",
                # "-ae",
                # "7",
                path_to_cropped_images_folder,
                images_no_background,
            ]
        )
    else:
        segmentation(path_to_cropped_images_folder, images_no_background)


if __name__ == "__main__":
    main()
