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
from segmentation.inference.inference import segmentation


@click.group()
def main():
    """Entrypoint for scripts"""
    pass


@main.command()
@click.option("--path_to_video", default="video/video_blue.MP4", type=str)
@click.option("--path_to_images_folder", default="images/", type=str)
@click.option("--amount_of_frames", default=150, type=int)
def extract_images_from_video(
    path_to_video: str,
    path_to_images_folder: str,
    amount_of_frames: int,
) -> None:
    """
    Extract predefined number of images from video
    @param path_to_video: path to video file
    @param path_to_images_folder: path to image folder
    @param amount_of_frames: desirable amount of images to extract
    @return: None
    """
    print("start extract_images_from_video")
    os.makedirs(path_to_images_folder, exist_ok=True)
    path_to_images_folder = Path(path_to_images_folder)
    # Read the video from specified path
    cam = cv2.VideoCapture(path_to_video)

    frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    reducer = frame_count // amount_of_frames

    # frame
    frame_number = 0
    frame_to_write_number = 0

    with tqdm(total=frame_count) as pbar:
        while True:
            # reading from frame
            ret, frame = cam.read()
            if ret:
                if frame_number % reducer == 0:
                    name = path_to_images_folder / f"{frame_to_write_number:03d}.jpg"
                    cv2.imwrite(str(name), frame)
                    frame_to_write_number += 1
                frame_number += 1
                pbar.update()
            else:
                break


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
    delta = (w - h) // 2

    for image_path in tqdm(images, total=len(images)):
        image = cv2.imread(str(image_path))

        image = image[250:h-650, delta+450: w - (delta + 450)]

        width = 800
        height = 800
        dim = (width, height)

        # resize image
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(path_to_cropped_images_folder / image_path.stem / ".png"), image)


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
                # "15",
                path_to_cropped_images_folder,
                images_no_background,
            ]
        )
    else:
        segmentation(path_to_cropped_images_folder, images_no_background)


if __name__ == "__main__":
    main()
