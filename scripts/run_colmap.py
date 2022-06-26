import os
import sys
import click
import shutil
from pathlib import Path


def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


@click.command()
@click.option("--images", default="images", help="input path to the images")
@click.option("--colmap_db", default="colmap.db", help="colmap database filename")
@click.option(
    "--text",
    default="colmap_text",
    help="input path to the colmap text files",
)
@click.option(
    "--colmap_camera_model",
    default="OPENCV",
    type=click.Choice(
        ["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"]
    ),
    help="camera model",
)
@click.option(
    "--colmap_camera_params",
    default="",
    help="intrinsic parameters, depending on the chosen model"
    "Format: fx,fy,cx,cy,dist",
)
@click.option(
    "--colmap_matcher",
    default="sequential",
    type=click.Choice(
        ["exhaustive", "sequential", "spatial", "transitive", "vocab_tree"]
    ),
    help="select which matcher colmap should use. sequential for videos,"
    " exhaustive for adhoc images",
)
def run_colmap(
    images, colmap_db, text, colmap_camera_model, colmap_camera_params, colmap_matcher
):
    # vars preprocessing

    # images
    images = '"' + images + '"'

    # db
    db = colmap_db
    db_noext = str(Path(db).with_suffix(""))

    # text
    if text == "text":
        text = db_noext + "_text"

    # sparse
    sparse = db_noext + "_sparse"

    print(
        f"running colmap with:"
        f"\n\timages={images}\n\tdb={db}\n\tsparse={sparse}\n\ttext={text}"
    )

    warning_msg = (
        input(
            f"warning! folders '{sparse}' and '{text}' "
            f"will be deleted/replaced. continue? (Y/n)"
        )
        .lower()
        .strip()
        + "y"
    )

    if warning_msg[:1] != "y":
        sys.exit(1)
    if os.path.exists(db):
        os.remove(db)

    do_system(
        f"colmap feature_extractor "
        f"--ImageReader.camera_model {colmap_camera_model} "
        f'--ImageReader.camera_params "{colmap_camera_params}" '
        f"--SiftExtraction.estimate_affine_shape=true "
        f"--SiftExtraction.domain_size_pooling=true "
        f"--ImageReader.single_camera 1 "
        f"--database_path {db} "
        f"--image_path {images}"
    )

    do_system(
        f"colmap {colmap_matcher}_matcher "
        f"--SiftMatching.guided_matching=true "
        f"--database_path {db}"
    )

    try:
        shutil.rmtree(sparse)
    except:
        pass

    do_system(f"mkdir {sparse}")

    do_system(
        f"colmap mapper "
        f"--database_path {db} "
        f"--image_path {images} "
        f"--output_path {sparse}"
    )

    do_system(
        f"colmap bundle_adjuster "
        f"--input_path {sparse}/0 "
        f"--output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1"
    )

    try:
        shutil.rmtree(text)
    except:
        pass

    do_system(f"mkdir {text}")

    do_system(
        f"colmap model_converter "
        f"--input_path {sparse}/0 "
        f"--output_path {text} "
        f"--output_type TXT"
    )


if __name__ == "__main__":
    run_colmap()
