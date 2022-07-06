import os
import cv2
import json
import math
import click
import numpy as np
from pathlib import Path


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def closest_point_2_lines(oa, da, ob, db):
    """
    Returns point closest to both rays of form o+t*d,
    and a weight factor that goes to 0 if the lines are parallel
    """

    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def get_colmap_cameras(text_folder):
    with open(os.path.join(text_folder, "cameras.txt"), "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            els = line.split(" ")
            w, h = float(els[2]), float(els[3])
            fl_x, fl_y = float(els[4]), float(els[4])
            k1, k2, p1, p2 = 0, 0, 0, 0
            cx, cy = w / 2, h / 2

            if els[1] == "SIMPLE_PINHOLE":
                cx, cy = float(els[5]), float(els[6])

            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6]), float(els[7])

            elif els[1] == "SIMPLE_RADIAL":
                cx, cy = float(els[5]), float(els[6])
                k1 = float(els[7])

            elif els[1] == "RADIAL":
                cx, cy = float(els[5]), float(els[6])
                k1, k2 = float(els[7]), float(els[8])

            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx, cy = float(els[6]), float(els[7])
                k1, k2 = float(els[8]), float(els[9])
                p1, p2 = float(els[10]), float(els[11])

            else:
                print("unknown camera model ", els[1])

            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi

    print(
        f"camera:"
        f"\n\tres={w,h}"
        f"\n\tcenter={cx,cy}\n\t"
        f"focal={fl_x,fl_y}\n\t"
        f"fov={fovx,fovy}\n\t"
        f"k={k1,k2} "
        f"p={p1,p2} "
    )

    return w, h, cx, cy, fl_x, fl_y, fovx, fovy, k1, k2, p1, p2, angle_x, angle_y


def get_colmap_images(text_folder, image_folder, out):
    with open(os.path.join(text_folder, "images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        up = np.zeros(3)
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i % 2 == 1:
                elems = line.split(" ")
                image_rel = os.path.relpath(image_folder)
                name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
                b = sharpness(name)
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                c2w[0:3, 2] *= -1  # flip the y and z axis
                c2w[0:3, 1] *= -1
                c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
                c2w[2, :] *= -1  # flip whole world upside down
                up += c2w[0:3, 1]

                file_path = f"../images/{Path(name).stem}"
                frame = {
                    "file_path": file_path,
                    "sharpness": b,
                    "transform_matrix": c2w,
                }
                out["frames"].append(frame)
    return out, up


@click.command()
@click.option("--aabb_scale", default=4)
@click.option("--image_folder", default="data/processed/images")
@click.option("--colmap_text_folder", default="data/processed/colmap_db/colmap_text")
@click.option("--output", default="data/processed/configs/nerf_transforms.json")
def colmap2nerf(aabb_scale, image_folder, colmap_text_folder, output):
    """
    Convert colmap format to nerf
    @param aabb_scale: large scene scale factor.
    1=scene fits in unit cube; power of 2 up to 16
    @param image_folder: input path to the images
    @param colmap_text_folder: path to colmap text folder
    @param output: name of output file
    """
    print("start colmap2nerf")
    (
        w,
        h,
        cx,
        cy,
        fl_x,
        fl_y,
        fovx,
        fovy,
        k1,
        k2,
        p1,
        p2,
        angle_x,
        angle_y,
    ) = get_colmap_cameras(colmap_text_folder)

    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": aabb_scale,
        "frames": [],
    }

    out, up = get_colmap_images(colmap_text_folder, image_folder, out)
    nframes = len(out["frames"])

    # don't keep colmap coords - reorient the scene to be easier to work with
    up = up / np.linalg.norm(up)
    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(
            R, f["transform_matrix"]
        )  # rotate up to be the z axis

    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3, :]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= totp

    avglen = 0.0
    for f in out["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
    avglen /= nframes
    print("avg camera distance from origin", avglen)
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes, "frames")
    print(f"writing {output}")
    with open(output, "w") as outfile:
        json.dump(out, outfile, indent=2)


if __name__ == "__main__":
    colmap2nerf()
