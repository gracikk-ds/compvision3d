import os
import cv2
import json
import math
import click
import numpy as np
from pathlib import Path
import quaternion



@click.command()
@click.option("--transforms", default=r"data/materials/transforms_train.json", type=str)
@click.option("--output_path", default=r"data/processed/configs/nerf_transforms.json", type=str)
@click.option("--rot_noise", default=0.0, type=float)
@click.option("--tr_noise", default=0.0, type=float)
@click.option("--invalid_noise", default=0.0, type=float)
@click.option("--mode", default=0, type=int)
def colmap2nerf(transforms, output_path, rot_noise, tr_noise, invalid_noise, mode):
    
    with open(transforms, 'r') as inp:
        true = json.load(inp)
    
    for frame in true['frames']:
        frame['file_path'] = os.path.join('../../materials',frame['file_path'])
        if mode==0:
            m = np.array(frame['transform_matrix'])
            R = m[:3,:3]
            R = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(R)) + np.random.normal(0,rot_noise, size=(3,))
            R = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(R))
            m[:3,:3] = R

            t = m[3,:3]
            t = t + np.random.normal(0,tr_noise, size=(3,))
            m[3,:3] = t
        else:
            m = np.array(frame['transform_matrix'])
            m += np.random.normal(0,invalid_noise, size=(4,4))
        frame['transform_matrix'] = m.tolist()
    
    with open(output_path, 'w') as out:
        json.dump(true, out)

if __name__ == "__main__":
    colmap2nerf()
