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

import numpy as np
import quaternion
import json


@click.group()
def main():
    """Entrypoint for scripts"""
    pass


@main.command()
@click.option("--data1", default="data/milk_artur/", type=str)
@click.option("--data2", default="data/processed/", type=str)
def read_imgtxt(data1, data2, ):
    with open(os.path.join(data1,'meta.json')) as m1:
        meta1 = json.load(m1)
    with open(os.path.join(data2,'meta.json')) as m2:
        meta2= json.load(m2)
    
    if os.path.exists('data/mixup'):
        os.system(f'rm -r data/mixup')
    os.makedirs('data/mixup')
    os.mkdir('data/mixup/images')
    os.mkdir('data/mixup/colmap_db')
    os.system(f"cp {os.path.join(data1, 'colmap_db/cameras.txt')} data/mixup/colmap_db/cameras.txt") 
    
    img_id = 0
    with open('data/mixup/colmap_db/images.txt', 'w') as out:
        # from data1
        data2_scale = (float(meta2['camera_height'])-float(meta2['stand_height']))/(float(meta1['camera_height'])-float(meta1['stand_height']))
        with open(os.path.join(data1,'colmap_db/images.txt')) as f:
            i = 0
            for line in f:
                line = line.strip()
                if line[0] == "#":
                    continue
                i = i + 1
                      
                if i > (75*2): # first 75 frames since there are 150 total
                    break
                if i % 2 == 1:
                    elems = line.split(
                        " ")  # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                    elems[0] = str(img_id)
                    img_id += 1
                    out.write(' '.join(elems)+'\n')
                    out.write('0 0 -1\n')
                    os.system(f"cp {os.path.join(data1, 'images', elems[-1])} {os.path.join('data/mixup/images', elems[-1])}") 
                         
        # from data2
        with open(os.path.join(data2,'colmap_db/images.txt')) as f:
            i = 0
            for line in f:
                line = line.strip()
                if line[0] == "#":
                    continue
                i = i + 1
                if i < (75*2): # after 75 frames since there are 150 total
                    continue
                if i % 2 == 1:
                    elems = line.split(
                        " ")  # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                    elems[0] = str(img_id)
                    img_id += 1
                    for el in range(5,8):
                        elems[el] = str(float(elems[el])*data2_scale)
                    true_img_name = elems[-1]
                    elems[-1] = elems[-1].split('.')[0]+'_2.png'
                    out.write(' '.join(elems)+'\n')
                    out.write('0 0 -1\n')
                    os.system(f"cp {os.path.join(data2, 'images', true_img_name)} {os.path.join('data/mixup/images', elems[-1])}") 

                    
if __name__ == "__main__":
    main()
