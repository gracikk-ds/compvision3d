import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader

# custom modules
from src.unet.u2net import U2NET
from src.data.data_loader import (
    RescaleT,
    ToTensorLab,
    SalObjDataset,
)


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def normalization(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn * 255


def segmentation(
    images: str,
    output: str,
    trimap: bool = True,
    model_dir: str = "checkpoints/u2net.pth",
):
    """
    Run segmentation model
    @param model_dir: model checkpoint file
    @param images: input images folder
    @param output: output path
    @param trimap: create trimap or not
    """
    ROOT = get_project_root()
    images = ROOT / images
    output = ROOT / output
    output_masks = output / "masks"
    output_rgba = output / "rgba"
    output_masks.mkdir(exist_ok=True, parents=True)
    output_rgba.mkdir(exist_ok=True, parents=True)
    print("output: ", output)
    print("images: ", images)

    net = U2NET(3, 1)
    net.load_state_dict(torch.load(str(ROOT / model_dir), map_location=torch.device("cpu")))
    net.eval()

    files = list(images.glob("*.jpg"))
    loader = DataLoader(
        SalObjDataset(
            img_name_list=files,
            lbl_name_list=[],
            transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]),
        ),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=1,
    )

    with torch.no_grad():
        for data, filename in tqdm(zip(loader, files), total=len(files)):
            img_rgb = cv2.imread(str(filename))
            img_rgba = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2RGBA)

            d1, d2, d3, d4, d5, d6, d7 = net(data["image"].type(torch.FloatTensor))

            u2netresult = normalization(d1[:, 0, :, :]).squeeze(0).numpy()
            u2netresult = cv2.resize(
                u2netresult, (img_rgb.shape[1], img_rgb.shape[0])
            ).astype(np.uint8)

            if trimap:
                u2netresult[(u2netresult <= 235) & (u2netresult != 0)] = 127
                u2netresult[u2netresult >= 235] = 255

            else:
                u2netresult[u2netresult <= 230] = 0
                u2netresult[u2netresult >= 230] = 255

            img_rgba[:, :, 3] = u2netresult

            cv2.imwrite(
                str(output_masks / filename.with_suffix(".png").name), u2netresult
            )

            cv2.imwrite(
                str(output_rgba / filename.with_suffix(".png").name), img_rgba
            )
