import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor

import albumentations as A
import cv2
import matplotlib.pyplot as plt
# import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import pytorch as AP
from PIL import Image
from tqdm import tqdm

jpeg70 = A.ImageCompression(70, 70, p=1.0)
jpeg80 = A.ImageCompression(80, 80, p=1.0)
jpeg90 = A.ImageCompression(90, 90, p=1.0)
jpeg100 = A.ImageCompression(100, 100, p=1.0)

jpeg = A.Compose(
    [
        A.OneOf(
            [jpeg70, jpeg80, jpeg90, jpeg100],
            p=1,
        ),
    ]
)


nothing = A.NoOp()

options = [jpeg70, jpeg80, jpeg90, jpeg100, nothing]


def set_augment(option):
    _augment = A.Compose([option])
    return _augment


def augment(_augment, x):
    return _augment(image=x)["image"]


class HighPassFilter:
    """docstring"""

    def __init__(self):
        pass

    def high_pass(self, x):
        """applies median blur filter"""
        x = F.pad(x, (1, 1, 1, 1), "reflect")
        b, c, w, h = x.shape

        unfold = torch.nn.Unfold(3, padding=1)
        windows = unfold(x).permute(0, 2, 1)
        d = windows.shape[1]
        windows = windows.reshape(b, d, c, -1)

        medians, _ = windows.median(-1)
        medians = medians.permute(0, 2, 1)
        n = int(medians.shape[2] ** 0.5)

        out = medians.reshape(b, c, n, n)
        out = (x - out)[:, :, 1:-1, 1:-1]  # don't need the padding anymore
        return out

    def apply(self, x):
        """docstring"""
        x = torch.stack([x[:, :, 0], x[:, :, 1], x[:, :, 2]])
        x = x.reshape(1, *x.shape).float()
        print(x.shape)
        return self.high_pass(x)

    def __call__(self, x):
        hp = self.high_pass(x)
        return torch.cat([x, hp], 1)


def spec_np(image):
    """docstring"""

    fft_image = np.fft.fft2(image)  # Perform 2D FFT
    shifted_fft = np.fft.fftshift(
        fft_image
    )  # Shift the zero-frequency component to the center of the spectrum
    magnitude_spectrum = np.abs(
        shifted_fft
    )  # Compute the magnitude spectrum (absolute values)
    fft = (
        np.log(1 + magnitude_spectrum).astype(np.float32) ** 4 / 100
    )  # Apply logarithmic transformation for better visualization
    return fft


def spec_img(img):
    """makes spectral image"""

    # Apply 2D FFT
    fft = torch.fft.fft2(img)

    # Compute magnitude of FFT output for visualization
    # We shift the zero frequency component to the center of the spectrum
    fft = torch.fft.fftshift(fft)
    fft = F.normalize(torch.log(torch.abs(fft)))

    return fft


def read_img(file):
    """docstring"""

    # img = Image.open(file).convert("L")
    img = Image.open(file)
    img = torch.from_numpy(np.array(img))
    high = HighPassFilter().apply(img)
    fft = spec_img(high)

    return img, fft


def process_img(path):
    # Load image and convert to grayscale

    name = path.split("/")[-1]

    files = os.listdir(path)
    files = [osp.join(path, f) for f in files]  # [:5]

    # imgs, ffts = zip(*list(tqdm(map(read_img, files), total=len(files))))
    with ThreadPoolExecutor() as ex:
        imgs, ffts = zip(*list(tqdm(ex.map(read_img, files), total=len(files))))

    img = torch.stack(imgs)
    # img = F.normalize(img.float(), p=2, dim=0)
    img = img.float().mean(dim=0)

    fft = torch.stack(ffts)
    fft = fft[~fft.mean(dim=(1, 2)).isinf()]

    # print(fft.min(), fft.max())
    # fft = fft.clamp(-10000, fft.max())
    # fft = F.normalize(fft.float(), p=2, dim=(0))
    # print(fft.min(), fft.max())
    fft = fft.float().mean(dim=0)

    avg = spec_img(img)  # avg of ffts

    plot(img, fft, name)
    # input('continue? ')
    return img, fft


def process_all(path):
    files = os.listdir(path)
    files = [osp.join(path, f) for f in files]  # [:5]

    for file in tqdm(files):
        name = path.split("/")[-1]
        img, fft = read_img(file)
        plot(img, fft, name)


def plot(imgs, names=None, title=""):
    """docstring"""

    names = ["" for _ in imgs] if not names else names

    n = len(imgs)
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))
    font = {}  #  {"fontname": "Arial", "fontsize": 48, "fontweight": "bold"}

    for i, img in enumerate(imgs):
        if len(img.shape) == 3:
            axs[i].imshow(img)
        else:
            axs[i].imshow(img, cmap="gray")
        axs[i].set_title(names[i].capitalize(), **font)

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle(title)

    plt.show()


def plot_collections(collections, names, title):
    row = len(collections)
    col = len(collections[0])

    fig, axs = plt.subplots(row, col, figsize=(3 * row, 3 * col))
    font = {}  #  {"fontname": "Arial", "fontsize": 48, "fontweight": "bold"}

    for i, c in enumerate(collections): 
       for j, img in enumerate(c):

            # if len(img.shape) == 3:
            if j in [1, 2]:
                axs[i][j].imshow(img, cmap="gray")
            else:
                axs[i][j].imshow(img)

            if i == 0: # topmost
                axs[i][j].set_title(names[j].capitalize())
            if j == 0: # leftmost
                axs[i][j].set_ylabel(title[i].capitalize())

    axs = axs.flatten()
    for ax in axs:
        # ax.axis("off")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.grid(False)

    plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # fig.suptitle(title)
    plt.show()
    quit()


def main():
    file = "final_dontdelete/celeba-hq.jpg"
    # file = "/Users/matthewhyatt/cs/.datasets/wblot/real/img_00164.png"
    file = '/Users/matthewhyatt/cs/.datasets/wblot/synth/cyclegan/img_00001.png'
    # file = '/Users/matthewhyatt/cs/.datasets/wblot/synth/cyclegan/img_00053.png'

    base = np.array(Image.open(file).convert("L"))
    names = ["jpeg70", "jpeg80", "jpeg90", "jpeg100", "nothing"]

    imgs, ffts, highs, blurs = [], [], [], []
    collections = []
    for i, o in enumerate(options):
        _aug = set_augment(o)
        img = augment(_aug, base)
        imgs.append(img)
        # img = torch.from_numpy(np.array(img))

        blurred = cv2.medianBlur(img, 3)
        blurs.append(blurred)
        high = img - blurred
        highs.append(high)

        # high = HighPassFilter().apply(img)
        fft = spec_np(torch.Tensor(high))
        ffts.append(fft)

        img = np.array(Image.open(file))
        collection = [img, blurred, high, fft]  # high[0]]
        collections.append(collection)

        subtitles = ["original", "blur", "high pass", "fft(high pass)"]
        # plot(collection, names=subtitles, title=names[i])
    plot_collections(collections, names=subtitles, title=names)

    quit()

    # highs = [highs[-1]-x for x in highs]
    plot(highs, names, title=f"Difference of High Pass Filters ( no_augment - x )")

    highs = [base - b for b in blurs]
    highs = [base - b for b in blurs]
    plot(
        highs,
        names,
        title=f"Degredation of High Pass Filters ( base_img_no_augment - blur )",
    )

    imgs = [x - imgs[-1] for x in imgs]
    plot(imgs, names, title=f"Degredation of Image ( base_img - aug_img )")

    ffts = [ffts[-1] - x for x in ffts]
    plot(ffts, names, title=f"Difference of FFT ( base_fft - aug_fft )")


if __name__ == "__main__":
    main()
