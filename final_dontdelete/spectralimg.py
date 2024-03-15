from concurrent.futures import ThreadPoolExecutor
import cv2
import os
import os.path as osp

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# import cv2
import numpy as np


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

    def __call__(self, x):
        hp = self.high_pass(x)
        return torch.cat([x, hp], 1)



def spec_img(img):
    """makes spectral image"""

    # Apply 2D FFT
    fft = torch.fft.fft2(img)

    # Compute magnitude of FFT output for visualization
    # We shift the zero frequency component to the center of the spectrum
    fft = torch.fft.fftshift(fft)
    fft = 20 * torch.log(torch.abs(fft))

    return fft


def high_pass(x):
    """docstring"""

    blur = cv2.medianBlur(x, 3)
    return torch.Tensor(x - blur)

def read_img(file):
    """docstring"""

    img = Image.open(file).convert("L")
    img = torch.from_numpy(np.array(img))
    high = high_pass(img.numpy())
    fft = spec_img(high)

    return img, fft


def process_img(path):
    # Load image and convert to grayscale

    name = path.split("/")[-1]

    files = os.listdir(path)
    files = [osp.join(path, f) for f in files] # [:5]

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
    files = [osp.join(path, f) for f in files] # [:5]

    for file in tqdm(files):
        name = path.split("/")[-1]
        img,fft = read_img(file)
        plot(img, fft, name, mode='show')
        quit()



def plot(img, fft, name, mode='save'):
    """docstring"""

    # Create subplots
    n = 2
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))

    # Original image
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Average Image")

    # Spectral image
    axs[1].imshow(fft)  # cmap="gray")
    axs[1].set_title("Average of FFTs")

    np.save(f"{name}_avg_img.np", img.numpy())
    np.save(f"{name}_ffc_avg.np", fft.numpy())

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if mode == 'save':
        plt.savefig(f"{name}.png")
    if mode == 'show':
        plt.show()


def plot_all(imgs, ffts, names):
    """docstring"""

    # Create subplots
    m, n = 2, len(imgs)
    fig, axs = plt.subplots(m, n, figsize=(5 * n, 5 * m))

    # Original image
    for i,(img,fft) in enumerate(zip(imgs,ffts)):
        axs[0][i].imshow(img, cmap="gray")
        axs[1][i].imshow(fft)
        font = {'fontname': 'Arial', 'fontsize': 48 , 'fontweight': 'bold'}
        axs[0][i].set_title(names[i].capitalize(), **font)

    for ax in axs:
        for a in ax:
            a.axis("off")

    plt.tight_layout()
    plt.show()
    quit()
    plt.savefig(f"allh.png")


def main():

    # path = input('img path: ')
    real = "/Users/matthewhyatt/cs/.datasets/western_blots/real"
    synth = "/Users/matthewhyatt/cs/.datasets/western_blots/synth"
    gans = [os.path.join(synth, s) for s in os.listdir(synth)]

    paths = gans + [real]
    names = [s.split("/")[-1] for s in paths]

    # paths = [x for x in paths if 'cyclegan' in x]

    imgs,ffts = zip(*[process_all(p) for p in paths])
    # imgs,ffts = zip(*[process_img(p) for p in paths])
    plot_all(imgs,ffts,names)

if __name__ == "__main__":
    main()
