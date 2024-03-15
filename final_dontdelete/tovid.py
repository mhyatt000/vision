

import os
import imageio

def images_to_video(paths, outpath):
    
    imgs = [imageio.imread(path) for path in paths]
    imageio.mimsave(outpath, imgs, fps=2)  # Use your desired frame rate

paths = [
    "/Users/matthewhyatt/Desktop/celeba/nothing.png",
    "/Users/matthewhyatt/Desktop/celeba/jpeg100.png",
    "/Users/matthewhyatt/Desktop/celeba/jpeg90.png",
    "/Users/matthewhyatt/Desktop/celeba/jpeg80.png",
    "/Users/matthewhyatt/Desktop/celeba/jpeg70.png",
]

folder = "/".join(paths[0].split('/')[:-1])
print(folder)



images_to_video(paths, os.path.join(folder,'output.mp4'))
