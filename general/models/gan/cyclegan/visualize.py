

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""

    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(titles[i]) if titles else None
    return axes


def imshow(Xh, size=(20, 10)):
    """shows fake images"""

    args = get_args()

    Xh = Xh.detach().cpu()
    fix, ax = plt.subplots(figsize=size) if size else plt.subplots()
    plt.imshow(
        vutils.make_grid(Xh, nrow=int(len(Xh) ** 0.5), padding=2, normalize=True).permute(1, 2, 0)
    )
    plt.axis("off")
    # plt.show()
    plt.pause(1)



def DLoss(real, fake):
    return torch.mean((real - 1) ** 2) + torch.mean(fake**2)

