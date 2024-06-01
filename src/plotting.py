from matplotlib import pyplot as plt


class Orientation:
    HORIZONTAL = 0
    VERTICAL = 1


def plot_image(file_name, image, title=None):
    plt.figure()
    if title is None:
        title = f"Image of {file_name}"
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_histogram(image, title=None):
    plt.figure()
    plt.hist(image.flatten(), bins=256, range=(0, 256), alpha=0.75)
    plt.title('Histogram of Gray Scale Image')
    plt.xlabel('Gray Level')
    plt.ylabel('Pixel Count')
    plt.show()


def plot_gray_level(data, coord, orientation: Orientation, file_name: str):
    plt.figure()
    if orientation == Orientation.HORIZONTAL:
        plt.plot(data[coord, :])
        plt.title(f"Change the gray level along the horizontal line x={coord} in {file_name}")
    else:
        plt.plot(data[:, coord])
        plt.title(f"Change the gray level along the vertical line y={coord} in {file_name}")
    plt.show()
