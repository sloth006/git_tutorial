from PIL import Image # conda install pillow
import numpy as np # conda install numpy

if __name__ == '__main__':
    tensors = np.load("../results/initial_images_bchw.npy")

    # reference from torchvision
    # for each batch
    for i in range(tensors.shape[0]):
        tensor = (tensors[i] * 255 - 0.5).clip(0, 255).astype('uint8')
        tensor_hwc = np.moveaxis(tensor, 0, 2)
        img = Image.fromarray(tensor_hwc)
        img.save(f'./initial_img_{i}.png')

    tensors = np.load("../results/sampled_images_bchw.npy")

    # reference from torchvision
    # for each batch
    for i in range(tensors.shape[0]):
        tensor = (tensors[i] * 255 - 0.5).clip(0, 255).astype('uint8')
        tensor_hwc = np.moveaxis(tensor, 0, 2)
        img = Image.fromarray(tensor_hwc)
        img.save(f'./sampled_img_{i}.png')