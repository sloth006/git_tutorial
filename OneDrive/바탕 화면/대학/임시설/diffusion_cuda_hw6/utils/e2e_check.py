from PIL import Image # conda install pillow
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == "__main__":
    # load the image
    img = Image.open("../test/test_img/seed7524.png")
    img = np.array(img).flatten().reshape(1, -1)

    # load the new image
    new_img = Image.open("sampled_img_0.png")
    new_img = np.array(new_img).flatten().reshape(1, -1)

    # similarity
    similarity = cosine_similarity(img, new_img).squeeze()

    print(f'Cosine similarity is {similarity: .3f}. Please check if the similarity is close to 1.0')