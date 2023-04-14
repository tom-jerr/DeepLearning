import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
if __name__=='__main__':
    test1=np.load("valid_output_251.npy")

    im = Image.fromarray(test1[0, 0, :, :, 0] * 255.0)  # numpy 转 image类
    im = im.convert('RGB')
    plt.imshow(im)
    plt.show()
    print(test1)