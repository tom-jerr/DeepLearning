# -*- coding: utf-8 -*-


import sys
import cv2
import matplotlib.pyplot as plt


def get_selective_search():
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return gs


def config(gs, img, strategy='q'):
    gs.setBaseImage(img)

    if (strategy == 's'):
        gs.switchToSingleStrategy()
    elif (strategy == 'f'):
        gs.switchToSelectiveSearchFast()
    elif (strategy == 'q'):
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)


def get_rects(gs):
    rects = gs.process()
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]

    return rects


if __name__ == '__main__':
    """
    选择性搜索算法操作
    """
    gs = get_selective_search()
    path = "utils/lena.jpg"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    config(gs, img, strategy='q')
    print("config is OK!")
    rects = get_rects(gs)
    print(rects)
    imout = img.copy()

    for i,rect in enumerate(rects):
        if (i<100):
            x,y,w,h = rect
            cv2.rectangle(imout,(x,y),(x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break

    plt.imshow(imout)
    # plt.imshow(img)
    plt.show()
