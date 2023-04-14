import numpy as np
import matplotlib.pyplot as plt

#极大值抑制算法——剔除过多的搜索框
def NMS(dets, thresh):

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2-y1+1)*(x2-x1+1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22-x11+1)
        h = np.maximum(0, y22-y11+1)
        overlaps = w*h
        IOU = overlaps/(areas[i] + areas[index[1:]] - overlaps)

        print(np.where(IOU <= thresh))
        idx = np.where(IOU <= thresh)[0]    #获取保留下来的索引
        index = index[idx + 1]

    return keep

#显示搜索框
def plot_bbox(dets, c='k', title_name="title"):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    plt.plot([x1, x2], [y1, y1], c)
    plt.plot([x1, x1], [y1, y2], c)
    plt.plot([x1, x2], [y2, y2], c)
    plt.plot([x2, x2], [y1, y2], c)
    plt.title(title_name)

if __name__ == '__main__':
    boxes = np.array([[59, 120, 137, 368, 0.124648176],
                      [221, 89, 369, 367, 0.35818103],
                      [54, 154, 148, 382, 0.13638769]])


    plot_bbox(boxes, 'k', title_name="before nms")  # before nms
    plt.show()

    keep = NMS(boxes, thresh=0.35)

    plot_bbox(boxes[keep], 'r', title_name="after_nme")  # after nms
    plt.show()
