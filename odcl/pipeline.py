from numpy.lib.shape_base import dsplit
import numpy as np
from collections import namedtuple
from tflite_runtime import interpreter
import cv2, platform, math, random, argparse
import colorsys
from termcolor import colored

_EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]

Target = namedtuple("Target", ["id", "score", "bbox"])


class BBox(object):
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.area = (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def overlap(self, other):
        """check if this bbox overlaps with another bbox"""
        xc = (self.xmin <= other.xmax) and (other.xmin <= self.xmax)
        yc = (self.ymin <= other.ymax) and (other.ymin <= self.ymax)
        if xc and yc:
            return True
        else:
            return False


class TargetDrawer(object):
    def __init__(self, labels):
        # get labels
        self.labels = labels
        # get colors
        self.colors = {}
        for l in labels.keys():
            self.colors[l] = self.get_rand_color()

    @staticmethod
    def get_rand_color():
        # get a random color
        rgb = colorsys.hsv_to_rgb(
            random.uniform(0, 1),
            random.uniform(0.6, 1),
            random.uniform(0.8, 1),
        )
        rgb = [min(255, int(c * 255)) for c in rgb]
        return rgb

    def draw_tile_frame(self, img, alpha=0.9):
        """draw a frame around the input. Useful for visualizing tiles."""
        pt1 = (1, 1)
        pt2 = (img.shape[1] - 1, img.shape[0] - 1)
        cpy = img.copy()
        img = cv2.rectangle(img, pt1, pt2, color=(255, 255, 255), thickness=2)
        return cv2.addWeighted(cpy, alpha, img, (1 - alpha), 0)

    def draw_target_bbox(self, img, target, color=None):
        """Draw a bbox, class label, and confidence score around a target onto image

        updated image with target drawn onto it
        """
        w, h = img.shape[1], img.shape[0]
        xmin, xmax = math.ceil(target.bbox.xmin * w), math.floor(target.bbox.xmax * w)
        ymin, ymax = math.ceil(target.bbox.ymin * h), math.floor(target.bbox.ymax * h)
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)

        if color is None:
            color = self.colors[target.id]

        # draw rectangle
        img = cv2.rectangle(img, pt1, pt2, color, 2)
        # draw text
        textpt = (pt1[0], pt1[1] + 25)
        text = self.labels[target.id] + " : " + str(round(target.score * 100))
        img = cv2.putText(
            img,
            text,
            textpt,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        return img

    def draw_all(self, img, targets, color=None):
        """Draw all current targets onto img

        Parameters
        ----------
        img : cv2 Image
            (H, W, 3) 8-bit
        targets: list of Target
            targets to draw

        Returns
        -------
        (H, W, 3) 8-bit image
            Image with targets drawn
        """
        for target in targets:
            img = self.draw_target_bbox(img, target, color=color)
        return img

    def make_target_bbox_img_opencv(self, img, targets):
        return self.draw_all(img, targets)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run inference on video stream")
    ap.add_argument("--model", type=str, required=True, help="path to tflite model")
    ap.add_argument("--labels", type=str, required=True, help="path to labels file")
    ap.add_argument(
        "--cpu",
        type=str,
        required=False,
        default="cpu",
        help=colored("cpu", "blue")
        + " if using CPU, "
        + colored("tpu", "blue")
        + " if using TPU.",
    )
    opts = ap.parse_args()

    from vsutils import VideoStreamCV
    from interpreter import TargetInterpreter

    target_interpreter = TargetInterpreter(opts.model, opts.labels, opts.cpu, 0.33)

    vs = VideoStreamCV()
    draw = TargetDrawer(target_interpreter.labels)
    while True:
        img = vs.get_img()
        target_interpreter.interpret(img)
        for t in target_interpreter.targets:
            obj_id_str = target_interpreter.labels[t.id]
            xmin = round(t.bbox[0], 2)
            ymin = round(t.bbox[1], 2)
            xmax = round(t.bbox[2], 2)
            ymax = round(t.bbox[3], 2)
            # obj_bbox_str = "({}, {}) to ({}, {})".format(xmin, ymin, xmax, ymax)
            # print("\tfound: id={}, bbox=[{}]".format(obj_id_str, obj_bbox_str))
        img = draw.draw_all(img, target_interpreter.targets)
        cv2.imshow("image", img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
