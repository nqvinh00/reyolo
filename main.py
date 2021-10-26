import torch
from .darknet import Darknet
from .helpers import test_input

module = Darknet("./cfg/yolov3.cfg")
module.load_weight("./weight/yolov3.weights")
width, height = int(module.net["width"]), int(module.net["height"])
img = test_input("./test_img/dog-cycle-car.png", (height, width))
pred = module(img, torch.cuda.is_available())
print(pred)
