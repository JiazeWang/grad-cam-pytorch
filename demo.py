#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
from hacnn import HACNN
from mgn import MGN
import click
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models, transforms

from grad_cam import BackPropagation, Deconvnet, GradCAM, GuidedBackPropagation
from functools import partial
import pickle

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def save_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))


def save_gradcam(filename, gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@click.command()
@click.option("-i", "--image-path", type=str, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-t", "--target-layer", type=str, required=True)
@click.option("-k", "--topk", type=int, default=1)
@click.option("--cuda/--no-cuda", default=True)
def main(image_path, target_layer, arch, topk, cuda):

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on the GPU:", torch.cuda.get_device_name(current_device))
    else:
        print("Running on the CPU")

    # Synset words
    classes = list()


    # Model from torchvision
    model = MGN()
    """
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    checkpoint = torch.load("hacnn_market_xent.pth.tar", pickle_module=pickle)

        #checkpoint = torch.load(args.load_weights)
    #checkpoint = torch.load("hacnn_market_xent.pth.tar")
    pretrain_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    """
    pretrain_dict = torch.load("model_700.pt")
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    #model = models.__dict__[arch](pretrained=True)

    model.to(device)
    model.eval()
    place = []
    with open("samples/new.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    # Image preprocessing
    with open("new.txt") as lines:
        for line in lines:
            line = line.strip()
            line = "/mnt/SSD/jzwang/market1501/query/"+line
            place.append(line)
    for line in place:
        image_path = line
        raw_image = cv2.imread(image_path)[..., ::-1]
        raw_image = cv2.resize(raw_image, (128, 384))
        image = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )(raw_image).unsqueeze(0)
        image = image.to(device)

        """
        Common usage:
        1. Wrap your model with visualization classes defined in grad_cam.py
        2. Run forward() with an image
        3. Run backward() with a specific class
        4. Run generate() to export result
        """

        # =========================================================================
        print("Vanilla Backpropagation")

        bp = BackPropagation(model=model)
        predictions = bp.forward(image)

        for i in range(topk):
            print("[{:.5f}] {}".format(predictions[i][0], classes[predictions[i][1]]))

            bp.backward(idx=predictions[i][1])
            gradient = bp.generate()


        # Remove all the hook function in the "model"
        bp.remove_hook()

        # =========================================================================
        print("Deconvolution")

        deconv = Deconvnet(model=model)
        _ = deconv.forward(image)

        for i in range(topk):
            print("[{:.5f}] {}".format(predictions[i][0], classes[predictions[i][1]]))

            deconv.backward(idx=predictions[i][1])
            gradient = deconv.generate()

        deconv.remove_hook()

        # =========================================================================
        print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM")

        gcam = GradCAM(model=model)
        _ = gcam.forward(image)


        gbp = GuidedBackPropagation(model=model)

        _ = gbp.forward(image)

        t = ['ha1', 'p1', 'p2', 'p3']
        for i in range(topk):
            print("[{:.5f}] {}".format(predictions[i][0], classes[predictions[i][1]]))

            # Grad-CAM
            for target_layer in t:
                gcam.backward(idx=predictions[i][1])
                #print("1")
                region = gcam.generate(target_layer=target_layer)
                #print(2)
                save_gradcam(
                    "results/{}-gradcam-{}.png".format(
                        line, target_layer
                    ),
                    region,
                    raw_image,
                )
                #print(3)

                # Guided Backpropagation
                gbp.backward(idx=predictions[i][1])
                gradient = gbp.generate()

                # Guided Grad-CAM
                h, w, _ = gradient.shape
                region = cv2.resize(region, (w, h))[..., np.newaxis]
                output = gradient * region



if __name__ == "__main__":
    main()
