import torch
import torchvision
import torchvision.models.detection as models
from PIL import Image
import cv2
import numpy as np
from torchvision.transforms import transforms as transforms
import matplotlib.pyplot as plt
import os
import random
import torch.nn.functional as F

def get_outputs(image, model, threshold):
    with torch.no_grad():
        outputs = model(image)

    # get all the scores
    scores = list(outputs[0]['scores'].numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get all the masks
    masks = (outputs[0]['masks']>0.6).squeeze().numpy()
    # discard masks for objects whose score is below the threshold
    masks = masks[:thresholded_preds_count]
    # discard scores below threshold
    scores = scores[:thresholded_preds_count]
     # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]

    return masks, scores

def draw_perturbation_map(orig_image, masks, camo):
    alpha = 1
    beta = 0.8
    gamma = 0

    # rget shape
    orig_image_shape = orig_image.shape # save the original shape of the image in format (B, C, H, W)

    # convert camo into size of image
    camo = torch.reshape(camo, (1, 3, 16, 16))
    camo = F.interpolate(camo, size=(orig_image_shape[1], orig_image_shape[2]))
    camo = camo.squeeze(0)

    image = orig_image.clone()
    
    # apply mask on the image
    for i in range(len(masks)):
        # convert to 3 channel mask
        mask = masks[i]
        mask = torch.tensor(mask)
        mask = torch.stack([mask, mask, mask], dim=0)
        # invert mask
        inverseMask = torch.bitwise_not(mask).type(torch.uint8)
        mask = mask.type(torch.uint8)
        # apply perturbation to mask
        mask = torch.multiply(camo, mask)
        # apply inverse inverse mask to image
        image = torch.multiply(image, (inverseMask+0.05)) # higher value means more transparency
        image = torch.add(image, mask)

    return image

def main():

    # open image
    image_path = "image.png"
    image_name = os.path.splitext(os.path.split(image_path)[1])[0]
    image = Image.open(image_path).convert('RGB')
    print(f"Image {image_name} loaded...")

    # load the model and set to eval
    weights = models.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = models.maskrcnn_resnet50_fpn_v2(weights=weights, progress=True, num_classes=91)
    model.eval()

    # transform to convert the image to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
   
    # transform the image
    image = transform(image)
    # keep a copy of the original image for OpenCV functions and applying masks
    orig_image = image.clone()
    
    # add a batch dimension and get the masks and original detection scores
    threshold = 0.90
    image = image.unsqueeze(0)
    masks, scores = get_outputs(image, model, threshold)

    # create 800 random camouflages 64 x 64 x 3
    num_camouflages = 10
    image_size = 16
    channels = 3

    camouflages = torch.zeros((num_camouflages, channels, image_size, image_size))

    for i in range(num_camouflages):
        camouflage = torch.randn((channels, image_size, image_size))
        camouflage = F.softmax(camouflage, dim=0)  # Apply softmax to ensure values are between 0 and 1
        camouflages[i] = camouflage

    # iterative camouflage attack
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    score = float('inf')
    attack_steps = 1
    step = 0

    while (step < attack_steps):
        
        # iterate through camouflages and find the one that minimizes the loss
        i = 0
        for camo in camouflages:
            # apply perturbation to image
            perturbated_image = draw_perturbation_map(orig_image, masks, camo)
            # save image
            print("Saving image...")
            torchvision.utils.save_image(perturbated_image, f"imagewmask_{i}.png")
            perturbated_image = perturbated_image.unsqueeze(0)
            # get the scores
            perturbated_masks, perturbated_scores = get_outputs(perturbated_image, model, threshold)
            print(f"Scores: {perturbated_scores}")
            perturbated_scores = torch.tensor(perturbated_scores)
            scores = torch.tensor(scores)
            zero = torch.zeros((len(perturbated_scores)))
            print(scores)
            print(perturbated_scores)
            # calculate the loss
            loss = cross_entropy_loss(perturbated_scores, scores)
            print(f"Loss: {loss}")
            # find the camouflage that minimizes the loss
            if loss < score:
                score = loss
                best_camo = camo
                print(f"New best camouflage found with loss {loss}")
                torchvision.utils.save_image(best_camo, "bestcamo.png")
            i += 1
        
        step += 1

    return 0

if __name__ == "__main__":
    main()