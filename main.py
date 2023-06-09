import torch
import torchvision
import torchvision.models.detection as models
from PIL import Image
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
    # get all the masks
    masks = (outputs[0]['masks']>0.6).squeeze().numpy()
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # discard masks for objects whose score is below the threshold
    masks = masks[:thresholded_preds_count]
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    #discard scores below 0.20 detection threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > 0.20]
    thresholded_preds_count = len(thresholded_preds_inidices)
    scores = scores[:thresholded_preds_count]

    return masks, scores

def draw_perturbation_map(orig_image, masks, camo):
    alpha = 1
    beta = 0.8
    gamma = 0

    # rget shape
    orig_image_shape = orig_image.shape # save the original shape of the image in format (B, C, H, W)

    # convert camo into size of image
    camo = F.interpolate(camo, size=(orig_image_shape[1], orig_image_shape[2]), mode='nearest')
    camo = camo.squeeze(0)
    #torchvision.utils.save_image(camo, 'camo.png')

    image = orig_image.clone()
    # apply mask on the image
    for i in range(len(masks[0:4])):
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
    image_path = "000179.png"
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
        #transforms.Resize((500, 500), antialias=True)
    ])
   
    # transform the image
    image = transform(image)
    # keep a copy of the original image for OpenCV functions and applying masks
    orig_image = image.clone()
    
    # add a batch dimension and get the masks and original detection scores
    threshold = 0.70
    image = image.unsqueeze(0)
    masks, scores = get_outputs(image, model, threshold) # boxes in boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x between 0 and W and values of y between 0 and H
                                                                # scores (Tensor[N]): the scores or each prediction.
                                                                # masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range.
    # create random camouflages 64 x 64 x 3
    num_camouflages = 10
    camo_size = 256
    channels = 3

    camouflages = torch.zeros((num_camouflages, channels, camo_size, camo_size))
    for i in range(num_camouflages):
        camouflage = torch.randn((channels, camo_size, camo_size))
        camouflage = F.softmax(camouflage, dim=0)  # Apply softmax to ensure values are between 0 and 1
        camouflages[i] = camouflage

    # iterative camouflage attack
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    score = 0
    attack_steps = 1
    step = 0

    while (step < attack_steps):
        
        # iterate through camouflages and find the one that minimizes the loss
        for camo in camouflages:
            # reshape camo
            camo = torch.reshape(camo, (1, 3, camo_size, camo_size))
            # apply perturbation to image
            perturbated_image = draw_perturbation_map(orig_image, masks, camo)
            perturbated_image = perturbated_image.unsqueeze(0)
            # get the scores
            perturbated_masks, perturbated_scores = get_outputs(perturbated_image, model, threshold)
            print(f"Scores: {scores[0:4]}\nPerturbated scores: {perturbated_scores[0:4]}")
            # perturbated_scores = torch.tensor(perturbated_scores, dtype=torch.float32)
            # mean_perturbated_scores = torch.mean(perturbated_scores).unsqueeze(0)
            # print(f"Mean score: {mean_scores}\nMean perturbated score: {mean_perturbated_scores}")
            #zero = torch.zeros((len(perturbated_scores)))
            # calculate the loss
            # loss = cross_entropy_loss(perturbated_scores, scores) # keeps coming out to -0.0??
            loss = []
            for i in range(len(perturbated_scores[0:4])):
                try:
                    loss.append(-perturbated_scores[i]*np.log(scores[i])-((1-perturbated_scores[i])*np.log(1-scores[i])))
                except:
                    continue
            #loss = -mean_perturbated_scores*np.log(mean_scores)-((1-mean_perturbated_scores)*np.log(1-mean_scores))
            loss = (1/len(perturbated_scores[0:4]))*sum(loss)
            print(f"Loss: {loss}")
            # find the camouflage that minimizes the loss
            if loss > score:
                score = loss
                best_camo = camo
                print(f"New best camouflage found with loss {loss}")
                torchvision.utils.save_image(best_camo, "bestcamo.png")
                # save image w/ camouflage
                print("Saving image with camouflage...")
                torchvision.utils.save_image(perturbated_image, f"imagewmask.png")

    return 0

if __name__ == "__main__":
    main()