import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import U2NET, U2NETP  # Import U^2-Net models

# Normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi)

# Save the output image
def save_output(image_name, pred, output_path):
    predict = pred.squeeze().cpu().data.numpy()
    im = Image.fromarray(predict * 255).convert('L')  # 'L' for grayscale

    # Resize output mask to original image size
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    # Construct output path with folder structure
    relative_path = os.path.relpath(image_name, INPUT_FOLDER)
    output_image_path = os.path.join(output_path, relative_path)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    imo.save(output_image_path)

# Main function
def main():
    model_name = 'u2net'  # Full model (173.6 MB) | Change to 'u2netp' for lightweight version

    # Modified Input/Output Paths
    global INPUT_FOLDER  # Used inside `save_output` function
    INPUT_FOLDER = "../../assets/laplacian_filtered_images"
    OUTPUT_FOLDER = "../../assets/segmented_images"

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    # Collect images recursively from all folders
    img_name_list = glob.glob(os.path.join(INPUT_FOLDER, '**', '*.jpg'), recursive=True)

    print(f"Found {len(img_name_list)} images for segmentation.")

    # Dataloader
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    # Model Initialization
    if model_name == 'u2net':
        print("...Loading U2NET (173.6 MB)")
        net = U2NET(3, 1)
    elif model_name == 'u2netp':
        print("...Loading U2NETP (4.7 MB)")
        net = U2NETP(3, 1)

    # Model Loading
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # Create Output Directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Inference and Saving Results
    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("Processing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image'].type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, _, _, _, _, _, _ = net(inputs_test)

        # Normalize and Save
        pred = normPRED(d1[:, 0, :, :])
        save_output(img_name_list[i_test], pred, OUTPUT_FOLDER)

        del d1  # Clear memory for next batch

    print(f"Segmentation complete! Results saved in '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    main()
