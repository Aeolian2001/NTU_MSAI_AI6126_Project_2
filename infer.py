import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.srresnet_arch import MSRResNet

# configuration
####### Modify to your paths
model_path = './net_g_855000.pth'
folder = './dataset/test/LQ'
output_path = 'results/855000_ttk_2'
############################

device = 'cuda'
device = torch.device(device)

# set up model
model = MSRResNet(
    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=24, upscale=4)
print(f'Number of Params: {sum(p.numel() for p in model.parameters())}')
model.load_state_dict(torch.load(model_path)['params'], strict=True)
model.eval()
model = model.to(device)

os.makedirs(output_path, exist_ok=True)
for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
    imgname = os.path.splitext(os.path.basename(path))[0]
    print(idx, imgname)
    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]],
                                        (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)
    # inference
    with torch.no_grad():
        output = model(img)
        
        # TTA
        img_flip = torch.flip(img, dims=[3])  # flip
        output_flip = model(img_flip)
        output_flip = torch.flip(output_flip, dims=[3])  # flip back

        img_vflip = torch.flip(img, dims=[2])
        output_vflip = model(img_vflip)
        output_vflip = torch.flip(output_vflip, dims=[2])

        # Average both outputs
        output = (output + output_flip + output_vflip) / 3
        
    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(f'{output_path}/{imgname}.png', output)