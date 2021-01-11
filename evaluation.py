import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn

from torch.autograd import Variable
from torchsummary import summary

from dataloader import get_dataloader
from model import VanillaVAE

from skimage.metrics import ( peak_signal_noise_ratio, structural_similarity )
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.nn import functional as F
import matplotlib.pyplot as plt


def Float_tensor_to_numpy_uint8(img):
    img *= 255
    img = img.to(torch.uint8)
    return img.numpy()

def main(model_path, model_name):	
    trn_dataset, eval_dataset, trn_loader, eval_loader = get_dataloader(batch_size=1, num_workers=4)

    z_dim = int( model_path[ model_path.find('z_dim')+len('z_dim') : model_path.find('_WD_')] )
    model = VanillaVAE( in_channels=1,
                        latent_dim=z_dim, 
                        hidden_dims=[32, 64, 64, 128, 256])
    print(summary(model, (1, 64, 64), device='cpu'))

    checkpoint = torch.load(os.path.join(model_path, model_name))
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.cuda()
    else:
        device = torch.device('cpu')

    def calculate_metrics(mode, loader, device):
        total_mse = 0
        total_psnr = 0
        total_ssim = 0
        latent_features = []
        for i, mini_batch in enumerate(loader):
            input_data = mini_batch.to(device)
            result = model(input_data)

            latent_features.append(result[2].cpu().detach().squeeze(0).numpy())

            MSE = F.mse_loss(result[1].cpu(), result[0].cpu())

            original_image = Float_tensor_to_numpy_uint8(result[1].cpu().detach().squeeze(0).squeeze(0))
            recon_image = Float_tensor_to_numpy_uint8(result[0].cpu().detach().squeeze(0).squeeze(0))
            PSNR = peak_signal_noise_ratio(original_image, recon_image)
            SSIM = structural_similarity(original_image, recon_image)

            plt.imsave("result_{}/original_image_{}.png".format(mode, i), original_image)
            plt.imsave("result_{}/recon_image_{}.png".format(mode, i), recon_image)

            total_mse += MSE.item()
            total_psnr += PSNR
            total_ssim += SSIM
        latent_features = np.array(latent_features)

        pca_model = PCA(n_components=2)
        pca_features = pca_model.fit_transform(latent_features)

        tsne = TSNE(random_state=0)
        tsne_features = tsne.fit_transform(latent_features)

        return total_mse/len(loader), total_psnr/len(loader), total_ssim/len(loader), pca_features, tsne_features

    trn_mse, trn_psnr, trn_ssim, trn_pca_features, trn_tsne_features = calculate_metrics('train', trn_loader, device)
    print(f"Trn  - Avg_MSE: {trn_mse} Avg_PSNR: {trn_psnr} Avg_SSIM: {trn_ssim}")
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(trn_pca_features[:,0], trn_pca_features[:,1])
    ax[1].scatter(trn_tsne_features[:,0], trn_tsne_features[:,1])
    fig.suptitle('Train set latent feature plot')
    ax[0].set_title("PCA")
    ax[1].set_title("T-sne")

    eval_mse, eval_psnr, eval_ssim, eval_pca_features, eval_tsne_features = calculate_metrics('eval', eval_loader, device)
    print(f"Eval - Avg_MSE: {eval_mse} Avg_PSNR: {eval_psnr} Avg_SSIM: {eval_ssim}")
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(eval_pca_features[:,0], eval_pca_features[:,1])
    ax[1].scatter(eval_tsne_features[:,0], eval_tsne_features[:,1])
    fig.suptitle('Evaluation set latent feature plot')
    ax[0].set_title("PCA")
    ax[1].set_title("T-sne")

    plt.show()


print('[start script] {}'.format(os.path.abspath(__file__)))
if __name__ == '__main__':

    model_path = 'log/ep_2000_steps10_lr_0.0003_bs_512_z_dim32_WD_0.0_wKLD_MAX_0.0001_start0.0_duration2000.0_cudaTrue'
    model_name = 'VAE_last.pkl'

    main(model_path, model_name)
