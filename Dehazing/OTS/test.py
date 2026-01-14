import os
import time
import torch
import argparse
import sys
from utils import Adder
from pytorch_msssim import ssim
import torch.nn.functional as f
from data import test_dataloader
from torch.backends import cudnn
from torchvision.transforms import functional as F
from skimage.metrics import peak_signal_noise_ratio
from models.FSNet_UDPNet import build_net

def _eval(model, args):
    state_dict = torch.load(args.test_model, weights_only=False)['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    factor = 8 #32

    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()

        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            tm = time.time()

            pred = model(input_img)[2]
            pred = pred[:,:,:h,:w]

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()


            label_img = (label_img).cuda()
            psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))
            down_ratio = max(1, round(min(H, W) / 256))	
            ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))), 
                            f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False)	
            print('%d iter PSNR_dehazing: %.3f ssim: %f' % (iter_idx + 1, psnr_val, ssim_val))
            ssim_adder(ssim_val)

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)
            
            psnr_mimo = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr_val)

            print('%d iter PSNR: %.3f time: %f' % (iter_idx + 1, psnr_mimo, elapsed))

        print('==========================================================')
        print('The average PSNR is %.3f dB' % (psnr_adder.average()))
        print('The average SSIM is %.5f dB' % (ssim_adder.average()))

        print("Average time: %f" % adder.average())

def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net()
    # print(model)

    if torch.cuda.is_available():
        model.cuda()
    _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='FSNet', type=str)
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--data_dir', type=str, default='/data1t/reside-outdoor')
    # Test
    parser.add_argument('--test_model', type=str, default='/home/zzy/DepthDehaze/OTS/results/dual2_epoch_on/checkpoints/epoch_epoch=19_psnr_val_psnr=40.43.ckpt')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.result_dir = os.path.join('results/', args.model_name, 'test')
    os.makedirs(args.result_dir, exist_ok=True)
    results_file = os.path.join(os.path.join('results/', args.model_name), 'results.txt')
    # sys.stdout = open(results_file, 'w')
    # sys.stderr = sys.stdout
    print(args)
    main(args)
