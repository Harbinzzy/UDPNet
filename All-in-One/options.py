import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=150, help='maximum number of epochs to train the total model.')
parser.add_argument('--every_epochs', type=int, default=150, help='maximum number of epochs to train the total model.')

parser.add_argument('--batch_size', type=int,default=8,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=24, help='number of workers.')
# parser.add_argument('--total_steps', type=int, default=2603100, help='Total training steps')
# parser.add_argument('--logs_name', type=str, default="EM-Prompt", help='log name')


# path
parser.add_argument('--data_file_dir', type=str, default='/data2t/AirNet-Data/data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_path', type=str, default="/data2t/AirNet-Data/test/denoise/", help='save path of test noisy images')
parser.add_argument('--derain_path', type=str, default="/data2t/AirNet-Data/test/derain/", help='save path of test raining images')
parser.add_argument('--dehaze_path', type=str, default="/data2t/AirNet-Data/test/dehaze/", help='save path of test hazy images')
parser.add_argument('--lol_path', type=str, default="/data2t/AirNet-Data/test/lolv1/", help='save path of test hazy images')
parser.add_argument('--gopro_path', type=str, default="/data2t/AirNet-Data/test/gopro/", help='save path of test hazy images')
parser.add_argument('--denoise_dir', type=str, default='/data2t/AirNet-Data/data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='/data2t/AirNet-Data/data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='/data2t/AirNet-Data/data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--lol_dir', type=str, default='/data2t/AirNet-Data/data/Train/LOL/',
                    help='where training images of dehazing saves.')
parser.add_argument('--gopro_dir', type=str, default='/data2t/AirNet-Data/data/Train/GoPro/',
                    help='where training images of dehazing saves.')
parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'gopro','lol'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="promptir",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--input_ckpt",type=str,default="ckpt/model.ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 4,help = "Number of GPUs to use for training")
parser.add_argument("--logs_name",type=str,default= "try",help = "Number of GPUs to use for training")


options = parser.parse_args()

