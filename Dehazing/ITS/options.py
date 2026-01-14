import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--every_epochs', type=int, default=1100, help='maximum number of epochs to train the total model')
parser.add_argument('--batch_size', type=int, default=8, help='batch size to use per GPU')
parser.add_argument('--lr', type=float, default=4e-4, help='learning rate')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers')

# path
parser.add_argument('--train_data_dir', type=str, default='/data1t/dehaze/Haze4K', help='images of the training set')
parser.add_argument('--val_data_dir', type=str, default='/data1t/dehaze/Haze4K',help='images of the testing set')
parser.add_argument('--ckpt_path', type=str, default="ckpt", help='name of the directory where the checkpoint is to be resumed')
parser.add_argument("--ckpt_dir", type=str, default="train_ckpt", help = "name of the directory where the checkpoint is to be saved")
parser.add_argument("--input_ckpt", type=str, default="train_ckpt/model.ckpt", help = "Name of the Directory where the checkpoint is to be resumed")
parser.add_argument("--logs_name", type=str, default= "try", help = "Number of GPUs to use for training")

options = parser.parse_args()
