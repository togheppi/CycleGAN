import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import Generator
import utils
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='horse2zebra', help='input dataset')
parser.add_argument('--batch_size', type=int, default=1, help='test batch size')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--num_resnet', type=int, default=9, help='number of resnet blocks in generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
params = parser.parse_args()
print(params)

# Directories for loading data and saving results
data_dir = '../Data/' + params.dataset + '/'
save_dir = params.dataset + '_test_results/'
model_dir = params.dataset + '_model/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Data pre-processing
transform = transforms.Compose([transforms.Scale(params.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Test data
test_data_A = DatasetFromFolder(data_dir, subfolder='testA', transform=transform)
test_data_loader_A = torch.utils.data.DataLoader(dataset=test_data_A,
                                                 batch_size=params.batch_size,
                                                 shuffle=False)
test_data_B = DatasetFromFolder(data_dir, subfolder='testB', transform=transform)
test_data_loader_B = torch.utils.data.DataLoader(dataset=test_data_B,
                                                 batch_size=params.batch_size,
                                                 shuffle=False)

# Load model
G_A = Generator(3, params.ngf, 3, params.num_resnet)
G_B = Generator(3, params.ngf, 3, params.num_resnet)
G_A.cuda()
G_B.cuda()
G_A.load_state_dict(torch.load(model_dir + 'generator_A_param.pkl'))
G_B.load_state_dict(torch.load(model_dir + 'generator_B_param.pkl'))

# Test
for i, real_A in enumerate(test_data_loader_A):

    # input image data
    real_A = Variable(real_A.cuda())

    # A -> B -> A
    fake_B = G_A(real_A)
    recon_A = G_B(fake_B)

    # Show result for test data
    utils.plot_test_result(real_A, fake_B, recon_A, i, save=True, save_dir=save_dir + 'AtoB/')

    print('%d images are generated.' % (i + 1))

for i, real_B in enumerate(test_data_loader_B):

    # input image data
    real_B = Variable(real_B.cuda())

    # B -> A -> B
    fake_A = G_B(real_B)
    recon_B = G_A(fake_A)

    # Show result for test data
    utils.plot_test_result(real_B, fake_A, recon_B, i, save=True, save_dir=save_dir + 'BtoA/')

    print('%d images are generated.' % (i + 1))