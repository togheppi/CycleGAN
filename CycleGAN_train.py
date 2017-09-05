import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import Generator, Discriminator
import utils
import argparse
import os, itertools
from logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='apple2orange', help='input dataset')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--num_resnet', type=int, default=6, help='number of resnet blocks in generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=200, help='number of train epochs')
parser.add_argument('--decay_epoch', type=int, default=200, help='start decaying learning rate after this number')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lambdaA', type=float, default=10, help='lambdaA for cycle loss')
parser.add_argument('--lambdaB', type=float, default=10, help='lambdaB for cycle loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()
print(params)

# Directories for loading data and saving results
data_dir = '../Data/' + params.dataset + '/'
save_dir = params.dataset + '_results/'
model_dir = params.dataset + '_model/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Data pre-processing
transform = transforms.Compose([transforms.Scale(params.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Train data
train_data_A = DatasetFromFolder(data_dir, subfolder='trainA', transform=transform,
                               resize_scale=params.resize_scale, crop_size=params.crop_size, fliplr=params.fliplr)
train_data_loader_A = torch.utils.data.DataLoader(dataset=train_data_A,
                                                  batch_size=params.batch_size,
                                                  shuffle=True)
train_data_B = DatasetFromFolder(data_dir, subfolder='trainB', transform=transform,
                               resize_scale=params.resize_scale, crop_size=params.crop_size, fliplr=params.fliplr)
train_data_loader_B = torch.utils.data.DataLoader(dataset=train_data_B,
                                                  batch_size=params.batch_size,
                                                  shuffle=True)

# Test data
test_data_A = DatasetFromFolder(data_dir, subfolder='testA', transform=transform)
test_data_loader_A = torch.utils.data.DataLoader(dataset=test_data_A,
                                                 batch_size=params.batch_size,
                                                 shuffle=False)
test_data_B = DatasetFromFolder(data_dir, subfolder='testB', transform=transform)
test_data_loader_B = torch.utils.data.DataLoader(dataset=test_data_B,
                                                 batch_size=params.batch_size,
                                                 shuffle=False)
test_real_A = test_data_loader_A.__iter__().__next__()
test_real_B = test_data_loader_B.__iter__().__next__()

# Models
G_A = Generator(3, params.ngf, 3, 6)
G_B = Generator(3, params.ngf, 3, 6)
D_A = Discriminator(3, params.ndf, 1)
D_B = Discriminator(3, params.ndf, 1)
G_A.normal_weight_init(mean=0.0, std=0.02)
G_B.normal_weight_init(mean=0.0, std=0.02)
D_A.normal_weight_init(mean=0.0, std=0.02)
D_B.normal_weight_init(mean=0.0, std=0.02)
G_A.cuda()
G_B.cuda()
D_A.cuda()
D_B.cuda()


# Set the logger
D_A_log_dir = save_dir + 'D_A_logs'
D_B_log_dir = save_dir + 'D_B_logs'
if not os.path.exists(D_A_log_dir):
    os.mkdir(D_A_log_dir)
D_A_logger = Logger(D_A_log_dir)
if not os.path.exists(D_B_log_dir):
    os.mkdir(D_B_log_dir)
D_B_logger = Logger(D_B_log_dir)

G_A_log_dir = save_dir + 'G_A_logs'
G_B_log_dir = save_dir + 'G_B_logs'
if not os.path.exists(G_A_log_dir):
    os.mkdir(G_A_log_dir)
G_A_logger = Logger(G_A_log_dir)
if not os.path.exists(G_B_log_dir):
    os.mkdir(G_B_log_dir)
G_B_logger = Logger(G_B_log_dir)

cycle_A_log_dir = save_dir + 'cycle_A_logs'
cycle_B_log_dir = save_dir + 'cycle_B_logs'
if not os.path.exists(cycle_A_log_dir):
    os.mkdir(cycle_A_log_dir)
cycle_A_logger = Logger(cycle_A_log_dir)
if not os.path.exists(cycle_B_log_dir):
    os.mkdir(cycle_B_log_dir)
cycle_B_logger = Logger(cycle_B_log_dir)

img_log_dir = save_dir + 'img_logs'
if not os.path.exists(img_log_dir):
    os.mkdir(img_log_dir)
img_logger = Logger(img_log_dir)


# Loss function
MSE_loss = torch.nn.MSELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()

# optimizers
G_optimizer = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=params.lrG, betas=(params.beta1, params.beta2))
D_A_optimizer = torch.optim.Adam(D_A.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))
D_B_optimizer = torch.optim.Adam(D_B.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))

# Training GAN
D_A_avg_losses = []
D_B_avg_losses = []
G_A_avg_losses = []
G_B_avg_losses = []
cycle_A_avg_losses = []
cycle_B_avg_losses = []
step = 0
for epoch in range(params.num_epochs):
    D_A_losses = []
    D_B_losses = []
    G_A_losses = []
    G_B_losses = []
    cycle_A_losses = []
    cycle_B_losses = []

    # learning rate decay
    if (epoch + 1) > params.decay_epoch:
        D_A_optimizer.param_groups[0]['lr'] -= params.lrD / (params.num_epochs - params.decay_epoch)
        D_B_optimizer.param_groups[0]['lr'] -= params.lrD / (params.num_epochs - params.decay_epoch)
        G_optimizer.param_groups[0]['lr'] -= params.lrG / (params.num_epochs - params.decay_epoch)

    # training
    for i, (realA, realB) in enumerate(zip(train_data_loader_A, train_data_loader_B)):

        # input & target image data
        realA = Variable(realA.cuda())
        realB = Variable(realB.cuda())

        # Train generator G
        # A -> B
        fakeB = G_A(realA)
        D_B_fake_decision = D_B(fakeB)
        G_A_loss = MSE_loss(D_B_fake_decision, Variable(torch.ones(D_B_fake_decision.size()).cuda()))

        # forward cycle loss
        reconA = G_B(fakeB)
        cycle_A_loss = L1_loss(reconA, realA) * params.lambdaA

        # B -> A
        fakeA = G_B(realB)
        D_A_fake_decision = D_A(fakeA)
        G_B_loss = MSE_loss(D_A_fake_decision, Variable(torch.ones(D_A_fake_decision.size()).cuda()))

        # backward cycle loss
        reconB = G_A(fakeA)
        cycle_B_loss = L1_loss(reconB, realB) * params.lambdaB

        # Back propagation
        G_loss = G_A_loss + G_B_loss + cycle_A_loss + cycle_B_loss
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # Train discriminator D_A
        D_A_real_decision = D_A(realA)
        D_A_real_loss = MSE_loss(D_A_real_decision, Variable(torch.ones(D_A_real_decision.size()).cuda()))
        D_A_fake_decision = D_A(fakeA)
        D_A_fake_loss = MSE_loss(D_A_fake_decision, Variable(torch.zeros(D_A_fake_decision.size()).cuda()))

        # Back propagation
        D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
        D_A_optimizer.zero_grad()
        D_A_loss.backward()
        D_A_optimizer.step()

        # Train discriminator D_B
        D_B_real_decision = D_B(realB)
        D_B_real_loss = MSE_loss(D_B_real_decision, Variable(torch.ones(D_B_real_decision.size()).cuda()))
        D_B_fake_decision = D_B(fakeB)
        D_B_fake_loss = MSE_loss(D_B_fake_decision, Variable(torch.zeros(D_B_fake_decision.size()).cuda()))

        # Back propagation
        D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
        D_B_optimizer.zero_grad()
        D_B_loss.backward()
        D_B_optimizer.step()

        # loss values
        D_A_losses.append(D_A_loss.data[0])
        D_B_losses.append(D_B_loss.data[0])
        G_A_losses.append(G_A_loss.data[0])
        G_B_losses.append(G_B_loss.data[0])
        cycle_A_losses.append(cycle_A_loss.data[0])
        cycle_B_losses.append(cycle_B_loss.data[0])

        print('Epoch [%d/%d], Step [%d/%d], D_A_loss: %.4f, D_B_loss: %.4f, G_A_loss: %.4f, G_B_loss: %.4f'
              % (epoch+1, params.num_epochs, i+1, len(train_data_loader_A), D_A_loss.data[0], D_B_loss.data[0], G_A_loss.data[0], G_B_loss.data[0]))

        # ============ TensorBoard logging ============#
        D_A_logger.scalar_summary('losses', D_A_loss.data[0], step + 1)
        D_B_logger.scalar_summary('losses', D_B_loss.data[0], step + 1)
        G_A_logger.scalar_summary('losses', G_A_loss.data[0], step + 1)
        G_B_logger.scalar_summary('losses', G_B_loss.data[0], step + 1)
        cycle_A_logger.scalar_summary('losses', cycle_A_loss.data[0], step + 1)
        cycle_B_logger.scalar_summary('losses', cycle_B_loss.data[0], step + 1)
        step += 1

    D_A_avg_loss = torch.mean(torch.FloatTensor(D_A_losses))
    D_B_avg_loss = torch.mean(torch.FloatTensor(D_B_losses))
    G_A_avg_loss = torch.mean(torch.FloatTensor(G_A_losses))
    G_B_avg_loss = torch.mean(torch.FloatTensor(G_B_losses))
    cycle_A_avg_loss = torch.mean(torch.FloatTensor(cycle_A_losses))
    cycle_B_avg_loss = torch.mean(torch.FloatTensor(cycle_B_losses))

    # avg loss values for plot
    D_A_avg_losses.append(D_A_avg_loss)
    D_B_avg_losses.append(D_B_avg_loss)
    G_A_avg_losses.append(G_A_avg_loss)
    G_B_avg_losses.append(G_B_avg_loss)
    cycle_A_avg_losses.append(cycle_A_avg_loss)
    cycle_B_avg_losses.append(cycle_B_avg_loss)

    # Show result for test image
    test_real_A = Variable(test_real_A.cuda())
    test_fake_B = G_A(test_real_A)
    test_recon_A = G_B(test_fake_B)
    utils.plot_test_result(test_real_A, test_fake_B, test_recon_A, epoch, save=True, save_dir=save_dir + 'AtoB')

    test_real_B = Variable(test_real_B.cuda())
    test_fake_A = G_B(test_real_B)
    test_recon_B = G_A(test_fake_A)
    utils.plot_test_result(test_real_B, test_fake_A, test_recon_B, epoch, save=True, save_dir=save_dir + 'BtoA')

    # log the images
    info = {
        'real_image': utils.to_np(test_real_A.view(-1, params.input_size, params.input_size)),
        'gen_image': test_fake_B.view(-1, params.input_size, params.input_size),
        'recon_image': test_recon_A.view(-1, params.input_size, params.input_size)
    }

    for tag, images in info.items():
        img_logger.image_summary(tag, images, epoch + 1)



# Plot average losses
avg_losses = []
avg_losses.append(D_A_avg_losses)
avg_losses.append(D_B_avg_losses)
avg_losses.append(G_A_avg_losses)
avg_losses.append(G_B_avg_losses)
avg_losses.append(cycle_A_avg_losses)
avg_losses.append(cycle_B_avg_losses)
utils.plot_loss(avg_losses, params.num_epochs, save=True, save_dir=save_dir)

# Make gif
utils.make_gif(params.dataset, params.num_epochs, save_dir=save_dir)

# Save trained parameters of model
torch.save(G_A.state_dict(), model_dir + 'generator_A_param.pkl')
torch.save(G_B.state_dict(), model_dir + 'generator_B_param.pkl')
torch.save(D_A.state_dict(), model_dir + 'discriminator_A_param.pkl')
torch.save(D_B.state_dict(), model_dir + 'discriminator_B_param.pkl')
