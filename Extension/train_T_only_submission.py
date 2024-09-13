from __future__ import print_function
import os
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchvision import transforms

from models.resnet import ResNet18
from models.resnet_transition import ResNet18_T
import torch.backends.cudnn as cudnn

from data import data_dataset
import numpy as np

import torchattacks

parser = argparse.ArgumentParser(description='PyTorch CIFAR Standard_MAN2 Adversarial Training')

parser.add_argument('--nat-img-train', type=str, help='natural training data', default='./data/train_images.npy')
parser.add_argument('--nat-label-train', type=str, help='natural training label', default='./data/train_labels.npy')
parser.add_argument('--nat-img-test', type=str, help='natural test data', default='./data/test_images.npy')
parser.add_argument('--nat-label-test', type=str, help='natural test label', default='./data/test_labels.npy')

parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=4e-1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--epsilon', default=8/255,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2/255,
                    help='perturb step size')

parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--target-model-dir', default='./checkpoint/resnet_18/ori',
                    help='directory of model for saving checkpoint')
parser.add_argument('--model-dir', default='./checkpoint/resnet_18/MAN_only_T/ori',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr

    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1 # 45.86 80.28
    elif epoch >= 60:
        lr = args.lr * 0.5


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def craft_adversarial_example(model, x_natural, y, step_size=2/255, epsilon=8/255, perturb_steps=10):

    attack = torchattacks.PGD(model, eps=epsilon, alpha=step_size, steps=perturb_steps, random_start=True)

    x_adv = attack(x_natural, y)

    return x_adv



def craft_adversarial_example_spe(classifier, T_model, x_natural, x_adv, step_size=2 / 255, epsilon=8 / 255, perturb_steps=10):
    x_adv_spe = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv_spe.requires_grad_()
        with torch.enable_grad():
            logits_adv = classifier(x_adv_spe)
            outputs = F.softmax(logits_adv, dim=1)
            _, y_adv = torch.max(outputs.data, 1)

            T_adv = T_model(x_adv)
            T_adv_spe = T_model(x_adv_spe)

            loss_1 = F.cross_entropy(logits_adv, y_adv)
            loss_2 = nn.MSELoss()(T_adv_spe, T_adv)

            loss = loss_2 - loss_1

        grad = torch.autograd.grad(loss, [x_adv_spe])[0]
        x_adv_spe = x_adv_spe.detach() + step_size * torch.sign(grad.detach())
        x_adv_spe = torch.min(torch.max(x_adv_spe, x_natural - epsilon), x_natural + epsilon)
        x_adv_spe = torch.clamp(x_adv_spe, 0.0, 1.0)

    return x_adv_spe


def standard_loss(classifier, T_model, x_natural, y, optimizer, step_size=2/255, epsilon=8/255, perturb_steps=10, distance='l_inf'):

    # generate adversarial example
    classifier.eval()
    T_model.eval()

    x_adv = craft_adversarial_example(classifier, x_natural, y, step_size=step_size, epsilon=epsilon,
                                      perturb_steps=perturb_steps)

    x_adv_spe = craft_adversarial_example_spe(classifier, T_model, x_natural, x_adv, step_size=step_size, epsilon=epsilon,
                                                   perturb_steps=perturb_steps)

    T_model.train()

    optimizer.zero_grad()

    logits = classifier(x_natural)  # .detach()
    T_pre = T_model(x_natural)
    pred_labels = F.softmax(logits, dim=1)
    noisy_post = torch.bmm(pred_labels.unsqueeze(1), T_pre).squeeze(1)
    logits = torch.log(noisy_post + 1e-12)
    loss_nat = nn.NLLLoss()(logits, y)

    logits = classifier(x_adv)  # .detach()
    T_pre = T_model(x_adv)
    pred_labels = F.softmax(logits, dim=1)
    noisy_post = torch.bmm(pred_labels.unsqueeze(1), T_pre).squeeze(1)
    logits = torch.log(noisy_post + 1e-12)
    loss_adv = nn.NLLLoss()(logits, y)

    T_spe = T_model(x_adv_spe)
    loss_T = nn.MSELoss()(T_spe, T_pre)

    '''
    # Add loss_T
    
    same_idx = (pred_adv.squeeze()==pred_adv_all.squeeze()).nonzero()
    # diff_idx = (pred_adv.squeeze()!=pred_adv_all.squeeze()).nonzero()
    # print('The number of same idxs is {}'.format(len(same_idx)))

    loss_T_same = nn.MSELoss()(T_adv[same_idx.squeeze()], T_adv_all[same_idx.squeeze()])
    # loss_T_diff = nn.MSELoss()(T_adv[diff_idx], T_adv_all[diff_idx])

    loss_T = loss_T_same # - loss_T_diff
    '''
    loss = loss_adv + 0.1 * loss_nat + 700.0 * loss_T

    return loss


def train(args, classifier, T_model, device, train_loader, optimizer, epoch):

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss

        loss = standard_loss(classifier=classifier, T_model=T_model, x_natural=data, y=target, optimizer=optimizer,
                             step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.num_steps,
                             distance='l_inf')

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))


def main():
    # settings
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # setup data loader
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = data_dataset(img_path=args.nat_img_train, clean_label_path=args.nat_label_train,
                            transform=trans_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=False,
                                               shuffle=True, num_workers=4, pin_memory=True)
    testset = data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)

    # init model, ResNet18() can be also used here for training
    classifier = ResNet18(10).to(device)
    classifier.load_state_dict(torch.load(os.path.join(args.target_model_dir, 'best_model.pth')))
    classifier = torch.nn.DataParallel(classifier)

    T_model = ResNet18_T(100).to(device)
    T_model = torch.nn.DataParallel(T_model)
    cudnn.benchmark = True
    optimizer = optim.SGD(T_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # best_pred = 0

    classifier.eval()

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        # start = perf_counter()

        train(args, classifier, T_model, device, train_loader, optimizer, epoch)

        if epoch % args.save_freq == 0:
            print('================================================================')
            torch.save(T_model.module.state_dict(),
                       os.path.join(args.model_dir, 'best_model.pth'))
            print('save the model')

        print('================================================================')


if __name__ == '__main__':
    main()
