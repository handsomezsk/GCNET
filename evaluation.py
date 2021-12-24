import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
import tensorboardX as tX
import os
import shutil
from network import *
from read_data import *
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

h=256
w=512
maxdisp=96 #gc_net.py also need to change  must be a multiple of 32...maybe can cancel the outpadding of deconv
batch=1
num_epochs = 200
mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
device_ids = [0]

modeldir = './model/best_model.ckpt'

writer = tX.SummaryWriter(log_dir='log', comment='GCNet')
device = torch.device('cuda')

def main():
    test_dataset = KITTI2015('../dataset/data_scene_flow', mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    model = GcNet(h,w,maxdisp)
    model = nn.DataParallel(model)
    state_dict = torch.load(modeldir)
    model.load_state_dict(state_dict['state_dict'])
    
    validate(model, test_loader)
    
def validate(model, test_loader):
    '''
    validate 40 image pairs
    '''
    num_batches = len(validate_loader)
    idx = np.random.randint(num_batches)
    
    for i, batch in enumerate(test_loader):
        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)

        # plt.figure()
        # plt.imshow(target_disp.cpu().numpy().transpose(1,2,0).squeeze(2))
        # plt.show()
        # plt.close()
        start_time = time.time()
        with torch.no_grad():
            disp = model(left_img, right_img)
        print('spend time: {:.5}%'.format(time.time()-start_time))

        # error = torch.sum(delta > 3.0) / float(h * w )
        if i == idx:
            left_save = left_img
            disp_save = disp
        save_image(left_save[0], disp_save[0])

def save_image(left_image, disp):
    for i in range(3):
        left_image[i] = left_image[i] * std[i] + mean[i]
    b, r = left_image[0], left_image[2]
    left_image[0] = r  # BGR --> RGB
    left_image[2] = b
    # left_image = torch.from_numpy(left_image.cpu().numpy()[::-1])

    disp_img = disp.detach().cpu().numpy()
    fig = plt.figure( figsize=(12.84, 3.84) )
    plt.axis('off')  # hide axis
    plt.imshow(disp_img)
    plt.colorbar()

    writer.add_figure('image/disp', fig, global_step=None)
    writer.add_image('image/left', left_image, global_step=None)


if __name__=='__main__':
    main()
    writer.close()
   