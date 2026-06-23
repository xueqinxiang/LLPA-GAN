import os, sys
from pyexpat import features
import os.path as osp
import time
import random
import datetime
import argparse
from scipy import linalg
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid
from lib.utils import transf_to_CLIP_input, dummy_context_mgr
from lib.utils import mkdir_p, get_rank
from lib.datasets import prepare_data

from models.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist

############################################
from torchvision.transforms import RandomCrop
from networks.discriminator import word_level_correlation_focus_RF, ContextBlock
from lib.datasets import encode_tokens
import clip as clip

def prepare_labels(batch_size):
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    match_labels = Variable(torch.LongTensor(range(batch_size)))

    return real_labels, fake_labels, match_labels

############   GAN   ############
def train(dataloader, netG, netD, netC, GALIPNetD, text_encoder, image_encoder, optimizerG, optimizerD, scaler_G, scaler_D, args):
    batch_size = args.batch_size
    device = args.device
    epoch = args.current_epoch
    max_epoch = args.max_epoch
    z_dim = args.z_dim
    netG, netD, netC, image_encoder = netG.train(), netD.train(), netC.train(), image_encoder.train() # ******GALIP OK
    #netG, netD, image_encoder = netG.train(), netD.train(), image_encoder.train()

    # *************** 20230925
    GALIPNetD = GALIPNetD.train() # ******GALIP OK
    # *************** 20230925

    #####word level discriminator #####
    real_labels, fake_labels, match_labels = prepare_labels(batch_size)
    real_labels = real_labels.to(device)
    fake_labels = fake_labels.to(device)
    match_labels = match_labels.to(device)

    #contBlock = ContextBlock(256, 0.5) #DAMSMencoders
    #contBlock = ContextBlock(768, 0.5)  #ViT-L/14
    contBlock = ContextBlock(512, 0.5)  # ViT-B/32
    contBlock = contBlock.to(device)
    #####word level discriminator #####

    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=len(dataloader))
    for step, data in enumerate(dataloader, 0):
        ##############
        # Train D  
        ##############
        optimizerD.zero_grad()
        with torch.cuda.amp.autocast() if args.mixed_precision else dummy_context_mgr() as mpc:
            # prepare_data
            real, captions, CLIP_tokens, sent_emb, words_embs, keys, \
                class_ids, cap_lens, word_labels = prepare_data(data, text_encoder, device)

            real = real.requires_grad_()
            sent_emb = sent_emb.requires_grad_()
            words_embs = words_embs.requires_grad_()

            # predict real
            CLIP_real,real_emb = image_encoder(real)
            real_feats = GALIPNetD(CLIP_real)  # *************** 20230925 GALIP OK
            ###real_feats = netD(CLIP_real, sent_emb)
            real_feats_D = netD(real, sent_emb)
            pred_real, errD_real = predict_loss(netC, real_feats, sent_emb, negtive=False)  # *************** 20230925 GALIP OK
            errD_real_D = hinge_loss(real_feats_D, negtive=False)

            # predict mismatch
            mis_sent_emb = torch.cat((sent_emb[1:], sent_emb[0:1]), dim=0).detach()
            ###mis_real_feats = netD(CLIP_real, mis_sent_emb)
            mis_real_feats = netD(real, mis_sent_emb)
            errD_mis_D = hinge_loss(mis_real_feats, negtive=True)
            _, errD_mis = predict_loss(netC, real_feats, mis_sent_emb, negtive=True)  # *************** 20230925 GALIP OK
            # synthesize fake images
            noise = torch.randn(batch_size, z_dim).to(device)
            #netG.lstm.init_hidden(noise)
            if (args.multi_gpus == True):
                netG.module.lstm.init_hidden(noise)
            else:
                netG.lstm.init_hidden(noise)

            fake = netG(noise, sent_emb)
            CLIP_fake, fake_emb = image_encoder(fake)
            fake_feats = GALIPNetD(CLIP_fake.detach())  # *************** 20230925 GALIP OK
            ###fake_feats = netD(CLIP_fake.detach(), sent_emb)
            fake_feats_D = netD(fake.detach(), sent_emb)
            errD_fake_D = hinge_loss(fake_feats_D, negtive=True)
            _, errD_fake = predict_loss(netC, fake_feats, sent_emb, negtive=True) # *************** 20230925 GALIP OK
        # MA-GP
        if args.mixed_precision:
            errD_MAGP = MA_GP_MP(CLIP_real, sent_emb, pred_real, scaler_D) # *************** 20230925 GALIP OK
            #errD_MAGP = MA_GP_MP(real, sent_emb, real_feats, scaler_D)
        else:
            errD_MAGP = MA_GP_FP32(CLIP_real, sent_emb, pred_real)  # *************** 20230925 GALIP OK
            #errD_MAGP = MA_GP_FP32(real, sent_emb, real_feats)

        # whole D loss
        with torch.cuda.amp.autocast() if args.mixed_precision else dummy_context_mgr() as mpc:
            #errD = errD_real + (errD_fake + errD_mis)/2.0 + errD_MAGP #GALIP OK
            errD = errD_real + (errD_fake + errD_mis) / 2.0 + errD_MAGP \
                   + errD_real_D + (errD_fake_D + errD_mis_D) / 2.0
            #errD = errD_real + (errD_fake + errD_mis) / 2.0

        #####word level discriminator #####
        #### CLIP_real [3, 1024, 16, 16] # ViT-L/14
        result, resultfake = word_level_correlation_focus_RF(CLIP_real, words_embs,
                                            cap_lens, batch_size, class_ids, real_labels, word_labels,\
                                            contBlock, CLIP_fake)

        #print('\n errD1 = ', errD, errD_real, errD_MAGP) #errD_MAGP: 1.7117e+14
        #errD += ((1.0/1000.0)*(result) + (1.0/100.0)*(resultfake)) #Batch size 12 nf 32
        errD += ((1.0 / 10000.0) * (result) + (1.0 / 1000.0) * (resultfake))  # Batch size 12 nf 32
        #####word level discriminator #####
        #print('\n errD = ', errD)

        # update D
        if args.mixed_precision:
            scaler_D.scale(errD).backward()
            scaler_D.step(optimizerD)
            scaler_D.update()
            if scaler_D.get_scale()<args.scaler_min:
                scaler_D.update(16384.0)
        else:
            errD.backward()
            optimizerD.step()
        ##############
        # Train G  
        ##############
        optimizerG.zero_grad()
        with torch.cuda.amp.autocast() if args.mixed_precision else dummy_context_mgr() as mpc:
            #fake_feats = netD(CLIP_fake, sent_emb)
            #fake_feats_D = netD(fake, sent_emb)
            fake_feats = GALIPNetD(CLIP_fake) # *************** 20230925 GALIP OK
            #*****output = netC(fake_feats, sent_emb)
            #loss_gen = (-fake_feats_D).mean()
            #*****loss_gen = (-output).mean()
            #loss_gen = loss_gen * args.loss_gen

            #Minimize spherical distance between image and text features
            # clip_loss = 0
            # gen_img = fake
            # if args.clip_weight > 0:
            #     if gen_img.size(-1) > 64:
            #         gen_img = RandomCrop(64)(gen_img)
            #     gen_img = F.interpolate(gen_img, 224, mode='area')
            #     gen_img_features, gen_img_emb = image_encoder(gen_img.add(1).div(2))
            #     #*****clip_loss = spherical_distance(gen_img_features, sent_emb).mean()
            #     contained_nan = np.isnan(gen_img_emb.detach().cpu())
            #     contained_nan = contained_nan.numpy()
            #     if np.any(contained_nan == 1):
            #         print('gen_img_emb: contained_nan')
            #     clip_loss = spherical_distance(gen_img_emb, sent_emb).mean()
            #     # *****clip_loss = torch.cosine_similarity(gen_img_emb, sent_emb).mean()


            output = netC(fake_feats, sent_emb) # *************** 20230925 GALIP OK
            text_img_sim = torch.cosine_similarity(fake_emb, sent_emb).mean()
            errG = -output.mean() - args.sim_w * text_img_sim
            #errG = -output.mean() - args.sim_w * text_img_sim + loss_gen
            #errG = (loss_gen + args.clip_weight * clip_loss)
            ##errG = (loss_gen - args.sim_w * clip_loss)
        #print('\n loss_gen = ', loss_gen)
        #print('\n clip_loss = ', clip_loss)
        #print('errG = ', errG, -output.mean(), text_img_sim)
        #print(' ')

        if args.mixed_precision:
            scaler_G.scale(errG).backward()
            scaler_G.step(optimizerG)
            scaler_G.update()
            if scaler_G.get_scale()<args.scaler_min:
                scaler_G.update(16384.0)
        else:
            errG.backward()
            optimizerG.step()
        # update loop information
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            loop.update(1)
            loop.set_description(f'Train Epoch [{epoch}/{max_epoch}]')
            loop.set_postfix()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop.close()


def test(dataloader, text_encoder, netG, PTM, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size, multi_gpus):
    FID, TI_sim = calculate_FID_CLIP_sim(dataloader, text_encoder, netG, PTM, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size, multi_gpus)
    return FID, TI_sim


def save_model(netG, netD, netC, optG, optD, epoch, multi_gpus, step, save_path):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}, \
                'optimizers': {'optimizer_G': optG.state_dict(), 'optimizer_D': optD.state_dict()},\
                'epoch': epoch}
        torch.save(state, '%s/state_epoch_%03d_%03d.pth' % (save_path, epoch, step))


#########   MAGP   ########
def MA_GP_MP(img, sent, out, scaler):
    grads = torch.autograd.grad(outputs=scaler.scale(out),
                            inputs=(img, sent),
                            grad_outputs=torch.ones_like(out),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    inv_scale = 1./(scaler.get_scale()+float("1e-8"))
    #inv_scale = 1./scaler.get_scale()
    grads = [grad * inv_scale for grad in grads]
    with torch.cuda.amp.autocast():
        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0,grad1),dim=1)                        
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp


def MA_GP_FP32(img, sent, out):
    grads = torch.autograd.grad(outputs=out,
                            inputs=(img, sent),
                            grad_outputs=torch.ones(out.size()).cuda(),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    ###grad0 = grads[0].view(grads[0].size(0), -1)
    ###grad1 = grads[1].view(grads[1].size(0), -1)
    grad0 = grads[0].contiguous().view(grads[0].size(0), -1)
    grad1 = grads[1].contiguous().view(grads[1].size(0), -1)
    grad = torch.cat((grad0,grad1),dim=1)                        
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp = 2.0 * torch.mean((grad_l2norm) ** 6)
    # *****d_loss_gp = 0.01 * torch.mean((grad_l2norm) ** 1)
    return d_loss_gp

def gen_sample(netG, text_encoder, save_dir, device, multi_gpus, z_dim):
    """
    generate sample according to user defined captions.

    caption should be in the form of a list, and each element of the list is a description of the image in form of string.
    caption length should be no longer than 18 words.
    example captions see below
    """
    #captions = ['some horses in a field of green grass with a sky in the background']
    #captions = ['some horses in a field of green grass with a sunset in the background']
    #captions = ['some horses in a field of green grass with a road in the background']
    #captions = ['some horses in a field of green grass with a mountain in the background']

    #captions = ['a landscape in winter']
    captions = ['a landscape in spring']

    # captions = ['A herd of black and white cattle standing on a field']

    # captions = ['A herd of black and white cattle standing on a field',
    #  'A herd of black cattle standing on a field',
    #  'A herd of white cattle standing on a field',
    #  'A herd of brown cattle standing on a field',
    #  'A herd of black and white sheep standing on a field',
    #  'A herd of black sheep standing on a field',
    #  'A herd of white sheep standing on a field',
    #  'A herd of brown sheep standing on a field']

    split_dir = 'valid'
    s_tmp = save_dir
    fake_img_save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(fake_img_save_dir)

    tokens = clip.tokenize(captions, truncate=True)
    #CLIP_tokens = tokens[0]
    CLIP_tokens = tokens
    CLIP_tokens = CLIP_tokens.to(device)
    batch_size = 1

    for step in range(50):

        #######################################################
        # (1) Generate fake images
        ######################################################
        with torch.no_grad():

            sent_emb, words_embs = encode_tokens(text_encoder, CLIP_tokens)
            noise = torch.randn(batch_size, z_dim).to(device)
            if (multi_gpus == True):
                netG.module.lstm.init_hidden(noise)
            else:
                netG.lstm.init_hidden(noise)
            #netG.lstm.init_hidden(noise)
            fake_imgs = netG(noise, sent_emb, eval=True).float()
            fake_imgs = torch.clamp(fake_imgs, -1., 1.)

            #stage_mask = stage_masks[-1]
        for j in range(batch_size):
            # save generated image
            s_tmp = '%s/img' % fake_img_save_dir
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            im = fake_imgs[j].data.cpu().numpy()
            # [-1, 1] --> [0, 255]
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            # fullpath = '%s_%3d.png' % (s_tmp,i)
            fullpath = '%s_%d.png' % (s_tmp, step)
            im.save(fullpath)

def sample(dataloader, netG, text_encoder, save_dir, device, multi_gpus, z_dim, stamp):
    netG.eval()
    for step, data in enumerate(dataloader, 0):
        ######################################################
        # (1) Prepare_data
        ######################################################
        real, captions, CLIP_tokens, sent_emb, words_embs, keys, _, _, _ = prepare_data(data, text_encoder, device)
        ######################################################
        # (2) Generate fake images
        ######################################################
        batch_size = sent_emb.size(0)
        with torch.no_grad():
            noise = torch.randn(batch_size, z_dim).to(device)
            if (multi_gpus == True):
                netG.module.lstm.init_hidden(noise)
            else:
                netG.lstm.init_hidden(noise)
            #netG.lstm.init_hidden(noise)
            fake_imgs = netG(noise, sent_emb, eval=True).float()
            fake_imgs = torch.clamp(fake_imgs, -1., 1.)
            if multi_gpus==True:
                batch_img_name = 'step_%04d.png'%(step)
                batch_img_save_dir  = osp.join(save_dir, 'batch', str('gpu%d'%(get_rank())), 'imgs')
                batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
                batch_txt_name = 'step_%04d.txt'%(step)
                batch_txt_save_dir  = osp.join(save_dir, 'batch', str('gpu%d'%(get_rank())), 'txts')
                batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)
            else:
                batch_img_name = 'step_%04d.png'%(step)
                batch_img_save_dir  = osp.join(save_dir, 'batch', 'imgs')
                batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
                batch_txt_name = 'step_%04d.txt'%(step)
                batch_txt_save_dir  = osp.join(save_dir, 'batch', 'txts')
                batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)
            mkdir_p(batch_img_save_dir)
            vutils.save_image(fake_imgs.data, batch_img_save_name, nrow=8, value_range=(-1, 1), normalize=True)
            mkdir_p(batch_txt_save_dir)
            txt = open(batch_txt_save_name,'w')
            for cap in captions:
                txt.write(cap+'\n')
            txt.close()
            for j in range(batch_size):
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                ######################################################
                # (3) Save fake images
                ######################################################      
                if multi_gpus==True:
                    single_img_name = 'batch_%04d.png'%(j)
                    single_img_save_dir  = osp.join(save_dir, 'single', str('gpu%d'%(get_rank())), 'step%04d'%(step))
                    single_img_save_name = osp.join(single_img_save_dir, single_img_name)
                else:
                    single_img_name = 'step_%04d.png'%(step)
                    single_img_save_dir  = osp.join(save_dir, 'single', 'step%04d'%(step))
                    single_img_save_name = osp.join(single_img_save_dir, single_img_name)   
                mkdir_p(single_img_save_dir)   
                im.save(single_img_save_name)
        if (multi_gpus==True) and (get_rank() != 0):
            None
        else:
            print('Step: %d' % (step))


def calculate_FID_CLIP_sim(dataloader, text_encoder, netG, CLIP, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size, multi_gpus):
    """ Calculates the FID """
    clip_cos = torch.FloatTensor([0.0]).to(device)
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    n_gpu = dist.get_world_size()
    dl_length = dataloader.__len__()
    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length*times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs, captions, CLIP_tokens, sent_emb, words_embs, keys, _, _, _ = prepare_data(data, text_encoder, device)
            ######################################################
            # (2) Generate fake images
            ######################################################
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                noise = torch.randn(batch_size, z_dim).to(device)
                if (multi_gpus == True):
                    netG.module.lstm.init_hidden(noise)
                else:
                    netG.lstm.init_hidden(noise)
                #netG.lstm.init_hidden(noise)
                fake_imgs = netG(noise,sent_emb,eval=True).float()
                # norm_ip(fake_imgs, -1, 1)
                fake_imgs = torch.clamp(fake_imgs, -1., 1.)
                fake_imgs = torch.nan_to_num(fake_imgs, nan=-1.0, posinf=1.0, neginf=-1.0)
                clip_sim = calc_clip_sim(CLIP, fake_imgs, CLIP_tokens, device)
                clip_cos = clip_cos + clip_sim
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # concat pred from multi GPUs
                output = list(torch.empty_like(pred) for _ in range(n_gpu))
                dist.all_gather(output, pred)
                pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                pred_arr[start:end] = pred_all.cpu().data.numpy()
            # update loop information
            if (n_gpu!=1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                if epoch==-1:
                    loop.set_description('Evaluating]')
                else:
                    loop.set_description(f'Eval Epoch [{epoch}/{max_epoch}]')
                loop.set_postfix()
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop.close()
    # CLIP-score
    CLIP_score_gather = list(torch.empty_like(clip_cos) for _ in range(n_gpu))
    dist.all_gather(CLIP_score_gather, clip_cos)
    clip_score = torch.cat(CLIP_score_gather, dim=0).mean().item()/(dl_length*times)
    # FID
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value,clip_score


def calc_clip_sim(clip, fake, caps_clip, device):
    ''' calculate cosine similarity between fake and text features,
    '''
    # Calculate features
    fake = transf_to_CLIP_input(fake)
    fake_features = clip.encode_image(fake)
    text_features = clip.encode_text(caps_clip)
    text_img_sim = torch.cosine_similarity(fake_features, text_features).mean()
    return text_img_sim


def sample_one_batch(noise, sent, netG, multi_gpus, epoch, img_save_dir, writer):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        netG.eval()
        with torch.no_grad():
            B = noise.size(0)
            fixed_results_train = generate_samples(noise[:B//2], sent[:B//2], netG).cpu()
            torch.cuda.empty_cache()
            fixed_results_test = generate_samples(noise[B//2:], sent[B//2:], netG).cpu()
            torch.cuda.empty_cache()
            fixed_results = torch.cat((fixed_results_train, fixed_results_test), dim=0)
        img_name = 'samples_epoch_%03d.png'%(epoch)
        img_save_path = osp.join(img_save_dir, img_name)
        vutils.save_image(fixed_results.data, img_save_path, nrow=8, value_range=(-1, 1), normalize=True)


def generate_samples(noise, caption, model):
    with torch.no_grad():
        fake = model(noise, caption, eval=True)
    return fake


def predict_loss(predictor, img_feature, text_feature, negtive):
    output = predictor(img_feature, text_feature)
    err = hinge_loss(output, negtive)
    return output,err


def hinge_loss(output, negtive):
    if negtive==False:
        err = torch.mean(F.relu(1. - output))
        #err = torch.mean(F.relu(torch.ones_like(output) - output))
    else:
        err = torch.mean(F.relu(1. + output))
        #err = torch.mean(F.relu(torch.ones_like(output) + output))
    return err


def logit_loss(output, negtive):
    batch_size = output.size(0)
    real_labels = torch.FloatTensor(batch_size,1).fill_(1).to(output.device)
    fake_labels = torch.FloatTensor(batch_size,1).fill_(0).to(output.device)
    output = nn.Sigmoid()(output)
    if negtive==False:
        err = nn.BCELoss()(output, real_labels)
    else:
        err = nn.BCELoss()(output, fake_labels)
    return err


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    '''
    print('&'*20)
    print(sigma1)#, sigma1.type())
    print('&'*20)
    print(sigma2)#, sigma2.type())
    '''
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def spherical_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    # ViT-L/14
    y_size = y.size(-1)
    x_size = x.size(0)
    x = x.view(x_size, y_size, -1)
    y_expand = x.size(2)
    y = y.unsqueeze(2).contiguous()
    y = y.repeat(1, 1, y_expand)
    # ViT-L/14
    z = (x * y)
    h = z.mean(-1).sum(-1)
    q = h.arccos().pow(2)
    contained_nan = np.isnan(q.detach().cpu())
    contained_nan = contained_nan.numpy()
    if np.any(contained_nan == 1):
        q = torch.zeros(q.size()).cuda()
        print('spherical_distance: contained_nan')
    #return (x * y).sum(-1).arccos().pow(2)
    return q

# def spherical_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     x = F.normalize(x, dim=-1)
#     y = F.normalize(y, dim=-1)
#     q = (x * y).sum(-1).arccos().pow(2)
#     contained_nan = np.isnan(q.detach().cpu())
#     contained_nan = contained_nan.numpy()
#     if np.any(contained_nan == 1):
#         q = torch.zeros(q.size()).cuda()
#         print('spherical_distance: contained_nan')
#     return q
#     #return (x * y).sum(-1).arccos().pow(2)
