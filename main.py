# /usr/bin/env python3.6
import math
import re
import argparse
import warnings
import shutil
warnings.filterwarnings("ignore")
from pathlib import Path
from operator import itemgetter
from shutil import copytree, rmtree
import typing
from typing import Any, Callable, List, Tuple
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from dice3d import dice3d
from networks import weights_init
from dataloader import get_loaders
from utils import map_, save_dict_to_file, get_subj_nb
from utils import dice_coef, dice_batch, save_images, save_images_p, save_be_images, tqdm_, save_images_ent
from utils import probs2one_hot, probs2class, mask_resize, resize, get_mom_posmed, get_mom_posav, haussdorf,get_linreg_coef
from utils import soft_inertia, soft_size,soft_compactness,soft_length, saml_compactness,soft_eccentricity
from utils import exp_lr_scheduler
import datetime
from itertools import cycle
import os
from time import sleep
from bounds import CheckBounds
import matplotlib.pyplot as plt
from itertools import chain
import platform
import random
import ast
import networks 

def setup(args, n_class, dtype) -> Tuple[
    Any, Any, Any, List[Callable], List[float], List[Callable], List[float], Callable]:
    print(">>> Setting up, loading, ",args.model_weights)
    cpu: bool = args.cpu or not torch.cuda.is_available()
    if cpu:
        print("WARNING CUDA NOT AVAILABLE")
    device = torch.device("cpu") if cpu else torch.device("cuda")
    n_epoch = args.n_epoch
    if args.model_weights:
        if cpu:
            net = torch.load(args.model_weights, map_location='cpu')
        else:
            net = torch.load(args.model_weights)
    else:
        net_class = getattr(__import__('networks'), args.network)
        net = net_class(1, n_class).type(dtype).to(device)
        net.apply(weights_init)
    net.to(device)
    print(args.do_not_config_mod, "args.config_mod")
    if args.do_not_config_mod:
        print('WARNING all var updated')
    else:
        net = networks.configure_model(net)
        print('normalization statistics updated')
    if args.saveim:
        print("WARNING: Saving masks at each epc")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.l_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    if args.adamw:
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.l_rate, betas=(0.9, 0.999))
    print(args.target_losses)
    losses = eval(args.target_losses)
    loss_fns: List[Callable] = []
    for loss_name, loss_params, type_bounds, bounds_params, fn, _ in losses:
        loss_class = getattr(__import__('losses'), loss_name)
        loss_fns.append(loss_class(**loss_params, dtype=dtype, fn=fn))
        print("bounds_params", bounds_params)
        if bounds_params != None and type_bounds != 'PreciseBounds':
            bool_predexist = CheckBounds(**bounds_params)
            print("size prior file properly read : ", bool_predexist)
            if not bool_predexist:
                n_epoch = 0
    momfn = getattr(__import__('utils'), loss_params['moment_fn'])
    loss_weights = map_(itemgetter(5), losses)

    if args.scheduler:
        scheduler = getattr(__import__('scheduler'), args.scheduler)(**eval(args.scheduler_params))
    else:
        scheduler = ''

    return net, optimizer, device, loss_fns, loss_weights, scheduler, n_epoch, momfn


def do_epoch(args, mode: str, net: Any, device: Any, epc: int,
             loss_fns: List[Callable], loss_weights: List[float],
             new_w: int, C: int, metric_axis: List[int], savedir: str = "",
             optimizer: Any = None, target_loader: Any = None, best_dice3d_val: Any = None, momfn:Any = None, mom_est:Any=None, mom_coef_vec:Any=None,keep_lambda:Any=None):
    assert mode in ["train", "val"]
    L: int = len(loss_fns)
    indices = torch.tensor(metric_axis, device=device)
    desc = f">> TTA ({epc})"

    total_it_t, total_images_t = len(target_loader), len(target_loader.dataset)
    total_iteration = total_it_t
    total_images = total_images_t

    if args.debug:
        total_iteration = 10
    pho = 1
    dtype = eval(args.dtype)

    all_dices: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_sizes: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_gt_sizes: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_sizes2: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_moments: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_moments_pred: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_moments_pred0: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_moments_gt: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_moments_gt0: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_inter_card: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_card_gt: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_card_pred: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_gt = []
    all_pred = []
    if args.do_hd or args.do_asd:
        all_gt: Tensor = torch.zeros((total_images, args.wh, args.wh), dtype=dtype)
        all_pred: Tensor = torch.zeros((total_images, args.wh, args.wh), dtype=dtype)
    loss_log: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    loss_cons: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    loss_se: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    loss_tot: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    posim_log: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    haussdorf_log: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_grp: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_pnames = np.zeros([total_images]).astype('U256')
    dice_3d_log, dice_3d_sd_log = 0, 0
    hd_3d_log, asd_3d_log, hd_3d_sd_log, asd_3d_sd_log = 0, 0, 0, 0
    tq_iter = tqdm_(enumerate(target_loader), total=total_iteration, desc=desc)
    done: int = 0
    n_warmup = args.n_warmup
    mult_lw = [pho ** (epc - n_warmup + 1)] * len(loss_weights)
    mult_lw[0] = 1
    loss_weights = [a * b for a, b in zip(loss_weights, mult_lw)]
    losses_vec, target_vec, baseline_target_vec = [], [], []
    pen_count = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        count_losses = 0
        for j, target_data in tq_iter:
            target_data[1:] = [e.to(device) for e in target_data[1:]]  # Move all tensors to device
            filenames_target, target_image, target_gt = target_data[:3]
            # print(torch.unique(target_gt))
            # print("target", filenames_target)
            labels = target_data[3:3 + L]
            bounds = target_data[3 + L:]
            filenames_target = [f.split('.nii')[0] for f in filenames_target]
            assert len(labels) == len(bounds), len(bounds)
            B = len(target_image)
            # Reset gradients
            if optimizer:
                optimizer.zero_grad()

            # Forward
            pred_logits = net(target_image)
            pred_probs: Tensor = F.softmax(pred_logits/args.softmax_temp, dim=1)
            if new_w > 0:
                pred_probs = resize(pred_probs, new_w)
                labels = [resize(label, new_w) for label in labels]
                target = resize(target, new_w)
            predicted_mask: Tensor = probs2one_hot(pred_probs)  # Used only for dice computation
            #print(filenames_target,"inertia_gt", inertia_gt,"inertia_pred", inertia_pred,"inertia_probs",
            #      inertia_probs,"gt size",size_gt,"pred size",size_pred)
            # print(torch.unique(predicted_mask))
            assert len(bounds) == len(loss_fns) == len(loss_weights)
            #if epc < n_warmup:
            #    loss_weights = [0] * len(loss_weights)
            loss: Tensor = torch.zeros(1, requires_grad=True).to(device)
            loss_vec = []
            loss_kw = []
            #print(len(loss_fns), len(labels), len(loss_weights), len(bounds))
            for i, (loss_fn, label, w, bound) in enumerate(zip(loss_fns, labels, loss_weights, bounds)):
                if "EntKLProp" in eval(args.target_losses)[i][0]:
                    if epc > 0 and args.update_mom_est and i==0: #i=0 to prevent wrong update if two constraints, only update firsst
                        loss_fn.mom_est = mom_est
                    if epc > 0 and args.update_lin_reg:
                        if args.ind_mom == 1:
                            loss_fn.reg, loss_fn.reg2 = mom_coef_vec
                        else:
                            loss_fn.reg = mom_coef_vec
                    loss_1, loss_cons_prior, est_prop = loss_fn(pred_probs, label, bound, epc)
                    if epc == 0 and args.adw:
                        first_loss = loss_cons_prior.detach()
                        keep_lambda = 1/first_loss
                    loss_kw.append(loss_1.detach())
                    if args.n_warmup > 0 and epc < n_warmup:
                        loss = loss_1
                    else:
                        loss = loss_1 + keep_lambda*loss_cons_prior
                        loss_kw.append(keep_lambda*loss_cons_prior.detach())
                else:
                    loss = loss_fn(pred_probs, label, bound)
                    loss = w * loss
                    loss_1 = loss
                    loss_kw.append(loss_1.detach())
            if optimizer:
                loss.backward()
                optimizer.step()
            dices, inter_card, card_gt, card_pred = dice_coef(predicted_mask.detach(), target_gt.detach())
            assert dices.shape == (B, C), (dices.shape, B, C)
            sm_slice = slice(done, done + B)  # Values only for current batch
            all_dices[sm_slice, ...] = dices
            if eval(args.target_losses)[0][0] in ["EntKLProp"]:
                all_sizes[sm_slice, ...] = torch.round(
                    est_prop.detach() * target_image.shape[2] * target_image.shape[3])
            all_sizes2[sm_slice, ...] = torch.sum(predicted_mask, dim=(2, 3))
            all_moments[sm_slice, ...] = momfn(pred_probs.detach())[:,:,args.ind_mom]
            all_moments_pred0[sm_slice, ...] = momfn(predicted_mask.type(torch.float32))[:, :, 0]
            all_moments_pred[sm_slice, ...] = momfn(predicted_mask.type(torch.float32))[:, :, args.ind_mom]
            all_moments_gt[sm_slice, ...] = momfn(target_gt.type(torch.float32))[:, :, args.ind_mom]
            all_moments_gt0[sm_slice, ...] = momfn(target_gt.type(torch.float32))[:, :, 0]
            all_gt_sizes[sm_slice, ...] = torch.sum(target_gt, dim=(2, 3))
            all_grp[sm_slice, ...] = torch.FloatTensor(get_subj_nb(filenames_target)).unsqueeze(1).repeat(1, C)
            all_pnames[sm_slice] = filenames_target
            all_inter_card[sm_slice, ...] = inter_card
            all_card_gt[sm_slice, ...] = card_gt
            all_card_pred[sm_slice, ...] = card_pred
            if args.do_hd or args.do_asd:
                all_pred[sm_slice, ...] = probs2class(predicted_mask[:, :, :, :]).cpu().detach()
                all_gt[sm_slice, ...] = probs2class(target_gt).detach()
            loss_se[sm_slice] = loss_kw[0]
            if len(loss_kw) > 1:
                loss_cons[sm_slice] = loss_kw[1]
                #loss_tot[sm_slice] = np.sum(loss_kw)
                loss_tot[sm_slice] = torch.stack(loss_kw, dim=0).sum(dim=0)
            else:
                loss_cons[sm_slice] = 0
                loss_tot[sm_slice] = loss_kw[0]
            # # Save images
            if savedir and args.saveim and mode == "val":
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.simplefilter("ignore")
                    predicted_class: Tensor = probs2class(pred_probs)
                    save_images(predicted_class, filenames_target, savedir, mode, epc, False)
                    if args.entmap:
                        ent_map = torch.einsum("bcwh,bcwh->bwh", [-pred_probs, (pred_probs + 1e-10).log()])
                        save_images_ent(ent_map.detach(), filenames_target, savedir, 'ent_map', epc)

            # Logging
            big_slice = slice(0, done + B)  # Value for current and previous batches
            stat_dict = {**{f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis},
                         **{f"SZ{n}": all_sizes[big_slice, n].mean() for n in metric_axis}}

            size_dict = {**{f"SZ{n}": all_sizes[big_slice, n].mean() for n in metric_axis}}
            nice_dict = {k: f"{v:.4f}" for (k, v) in stat_dict.items()}
            done += B
            tq_iter.set_postfix(nice_dict)
    if args.dice_3d and (mode == 'val'):
        dice_3d_log, dice_3d_sd_log, asd_3d_log, asd_3d_sd_log, hd_3d_log, hd_3d_sd_log = dice3d(all_grp,
                                                                                                 all_inter_card,
                                                                                                 all_card_gt,
                                                                                                 all_card_pred,
                                                                                                 all_pred, all_gt,
                                                                                                 all_pnames,
                                                                                                 metric_axis,
                                                                                                 args.pprint,
                                                                                                 args.do_hd,
                                                                                                 args.do_asd,
                                                                                                 best_dice3d_val,savedir)
    dice_2d = torch.index_select(all_dices, 1, indices).mean().cpu().numpy().item()
    target_vec = [dice_3d_log, dice_3d_sd_log, asd_3d_log, asd_3d_sd_log, hd_3d_log, hd_3d_sd_log, dice_2d]
    size_mean = torch.index_select(all_sizes2, 1, indices).mean(dim=0).cpu().numpy()
    #mom_med = all_moments_pred.mean(dim=0).cpu().numpy().tolist()
    #mom_med0 = all_moments_pred0.mean(dim=0).cpu().numpy().tolist()
    if args.med:
        mom_est = get_mom_posmed(C, all_moments_pred, all_sizes2, args.th)
        mom_est0 = get_mom_posmed(C, all_moments_pred0, all_sizes2, args.th)
    else:
        mom_est = get_mom_posav(C, all_moments_pred, all_sizes2, args.th)
        #mom_med_gt = get_mom_posav(C, all_moments_gt, all_sizes2, args.th)
        #mom_med_gt0 = get_mom_posav(C, all_moments_gt0, all_sizes2, args.th)
        #print(mom_med_gt0,mom_med_gt)
        mom_est0 = get_mom_posav(C, all_moments_pred0, all_sizes2, args.th)
    if args.update_lin_reg:
        mom_coef = get_linreg_coef(C, all_moments_pred, all_sizes2, args.th)
        mom_coef0 = get_linreg_coef(C, all_moments_pred0, all_sizes2, args.th)
        #print(mom_coef)
    size_gt_mean = torch.index_select(all_gt_sizes, 1, indices).mean(dim=0).cpu().numpy()
    mask_pos = torch.index_select(all_sizes2, 1, indices) != 0
    gt_pos = torch.index_select(all_gt_sizes, 1, indices) != 0
    size_mean_pos = torch.index_select(all_sizes2, 1, indices).sum(dim=0).cpu().numpy() / mask_pos.sum(
        dim=0).cpu().numpy()
    gt_size_mean_pos = torch.index_select(all_gt_sizes, 1, indices).sum(dim=0).cpu().numpy() / gt_pos.sum(
        dim=0).cpu().numpy()
    size_mean2 = torch.index_select(all_sizes2, 1, indices).mean(dim=0).cpu().numpy()
    losses_vec = [loss_se.mean().item(), loss_cons.mean().item(), loss_tot.mean().item(), size_mean.mean(),
                  size_mean_pos.mean(), size_gt_mean.mean(), gt_size_mean_pos.mean()]
    if args.ind_mom == 1:
        mom_vec = [mom_est0, mom_est]
    else:
        mom_vec = mom_est0
    if not args.update_lin_reg:
        mom_coef_vec = []
    else:
        if args.ind_mom == 1:
            mom_coef_vec = [mom_coef0, mom_coef]
        else:
            mom_coef_vec = mom_coef0
    if not epc % 50 :
        #print(all_pnames.shape,mom_est0,np.repeat(str(mom_est0),len(all_pnames)).transpose().shape)
        df_t = pd.DataFrame({
            "val_ids": all_pnames,
            "pred_size": all_sizes2.cpu()})
        df_t.to_csv(Path(savedir, args.train_grp_regex + mode + str(epc) + "sizes.csv"),mode='a', float_format="%.4f", index_label="epoch")
        df_t = pd.DataFrame({
            "val_ids": all_pnames,
            "gt_moment": all_moments_gt.cpu(),
            "est_moment_0": np.repeat(str(mom_est0),len(all_pnames)).transpose(),
            "est_moment_1": np.repeat(str(mom_est),len(all_pnames)),
            "proposal_moment": all_moments.cpu(),
            "pred_moment": all_moments_pred.cpu()})
        df_t.to_csv(Path(savedir, args.train_grp_regex + mode + str(epc) + "ind_mom"+str(args.ind_mom)+ "moment.csv"), mode='a',float_format="%.4f", index_label="epoch")
    return losses_vec, target_vec, mom_vec, mom_coef_vec, keep_lambda


def run_subj(args, net, optimizer, device, loss_fns, loss_weights, scheduler, n_epoch, momfn):
    dtype = eval(args.dtype)
    shuffle = True
    metric_axis: List = args.metric_axis
    lr: float = args.l_rate
    savedir: str = args.workdir
    if args.oneslice:
        n_epoch = 100
    else:
        n_epoch: int = args.n_epoch
    n_class = args.n_class
    subj = args.train_grp_regex
    if "prostate" in savedir:
        if args.thl == "low":
            #args.th = [[0, 100], [384 * 384, 12000]]
            args.th = [[0, 100], [384 * 384, 384 * 384]]
        else:
            args.th = [[0,4000],[384*384,12000]]
    else:
        if args.thl == "low":
            args.th = [[0, 100, 100, 100, 100], [256 * 256, 256 * 256, 256 * 256.0, 256 * 256, 256 * 256]]
        else:
            # args.th = [[0,2214.0, 2498.0, 2243.5, 1853.0],[256*256,6642.0, 7494.0, 6730.5, 5559.0]]
            args.th = [[0,2214.0, 1498.0, 1243.5, 853.0], [256*256,7642.0, 8494.0, 7730.5, 6559.0]]

    target_loader, target_loader_val = get_loaders(args, args.target_dataset, args.target_folders,
                                                   args.batch_size, args.n_class,
                                                   args.debug, args.in_memory, dtype, shuffle, "target",
                                                   args.val_target_folders)

    print("metric axis", metric_axis)
    best_dice_pos: Tensor = np.zeros(1)
    best_dice: Tensor = np.zeros(1)
    best_hd3d_dice: Tensor = np.zeros(1)
    best_3d_dice: Tensor = 0
    last_3d_dice: Tensor = 0
    best_3d_asd: Tensor = 10000
    last_3d_asd: Tensor = 10000
    print("Results saved in ", savedir)
    print(">>> Starting the training")
    df = open(subj, 'w')
    mom_est = []
    mom_coef_vec = []
    keep_lambda = 1
    errors = []
    for i in range(n_epoch):

        if args.mode == "makeim":
            with torch.no_grad():

                val_losses_vec, val_target_vec, mom_est, mom_coef_vec, keep_lambda = do_epoch(args, "val", net, device,
                                                          i, loss_fns,
                                                          loss_weights,
                                                          args.resize,
                                                          n_class, metric_axis,
                                                          savedir=savedir,
                                                          target_loader=target_loader,
                                                          best_dice3d_val=best_3d_dice, momfn=momfn, mom_est= mom_est, mom_coef_vec=mom_coef_vec,keep_lambda=keep_lambda)
                #tra_losses_vec = val_losses_vec
                #tra_target_vec = val_target_vec
        else:
            val_losses_vec, val_target_vec, mom_est, mom_coef_vec, keep_lambda = do_epoch(args, "val", net, device,
                                                      i, loss_fns,
                                                      loss_weights,
                                                      args.resize,
                                                      n_class, metric_axis,
                                                      savedir=savedir,
                                                      optimizer=optimizer,
                                                      target_loader=target_loader,
                                                      best_dice3d_val=best_3d_dice,momfn=momfn, mom_est=mom_est,mom_coef_vec=mom_coef_vec,keep_lambda=keep_lambda)

            #tra_losses_vec = val_losses_vec
            #tra_target_vec = val_target_vec
        current_val_target_3d_dice = val_target_vec[0]
        current_val_target_3d_asd = val_target_vec[2]
        if args.dice_3d:
            if current_val_target_3d_dice > best_3d_dice:
                best_3d_dice = current_val_target_3d_dice
                with open(Path(savedir, subj + "3dbestepoch.txt"), 'w') as f:
                    f.write(str(i) + ',' + str(best_3d_dice))
                best_folder_3d = Path(savedir, subj + "best_epoch_3d")
                if best_folder_3d.exists():
                    rmtree(best_folder_3d)
                if args.saveim:
                    try:
                        copytree(Path(savedir, f"iter{i:03d}"), Path(best_folder_3d))
                    except shutil.Error as err:
                        errors.extend(err.args[0])

            if not args.oneslice:
                torch.save(net, Path(savedir, subj + "best_3d.pkl"))

            if current_val_target_3d_asd < best_3d_asd:
                best_3d_asd = current_val_target_3d_asd
                with open(Path(savedir, subj + "asd3dbestepoch.txt"), 'w') as f:
                    f.write(str(i) + ',' + str(best_3d_asd))
                best_folder_asd_3d = Path(savedir, subj + "best_epoch_asd")
                if best_folder_asd_3d.exists():
                    rmtree(best_folder_asd_3d)
                if args.saveim:
                    try:
                        copytree(Path(savedir, f"iter{i:03d}"), Path(best_folder_asd_3d))
                    except shutil.Error as err:
                        errors.extend(err.args[0])


        if not (i % 10) and not args.oneslice:
            print(subj, "epoch", str(i), savedir, 'best 3d dice', best_3d_dice, "mom_est", mom_est)
            if args.update_lin_reg:
                print(subj, "epoch", str(i), savedir, 'best 3d dice', best_3d_dice, "first mom_coef", mom_coef_vec[0])

        if i == n_epoch - 1 and not args.oneslice:
            last_3d_dice = val_target_vec[0]
            last_3d_asd = val_target_vec[2]
            with open(Path(savedir, subj + "last_epoch.txt"), 'w') as f:
                f.write(str(i)+',' + str(last_3d_dice) + ',' + str(last_3d_asd))
            last_folder = Path(savedir, subj + "last_epoch")
            if last_folder.exists():
                rmtree(last_folder)
            if args.saveim:
                try:
                    copytree(Path(savedir, f"iter{i:03d}"), Path(last_folder))
                except shutil.Error as err:
                    errors.extend(err.args[0])
            torch.save(net, Path(savedir, subj + "last.pkl"))

        # remove images from iteration
        if args.saveim:
            rmtree(Path(savedir, f"iter{i:03d}"))
        if not args.oneslice:
            df_t_tmp = pd.DataFrame({
                "epoch": i,
                "val_loss_s": [val_losses_vec[0]],
                "val_loss_cons": [val_losses_vec[1]],
                "val_loss_tot": [val_losses_vec[2]],
                "val_dice_3d_sd": [val_target_vec[1]],
                "val_size_mean": [np.int(val_losses_vec[3])],
                "val_gt_size_mean": [np.int(val_losses_vec[5])],
                #"val_size_mean_pos": [np.int(val_losses_vec[4])],
                "val_gt_size_mean_pos": [np.int(val_losses_vec[6])],
                'val_asd_sd': [val_target_vec[3]],
                'val_hd': [val_target_vec[4]],
                'val_hd_sd': [val_target_vec[5]],
                'val_dice': [val_target_vec[6]],
                'val_asd': [val_target_vec[2]],
                "val_dice_3d": [val_target_vec[0]]})

            if i == 0:
                df_t = df_t_tmp
            else:
                df_t = df_t.append(df_t_tmp)

            df_t.to_csv(Path(savedir, "_".join((args.target_folders.split("'")[1], subj, args.csv))),
                        float_format="%.4f", index=False)

        if args.flr == False:
            exp_lr_scheduler(optimizer, i, args.lr_decay, args.lr_decay_epoch)
    df.close()
    print(subj, " Results saved in ", savedir, "best 3d dice", best_3d_dice, "best asd 3d", best_3d_asd)
    print(subj, "last 3d dice", last_3d_dice, "last asd 3d", last_3d_asd)


def run(args: argparse.Namespace) -> None:
    savedir: str = args.workdir
    if args.dic_params!="":
        c=open(args.dic_params,'r').read()
        dic_params = ast.literal_eval(c)
        for e in dic_params.keys():
            setattr(args, e, dic_params[e])
        args.workdir = savedir
    # save args to dict
    d = vars(args)

    d['time'] = str(datetime.datetime.now())
    d['server'] = platform.node()

    save_dict_to_file(d, args.workdir, args.mode)

    n_class: int = args.n_class
    dtype = eval(args.dtype)

    # Proper params

    print(args.target_folders)
    subj_list = eval(args.regex_list)
    model_weights_dir = args.model_weights
    for subj in subj_list:
        if not args.global_model:
            subj_model_weights = str(Path(model_weights_dir,subj+"last.pkl"))
            args.model_weights = subj_model_weights
        net, optimizer, device, loss_fns, loss_weights, scheduler, n_epoch, momfn = setup(args, n_class, dtype)
        if args.oneslice:
            a=list(range(0,256))
            random.shuffle(a)
            for i in a:
                args.train_grp_regex = subj + "_" + str(i)+".nii"
                args.grp_regex = subj + "_" + str(i)+".nii"
                run_subj(args, net, optimizer, device, loss_fns, loss_weights, scheduler, n_epoch, momfn)
        else:
            args.train_grp_regex = subj
            args.grp_regex = subj
            run_subj(args, net, optimizer, device, loss_fns, loss_weights, scheduler, n_epoch, momfn)

    dsc, asd, last_dsc, last_asd = [], [], [], []
    for subj in subj_list:
        if args.oneslice:
            list_ = map_(lambda p: str(p.name), Path(savedir).glob("*.nii3dbestepoch.txt"))
            for l in list_ :
                with open(Path(savedir,l)) as f:
                    lines = f.readlines()
                    tmp_dsc = eval(lines[0])[1]
                    if tmp_dsc !=1:
                        dsc.append(eval(lines[0])[1])
        else:
            with open(Path(savedir, subj + "3dbestepoch.txt")) as f:
                lines = f.readlines()
                dsc.append(eval(lines[0])[1])
            with open(Path(savedir, subj + "asd3dbestepoch.txt")) as f:
                lines = f.readlines()
                asd.append(eval(lines[0])[1])
            with open(Path(savedir, subj + "last_epoch.txt")) as f:
                lines = f.readlines()
                last_dsc.append(eval(lines[0])[1])
                last_asd.append(eval(lines[0])[2])

    print("last_dsc",last_dsc)
            # print(x, dsc)
    print("Mean best 3d dice all subj", np.round(np.round(np.mean(dsc * 100), 3) * 100, 3))
    print("Mean last 3d dice all subj", np.round(np.round(np.mean(last_dsc * 100), 3) * 100, 3))
    print("Mean best 3d asd all subj", np.round(np.mean(asd), 3))
    print("Mean last 3d asd all subj", np.round(np.mean(last_asd), 3))

    with open(Path(savedir, "3dbestepoch.txt"), 'w') as f:
        f.write(str(np.round(np.round(np.mean(dsc * 100), 3) * 100, 3)))

    with open(Path(savedir, "3dlast_epoch.txt"), 'w') as f:
        f.write(str(np.round(np.round(np.mean(last_dsc * 100), 3) * 100, 3)))

    with open(Path(savedir, "asdlast_epoch.txt"), 'w') as f:
        f.write(str(np.round(np.mean(last_asd), 3)))

    with open(Path(savedir, "asd3dbestepoch.txt"), 'w') as f:
        f.write(str(np.round(np.mean(asd), 3)))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--target_dataset', type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--target_losses", type=str, required=True,
                        help="List of (loss_name, loss_params, bounds_name, bounds_params, fn, weight)")
    parser.add_argument("--target_folders", type=str, required=True,
                        help="List of (subfolder, transform, is_hot)")
    parser.add_argument("--val_target_folders", type=str, required=True,
                        help="List of (subfolder, transform, is_hot)")
    parser.add_argument("--network", type=str, required=True, help="The network to use")
    parser.add_argument("--grp_regex", type=str, required=True)
    parser.add_argument("--regex_list", type=str, required=True)
    parser.add_argument("--train_grp_regex", type=str, required=True)
    parser.add_argument("--n_class", type=int, required=True)
    parser.add_argument("--dic_params", type=str,default="")
    parser.add_argument("--mode", type=str, default="learn")
    parser.add_argument("--lin_aug_w", action="store_true")
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--trainval", action="store_true")
    parser.add_argument("--valonly", action="store_true")
    parser.add_argument("--flr", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--do_hd", type=bool, default=False)
    parser.add_argument("--global_model", action="store_true")
    parser.add_argument("--do_asd", type=bool, default=False)
    parser.add_argument("--saveim", type=bool, default=False)
    parser.add_argument("--thl", type=str, default='med')
    parser.add_argument("--do_not_config_mod", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--med", action="store_true")
    parser.add_argument("--csv", type=str, default='metrics.csv')
    parser.add_argument("--adamw", action="store_true")
    parser.add_argument("--dice_3d", action="store_true")
    parser.add_argument("--ontest", action="store_true")
    parser.add_argument("--adw", action="store_true")
    parser.add_argument("--oneslice", action="store_true")
    parser.add_argument("--testonly", action="store_true")
    parser.add_argument("--trainonly", action="store_true")
    parser.add_argument("--ontrain", action="store_true")
    parser.add_argument("--pprint", action="store_true")
    parser.add_argument("--entmap", action="store_true")
    parser.add_argument("--update_mom_est", action="store_true")
    parser.add_argument("--update_lin_reg", action="store_true")
    parser.add_argument("--model_weights", type=str, default='')
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--in_memory", action='store_true')
    parser.add_argument("--tta", action='store_true')
    parser.add_argument("--resize", type=int, default=0)
    parser.add_argument("--pho", nargs='?', type=float, default=1,
                        help='augment')
    parser.add_argument("--n_warmup", type=int, default=0)
    parser.add_argument("--wh", type=int, default=256)
    parser.add_argument('--n_epoch', nargs='?', type=int, default=200,
                        help='# of the epochs')
    parser.add_argument('--ind_mom', type=int, default=0,
                        help='index of the moment')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4,
                        help='Learning Rate')
    parser.add_argument('--lr_decay', nargs='?', type=float, default=0.7),
    parser.add_argument('--lr_decay_epoch', nargs='?', type=float, default=20),
    parser.add_argument('--weight_decay', nargs='?', type=float, default=1e-5,
                        help='L2 regularisation of network weights')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--dtype", type=str, default="torch.float32")
    parser.add_argument("--scheduler", type=str, default="DummyScheduler")
    parser.add_argument("--scheduler_params", type=str, default="{}")
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--softmax_temp", type=float, default=1)
    parser.add_argument("--train_case_nb", type=int, default=-1)
    parser.add_argument("--metric_axis", type=int, nargs='*', required=True, help="Classes to display metrics. \
        Display only the average of everything if empty")
    args = parser.parse_args()
    print(args)

    return args


if __name__ == '__main__':
    run(get_args())
