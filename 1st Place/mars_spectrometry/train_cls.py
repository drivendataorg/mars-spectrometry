import argparse
import os
import random

import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy
import config
import seaborn as sns

import common_utils
import mars_dataset
import models_cls
from common_utils import DotDict, timeit_context
import sklearn
import sklearn.metrics

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# seed_everything(seed=42)


def build_model(cfg):
    model_params = copy.copy(cfg['model_params'])
    cls = models_cls.__dict__[model_params['model_cls']]
    del model_params['model_cls']
    del model_params['model_type']
    model: nn.Module = cls(pretrained=True, **model_params)
    return model


def train(experiment_name: str, fold: str, continue_epoch: int = -1):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)

    model_params = DotDict(cfg["model_params"])
    model_type = model_params.model_type
    train_params = DotDict(cfg["train_params"])

    checkpoints_dir = f"{config.OUTPUT_DIR}/checkpoints/{model_str}_{fold}"
    tensorboard_dir = f"{config.OUTPUT_DIR}/tensorboard/{model_type}/{model_str}_{fold}"
    oof_dir = f"{config.OUTPUT_DIR}/oof/{model_str}_{fold}"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(oof_dir, exist_ok=True)
    print("\n", experiment_name, "\n")

    logger = SummaryWriter(log_dir=tensorboard_dir)

    scaler = torch.cuda.amp.GradScaler()
    dataset_train = mars_dataset.MarsSpectrometryDataset(
        is_training=True,
        output_pytorch_tensor=True,
        fold=fold,
        **cfg['dataset_params']
    )

    dataset_valid = mars_dataset.MarsSpectrometryDataset(
        is_training=False,
        output_pytorch_tensor=True,
        fold=fold,
        **cfg['dataset_params']
    )
    dataset_valid.sub_bg_prob = 0

    batch_size = cfg['train_data_loader']['batch_size']

    data_loaders = {
        "train": DataLoader(
            dataset_train,
            num_workers=cfg['train_data_loader']['num_workers'],
            shuffle=True,
            batch_size=batch_size
        ),
        "val": DataLoader(
            dataset_valid,
            num_workers=cfg['val_data_loader']['num_workers'],
            shuffle=False,
            batch_size=cfg['val_data_loader']['batch_size']
        ),
    }

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.train()

    initial_lr = float(train_params.initial_lr)
    if train_params.optimizer == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, nesterov=True)
    else:
        raise RuntimeError("Invalid optimiser" + train_params.optimizer)

    nb_epochs = train_params.nb_epochs
    if train_params.scheduler == "steps":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=train_params.optimiser_milestones,
            gamma=0.2,
            last_epoch=continue_epoch
        )
    elif train_params.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = common_utils.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=train_params.scheduler_period,
            T_mult=train_params.get('scheduler_t_mult', 1),
            eta_min=initial_lr / 1000.0,
            last_epoch=-1,
            first_epoch_lr_scale=0.01
        )
        for i in range(continue_epoch + 1):
            scheduler.step()
    else:
        raise RuntimeError("Invalid scheduler name")

    if continue_epoch > -1:
        print(f"{checkpoints_dir}/{continue_epoch:03}.pt")
        checkpoint = torch.load(f"{checkpoints_dir}/{continue_epoch:03}.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    grad_clip_value = train_params.get("grad_clip", 8.0)
    freeze_backbone_steps = train_params.get("freeze_backbone_steps", 0)
    print(f"grad clip: {grad_clip_value} freeze_backbone_steps {freeze_backbone_steps}")
    print(f"Num training samples: {len(dataset_train)} val {len(dataset_valid)}")

    cr_cls = torch.nn.BCEWithLogitsLoss()

    for epoch_num in tqdm(range(continue_epoch + 1, nb_epochs + 1)):
        for phase in ["train", "val"]:
            model.train(phase == "train")

            epoch_loss = common_utils.AverageMeter()
            data_loader = data_loaders[phase]
            val_labels = []
            val_predictions = []

            optimizer.zero_grad()

            data_iter = tqdm(data_loader, disable=True)
            for data in data_iter:
                with torch.set_grad_enabled(phase == "train"):
                    image = data['image'].float().cuda()
                    label = data['label'].float().cuda()

                    if phase == 'train':
                        label = label / (1 + 2*train_params.labels_smooth) + train_params.labels_smooth

                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast():
                        pred = model(image)
                        loss = cr_cls(pred, label)

                    if phase == "train":
                        scaler.scale(loss).backward()

                        # Unscales the gradients of optimizer's assigned params in-place
                        scaler.unscale_(optimizer)
                        if grad_clip_value > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                        scaler.step(optimizer)
                        scaler.update()

                    if phase == "val":
                        val_labels.append(data['label'].float().numpy())
                        val_predictions.append(torch.sigmoid(pred).detach().cpu().numpy())

                    epoch_loss.update(loss.detach().item(), batch_size)

                    data_iter.set_description(
                        f"{epoch_num} {phase[0]}"
                        f" Loss {epoch_loss.avg:1.4f}"
                    )

                    del loss

            if phase == "val":
                val_labels = np.concatenate(val_labels, axis=0).astype(np.float32)
                val_predictions = np.concatenate(val_predictions, axis=0).astype(np.float32)

                if np.isnan(val_predictions).sum() > 0:
                    print("NAN predictions")
                    continue

                accuracy = ((val_labels > 0.5) == (val_predictions > 0.5)).mean()
                logger.add_scalar("accuracy", accuracy, epoch_num)

                loss_clip3 = F.binary_cross_entropy(torch.from_numpy(np.clip(val_predictions, 1e-3, 1 - 1e-3)),
                                                    torch.from_numpy(val_labels))
                logger.add_scalar("bce_clip_1e-3", loss_clip3, epoch_num)

                loss_clip2 = F.binary_cross_entropy(torch.from_numpy(np.clip(val_predictions, 1e-2, 1 - 1e-2)),
                                                    torch.from_numpy(val_labels))
                logger.add_scalar("bce_clip_1e-2", loss_clip2, epoch_num)

                if epoch_num % 50 == 0:
                    np.savez(f'{oof_dir}/{epoch_num}.npz', labels=val_labels, predictions=val_predictions)

                print(
                    f"{epoch_num} {phase[0]}"
                    f" Loss {epoch_loss.avg:1.4f}"
                    f" Acc {accuracy:1.4f}"
                    f" L c3 {loss_clip3:1.4f}"
                    f" L c2 {loss_clip2:1.4f} {model_str} {fold}"
                )

            if epoch_num > 0:
                logger.add_scalar(f"loss_{phase}", epoch_loss.avg, epoch_num)

                if phase == "train":
                    logger.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_num)
                logger.flush()

            if phase == "train":
                scheduler.step()
                train_params.save_period = 100
                if (epoch_num % train_params.save_period == 0) or (epoch_num == nb_epochs):
                    torch.save(
                        {
                            "epoch": epoch_num,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        f"{checkpoints_dir}/{epoch_num:03}.pt",
                    )


def check_importance(predictions, labels):
    mean_pred = torch.stack([p[1] for p in predictions], dim=0).mean(dim=0)

    l = float(F.binary_cross_entropy(mean_pred, labels))
    print(f'All models {l:0.4f} {len(predictions)}')

    while len(predictions) > 1:
        l_without_model = [
            float(F.binary_cross_entropy(torch.stack([p[1] for i, p in enumerate(predictions) if i != n], dim=0).mean(dim=0), labels))
            for n in range(len(predictions))
        ]
        weak_model = np.argmin(l_without_model)
        l_next = l_without_model[weak_model]
        print(f'{l_next:0.4f} {l_next-l:0.4f} {predictions[weak_model][0]:32s}  {predictions[weak_model][2]:0.3f}  {len(predictions) - 1}')

        l = l_next
        del predictions[weak_model]

    print(predictions[0][0])


def check_oof():
    models = [
        # "100_vis_tr_16_3_512_v0.5_ps8",
        "111_cls3_seresnext50",
        "113_cls3_seresnext50_sbg5_norm_m",
        "120_lstm_1024_3_v0.5",
        "130_gru",
        # "131_gru_mix0.25",
        # "132_gru_sbg5",
        # "140_vis_trans_1d",
        # "141_vis_trans_1d_v1",
        # "143_vis_trans_1d_v1_mix0.5",
        # "144_vis_trans_1d_norm_m",
        # "145_vis_trans_1d_norm_m_sbg5",
        # "150_dpn68b_v1",
        # "151_dpn68b_v1_mix0.25",
        # "160_cls3_resnet34",
        # "161_cls3_resnet34_mix0.25",
        # "162_cls3_resnet34_mix0.25_clip3",
        # "163_cls_resnet34_sbg5",
        # "164_cls_resnet34",
        # "165_cls_resnet34_mix0.25",
        # "166_cls_resnet34_mix0.25_clip3",
        # "167_cls_resnet34_norm_m",
        # "170_cls3_enet_b2",
        # "171_cls_enet_b2_mix0.25",
        # "172_cls_enet_b2_mix0.25_sbg5",
        # "173_cls_enet_b2"
    ]

    folds = [0, 1, 2, 3]
    # folds = ['sam']
    epoch = 1100

    predictions = []
    labels = np.array([])

    for model in models:
        model_labels = []
        model_predictions = []
        for fold in folds:
            oof_dir = f"{config.OUTPUT_DIR}/oof/{model}_{fold}"
            data = np.load(f'{oof_dir}/{epoch}.npz')
            model_labels.append(data['labels'].copy())
            model_predictions.append(data['predictions'].copy())
        labels = np.concatenate(model_labels, axis=0)
        model_predictions = np.concatenate(model_predictions, axis=0)
        predictions.append(model_predictions)

    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)

    df = pd.DataFrame(predictions.reshape((predictions.shape[0], -1)).T, columns=[m[:3] for m in models])
    df['mean'] = df.mean(axis=1)

    accuracy = ((labels > 0.5) == (mean_pred > 0.5)).mean()
    print('accuracy', accuracy)

    for clip in [0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]:
        loss_clip = F.binary_cross_entropy(torch.from_numpy(mean_pred * (1-clip*2) + clip),
                                           torch.from_numpy(labels))

        print(f'loss clip {clip}: {loss_clip:0.4f}')


    models_corr = []
    import scipy.stats
    for model, pred in zip(models, predictions):
        cor = scipy.stats.pearsonr(mean_pred.flatten(), pred.flatten())[0]
        models_corr.append(cor)
        print(f'{model}     {cor:0.4f}')

    print()
    clip = 2e-3
    p = [[model, torch.from_numpy(prediction * (1-clip*2) + clip), c] for model, prediction, c in zip(models, predictions, models_corr)]
    check_importance(p, torch.from_numpy(labels))

    # cor2 = np.corrcoef(predictions.reshape((predictions.shape[0], -1)))
    # print(cor2)

    # sns.heatmap(df.corr(), annot=True, cmap='BrBG')
    # plt.show()


def predict_oof(model_str, fold, epoch, nb_augmentations):
    torch.set_grad_enabled(False)

    cfg = common_utils.load_config_data(model_str)

    dataset_valid = mars_dataset.MarsSpectrometryDataset(
        is_training=False,
        output_pytorch_tensor=True,
        dataset_type='train',
        fold=fold,
        **cfg['dataset_params']
    )

    data_loader = DataLoader(
        dataset_valid,
        num_workers=4,
        shuffle=False,
        batch_size=32
    )

    model = build_model(cfg)
    # print(model.__class__.__name__)
    model = model.cuda()
    model.eval()

    fn = f"{config.OUTPUT_DIR}/models/{model_str}_{epoch}_{fold}.pth"
    print(fn)
    checkpoint = torch.load(fn)
    model.load_state_dict(checkpoint["model_state_dict"])

    fold_predictions = []
    labels = []

    for aug_idx in range(nb_augmentations):
        predictions = []

        for data in data_loader:
            label = data['label'].float().cpu().numpy()
            image = data['image'].float().cuda()

            with torch.cuda.amp.autocast():
                pred = torch.sigmoid(model(image))

            if aug_idx == 0:
                labels.append(label)
            predictions.append(pred.float().detach().cpu().numpy())

        fold_predictions.append(np.concatenate(predictions, axis=0))

    fold_predictions = np.mean(fold_predictions, axis=0)
    labels = np.concatenate(labels, axis=0)

    return fold_predictions, labels


def check_oof_tta():
    folds = [0, 1, 2, 3]
    # folds = ['sam']
    nb_augmentations = 16

    predictions = []
    labels = np.array([])

    for model, epoch in config.MODELS:
        model_labels = []
        model_predictions = []
        for fold in folds:
            fold_predictions, fold_labels = predict_oof(model, fold=fold, epoch=epoch, nb_augmentations=nb_augmentations)
            model_labels.append(fold_labels)
            model_predictions.append(fold_predictions)
        labels = np.concatenate(model_labels, axis=0)
        model_predictions = np.concatenate(model_predictions, axis=0)
        predictions.append(model_predictions)

    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)

    df = pd.DataFrame(predictions.reshape((predictions.shape[0], -1)).T, columns=[m[0][:3] for m in config.MODELS])
    df['mean'] = df.mean(axis=1)

    accuracy = ((labels > 0.5) == (mean_pred > 0.5)).mean()
    print('accuracy', accuracy)
    print('nb_augmentations', nb_augmentations)

    for clip in [0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]:
        loss_clip = F.binary_cross_entropy(torch.from_numpy(mean_pred * (1-clip*2) + clip),
                                           torch.from_numpy(labels))

        print(f'loss clip {clip}: {loss_clip:0.4f}')

    models_corr = []
    import scipy.stats
    for model, pred in zip(config.MODELS, predictions):
        cor = scipy.stats.pearsonr(mean_pred.flatten(), pred.flatten())[0]
        clip = 2e-3
        loss = F.binary_cross_entropy(torch.from_numpy(pred * (1-clip*2) + clip),
                                      torch.from_numpy(labels))
        models_corr.append(cor)
        print(f'{model[0]:32s}  {cor:0.4f}  {float(loss):0.4f}')

    print()
    clip = 2e-3
    p = [[model[0], torch.from_numpy(prediction * (1-clip*2) + clip), c] for model, prediction, c in zip(config.MODELS, predictions, models_corr)]
    check_importance(p, torch.from_numpy(labels))

    # cor2 = np.corrcoef(predictions.reshape((predictions.shape[0], -1)))
    # print(cor2)

    # sns.heatmap(df.corr(), annot=True, cmap='BrBG')
    # plt.show()



def check(experiment_name: str, fold: int, epoch: int):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)

    model_params = DotDict(cfg["model_params"])
    model_type = model_params.model_type

    checkpoints_dir = f"{config.OUTPUT_DIR}/checkpoints/{model_str}_{fold}"
    oof_dir = f"{config.OUTPUT_DIR}/oof/{model_str}"
    print("\n", experiment_name, "\n")

    cfg['dataset_params']['mz_var'] = 0

    dataset_valid = mars_dataset.MarsSpectrometryDataset(
        is_training=False,
        output_pytorch_tensor=True,
        fold=fold,
        **cfg['dataset_params']
    )

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.eval()

    data_loader = DataLoader(
        dataset_valid,
        num_workers=1,
        shuffle=False,
        batch_size=1
    )

    print(f"{checkpoints_dir}/{epoch:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    all_labels = []
    all_predictions = []

    loss_bce = torch.nn.BCELoss()

    with torch.set_grad_enabled(False):
        data_iter = tqdm(data_loader, disable=True)
        for data in data_iter:
            label = data['label'].float().cpu().numpy()[0]
            image = data['image'].float().cuda()

            with torch.cuda.amp.autocast():
                pred = torch.sigmoid(model(image))

            pred = pred[0].float()

            loss_val = float(loss_bce(pred.cpu(), data['label'].float()[0]))

            pred = pred.detach().cpu().numpy()
            print(int(data['item']), data['sample_id'], data['instrument_type'], loss_val)
            print(' '.join([f'{v:0.3f}' for v in label]))
            print(' '.join([f'{v:0.3f}' for v in pred]))
            # print(label.shape, pred.shape)
            # pred = np.argmax(pred, axis=1)
            # print(pred.shape)

            all_labels.append(label)
            all_predictions.append(pred)



def export_model(experiment_name: str, fold: int, epoch: int):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)

    checkpoints_dir = f"{config.OUTPUT_DIR}/checkpoints/{model_str}_{fold}"
    print("\n", experiment_name, "\n")

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()

    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dst_dir = f'{config.OUTPUT_DIR}/models'
    os.makedirs(dst_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict()
        },
        f'{dst_dir}/{experiment_name}_{epoch}_{fold}.pth'
    )


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default="check")
    parser.add_argument("experiment", type=str, default="")
    parser.add_argument("--fold", type=str)
    parser.add_argument("--epoch", type=int, default=-1)

    args = parser.parse_args()
    action = args.action
    experiment_name = common_utils.normalize_experiment_name(args.experiment)

    if action == "train":
        train(
            experiment_name=experiment_name,
            continue_epoch=args.epoch,
            fold=args.fold
        )

    if action == "check":
        check(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold
        )

    if action == "check_oof":
        check_oof_tta()

    if action == "export_models":
        if experiment_name == "":
            model_names = config.MODELS
        else:
            model_names = [(experiment_name, args.epoch)]

        for model_name, epoch in model_names:
            print(model_name)
            for fold in range(config.NB_FOLDS):
                export_model(
                    experiment_name=model_name,
                    epoch=epoch,
                    fold=fold
                )

