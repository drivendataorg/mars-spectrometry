import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import common_utils
import config
import mars_dataset
from train_cls import build_model


def predict(models, folds, dataset_type, nb_augmentations):
    torch.set_grad_enabled(False)

    all_predictions = []
    all_labels = []
    sample_ids = []
    instrument_types = []

    for model_str, epoch in tqdm(models):
        cfg = common_utils.load_config_data(model_str)

        dataset_valid = mars_dataset.MarsSpectrometryDataset(
            is_training=False,
            output_pytorch_tensor=True,
            dataset_type=dataset_type,
            fold=folds[0],
            **cfg['dataset_params']
        )

        data_loader = DataLoader(
            dataset_valid,
            num_workers=4,
            shuffle=False,
            batch_size=32
        )

        for fold in folds:
            print(model_str, fold)

            model = build_model(cfg)
            print(model.__class__.__name__)
            model = model.cuda()
            model.eval()

            checkpoints_dir = f"{config.OUTPUT_DIR}/models"
            print(f"{checkpoints_dir}/{epoch:03}.pt")
            checkpoint = torch.load(f"{checkpoints_dir}/{model_str}_{epoch}_{fold}.pth")
            model.load_state_dict(checkpoint["model_state_dict"])

            for _ in range(nb_augmentations):
                labels = []
                predictions = []
                sample_ids = []
                instrument_types = []

                for data in data_loader:
                    label = data['label'].float().cpu().numpy()
                    image = data['image'].float().cuda()
                    for sample_id in data['sample_id']:
                        sample_ids.append(str(sample_id))

                    for instrument_type in data['instrument_type']:
                        instrument_types.append(str(instrument_type))

                    with torch.cuda.amp.autocast():
                        pred = torch.sigmoid(model(image))

                    labels.append(label)
                    predictions.append(pred.float().detach().cpu().numpy())

                all_labels = np.concatenate(labels, axis=0)
                all_predictions.append(np.concatenate(predictions, axis=0))

    mean_pred = np.mean(all_predictions, axis=0)
    labels = all_labels

    return mean_pred, labels, sample_ids, instrument_types


def check_oof():

    models = [
        # "100_vis_tr_16_3_512_v0.5_ps8",
        # "110_cls_seresnext50",
        # "111_cls3_seresnext50",
        # "112_cls3_seresnext50_adam",
        # "120_lstm_1024_3_v0.5",
        # "130_gru",
        # "131_gru_mix0.25",
        "132_gru_sbg5",
        # "140_vis_trans_1d",
        # "141_vis_trans_1d_v1",
        # "142_vis_trans_1d_mix0.25",
        # "143_vis_trans_1d_v1_mix0.5",
        # "144_vis_trans_1d_norm_m",
        # "150_dpn68b_v1",
        # "151_dpn68b_v1_mix0.25",
        # "160_cls3_resnet34",
        # "161_cls3_resnet34_mix0.25",
        # "162_cls3_resnet34_mix0.25_clip3",
        "163_cls_resnet34_sbg5",
        # "164_cls_resnet34",
        # "165_cls_resnet34_mix0.25",
        # "166_cls_resnet34_mix0.25_clip3",
        # "167_cls_resnet34_norm_m",
        # "168_cls_resnet34_adam",
        # "170_cls3_enet_b2",
        # "171_cls_enet_b2_mix0.25",
        "172_cls_enet_b2_mix0.25_sbg5",
        # "173_cls_enet_b2",
    ]

    all_predictions = []
    all_labels = []
    folds = [0, 1, 2, 3]
    epoch = 1100

    for fold in folds:
        predictions, labels, sample_ids, instrument_types = predict(
            models=[(m, epoch) for m in models],
            folds=[fold],
            dataset_type='train',
            nb_augmentations=4
        )
        all_predictions.append(predictions)
        all_labels.append(labels)

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    accuracy = ((all_labels > 0.5) == (all_predictions > 0.5)).mean()
    print('accuracy', accuracy)

    for clip in [0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]:
        loss_clip = F.binary_cross_entropy(torch.from_numpy(all_predictions * (1-clip*2) + clip),
                                           torch.from_numpy(all_labels))
        print(f'loss clip {clip}: {loss_clip:0.4f}')


def prepare_submission_1():
    models = [
        ("100_vis_tr_16_3_512_v0.5_ps8", 1100),
        ("111_cls3_seresnext50", 1100),
        ("113_cls3_seresnext50_sbg5_norm_m", 1500),
        ("120_lstm_1024_3_v0.5", 1100),
        ("130_gru", 1100),
        ("131_gru_mix0.25", 1100),
        ("132_gru_sbg5", 1100),
        ("140_vis_trans_1d", 1100),
        ("141_vis_trans_1d_v1", 1100),
        ("143_vis_trans_1d_v1_mix0.5", 1100),
        ("144_vis_trans_1d_norm_m", 1100),
        ("145_vis_trans_1d_norm_m_sbg5", 1100),
        ("150_dpn68b_v1", 1100),
        ("151_dpn68b_v1_mix0.25", 1100),
        ("160_cls3_resnet34", 1100),
        ("161_cls3_resnet34_mix0.25", 1100),
        ("162_cls3_resnet34_mix0.25_clip3", 1100),
        ("163_cls_resnet34_sbg5", 1100),
        ("164_cls_resnet34", 1100),
        ("165_cls_resnet34_mix0.25", 1100),
        ("166_cls_resnet34_mix0.25_clip3", 1100),
        ("167_cls_resnet34_norm_m", 1100),
        ("170_cls3_enet_b2", 1100),
        ("171_cls_enet_b2_mix0.25", 1100),
        ("172_cls_enet_b2_mix0.25_sbg5", 1100),
        ("173_cls_enet_b2", 1100),
    ]

    predictions, labels, sample_ids, instrument_types = predict(
        models=models,
        folds=[0, 1, 2, 3],
        dataset_type='test_val',
        nb_augmentations=16
    )
    commercial_mask = np.array([t == 'commercial' for t in instrument_types])

    clip_commercial = 0.002
    predictions[commercial_mask] = predictions[commercial_mask] * (1 - clip_commercial * 2) + clip_commercial

    clip_sam = 0.01
    predictions[~commercial_mask] = predictions[~commercial_mask] * (1 - clip_sam * 2) + clip_sam

    os.makedirs('submissions', exist_ok=True)
    df = pd.DataFrame({'sample_id': sample_ids})
    for i, col in enumerate(config.CLS_LABELS):
        df[col] = predictions[:, i]
    df.to_csv('submissions/sub_01.csv', index=False, float_format='%.6f')


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default="check")

    args = parser.parse_args()
    action = args.action

    if action == "check_oof":
        check_oof()

    if action == "prepare_submission":
        prepare_submission_1()
