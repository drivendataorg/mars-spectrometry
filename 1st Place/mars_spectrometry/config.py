DATA_DIR = 'data'
TRAIN_DIR = f'{DATA_DIR}/train_features'
OUTPUT_DIR = '../../mars_spectrometry_submission_output'

NB_FOLDS = 4

CLS_LABELS = [
    'basalt', 'carbonate', 'chloride', 'iron_oxide', 'oxalate', 'oxychlorine', 'phyllosilicate', 'silicate', 'sulfate', 'sulfide'
]

NB_CLS = len(CLS_LABELS)


# list of models and checkpoints to be used for submission
MODELS = [
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
