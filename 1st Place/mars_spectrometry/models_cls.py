import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import timm
import numpy as np


class SimpleCls(nn.Module):
    def __init__(self,
                 base_model_name,
                 dropout=0,
                 pretrained=True
                 ):
        super().__init__()

        nb_cls = config.NB_CLS
        self.dropout = dropout
        self.base_model = timm.create_model(base_model_name,
                                            in_chans=1,
                                            features_only=True,
                                            pretrained=pretrained)
        self.backbone_depths = list(self.base_model.feature_info.channels())
        print(f'{base_model_name}')
        self.fc = nn.Linear(self.backbone_depths[-1] * 2, nb_cls)

    def forward(self, inputs):
        output = self.base_model(inputs[:, None, :, :])
        features = output[-1]

        avg_pool = F.avg_pool2d(features, features.shape[2:])
        max_pool = F.max_pool2d(features, features.shape[2:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        x = avg_max_pool.view(avg_max_pool.size(0), -1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)
        x = self.fc(x)
        return x


import timm.models.vision_transformer


class VisTransEmbed(nn.Module):
    def __init__(self,
                 img_size=(100, 80),
                 patch_size=(1, 1),
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True
                 ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, pos1=None, pos2=None):
        B, C, H, W = x.shape
        x = self.proj(x)
        if pos1 is not None:
            x = x + pos1
        if pos2 is not None:
            x = x + pos2
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class VisTrans(nn.Module):
    def __init__(self,
                 nb_temp=80,
                 nb_mz=100,
                 input_channels=256,
                 pretrained=True,
                 patch_size=4,
                 embed_dim=512,
                 depth=6,
                 num_heads=4,
                 ):
        super().__init__()

        self.input_channels = input_channels
        self.base_model = timm.models.vision_transformer.VisionTransformer(
            img_size=(nb_mz, nb_temp), patch_size=(1, patch_size), in_chans=input_channels, num_classes=config.NB_CLS,
            embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VisTransEmbed, norm_layer=None,
            act_layer=None, weight_init=''
        )

    def forward(self, inputs):
        inputs_bins = torch.floor(inputs * (self.input_channels - 1)).long()
        one_hot_input = F.one_hot(inputs_bins, self.input_channels).permute(0, 3, 1, 2).float()
        output = self.base_model(one_hot_input)
        return output


class SimpleCls3(nn.Module):
    def __init__(self,
                 base_model_name,
                 pretrained=True,
                 have_pretrained=True,
                 ):
        super().__init__()
        self.base_model = timm.create_model(base_model_name,
                                            in_chans=3,
                                            num_classes=config.NB_CLS,
                                            pretrained=pretrained and have_pretrained)

    def forward(self, inputs):
        N, nb_mz, nb_temp = inputs.shape

        x = inputs[:, None, :, :]

        t = (torch.arange(nb_temp).float() / nb_temp).to(inputs.device)[None, None, None, :]
        mz = (torch.arange(nb_mz).float() / nb_mz).to(inputs.device)[None, None, :, None]
        x = torch.cat([x, t.repeat(N, 1, nb_mz, 1), mz.repeat(N, 1, 1, nb_temp)], dim=1)

        return self.base_model(x)


class VisTrans1D(nn.Module):
    def __init__(self,
                 nb_temp=80,
                 nb_mz=100,
                 pretrained=True,
                 embed_dim=1024,
                 depth=3,
                 num_heads=32,
                 drop_rate=0.0
                 ):
        super().__init__()

        self.base_model = timm.models.vision_transformer.VisionTransformer(
            img_size=(nb_temp, 1), patch_size=(1, 1), in_chans=nb_mz + 1, num_classes=config.NB_CLS,
            embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
            drop_rate=drop_rate, attn_drop_rate=0., drop_path_rate=0., embed_layer=VisTransEmbed, norm_layer=None,
            act_layer=None, weight_init=''
        )

    def forward(self, inputs):
        N, nb_mz, nb_temp = inputs.shape
        t = (torch.arange(nb_temp).float() / nb_temp).to(inputs.device)[None, None, :]
        x = torch.cat([inputs, t.repeat(N, 1, 1)], dim=1)

        return self.base_model(x[:, :, :, None])


class SimpleLSTM(nn.Module):
    def __init__(self,
                 hidden_dim=512,
                 n_layers=1,
                 drop_prob=0.0,
                 drop_prob_out=0.0,
                 avg_output=False,
                 use_gru=False,
                 pretrained=True,
                 ):
        super().__init__()
        self.drop_prob_out = drop_prob_out
        if use_gru:
            self.lstm = nn.GRU(101, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        else:
            self.lstm = nn.LSTM(101, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        self.avg_output = avg_output
        self.fc = nn.Linear(hidden_dim * 2, config.NB_CLS)

    def forward(self, inputs):
        N, nb_mz, nb_temp = inputs.shape
        inputs = inputs.permute(0, 2, 1)
        t = (torch.arange(nb_temp).float() / nb_temp).to(inputs.device)[None, :, None]
        x = torch.cat([inputs, t.repeat(N, 1, 1)], dim=2)

        lstm_out, hidden = self.lstm(x)
        if self.avg_output:
            x = lstm_out.mean(dim=1)
        else:
            x = lstm_out[:, -1, :]

        if self.drop_prob_out > 0:
            x = F.dropout(x, self.drop_prob_out, self.training)
        x = self.fc(x)
        return x


def print_summary():
    import pytorch_model_summary
    nb_temp = 80
    nb_mz = 100

    # model = SimpleCls(base_model_name='resnet34')
    # model = VisTrans(nb_temp=nb_temp, nb_mz=nb_mz)
    # model = SimpleLSTM(
    #     hidden_dim=512,
    #     n_layers=3,
    #     drop_prob=0.5,
    #     drop_prob_out=0.5,
    # )

    model = VisTrans1D()
    model.cuda()

    random_input = np.random.rand(4, nb_mz, nb_temp).astype(np.float32)

    res = model(torch.from_numpy(random_input).cuda())
    print(res.shape)

    print(pytorch_model_summary.summary(model, torch.zeros((4, nb_mz, nb_temp)).cuda(), max_depth=3))

    assert res.shape == (4, 10)


if __name__ == "__main__":
    print_summary()
