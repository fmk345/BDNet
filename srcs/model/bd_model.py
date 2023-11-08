from srcs.model.bd_modules import Conv, Deconv, ResBlock, MLP, TransformerBlock, CUnet
import torch.nn as nn
import torch
import torch.nn.functional as F
from srcs.model.bd_utils import PositionalEncoding
from einops import rearrange


# ---------- Our network -----------------------

class BDNeRV_RC(nn.Module):
    # recursive frame reconstruction
    def __init__(self):
        super(BDNeRV_RC, self).__init__()
        # params
        n_colors = 1
        n_resblock = 4
        n_feats = 32
        kernel_size = 3
        padding = 1

        pos_b, pos_l = 1.25, 80  # position encoding params
        mlp_dim_list = [2*pos_l, 512, n_feats*4*2] # (160, 512, 256)
        mlp_act = 'gelu'

        # main body
        self.mainbody = CUnet(n_feats=n_feats, n_resblock=n_resblock,
                              kernel_size=kernel_size, padding=padding)

        # output block
        OutBlock = [ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    Conv(input_channels=n_feats, n_feats=n_colors, kernel_size=kernel_size, padding=padding)]
        self.out = nn.Sequential(*OutBlock)

        # feature block
        FeatureBlock = [Conv(input_channels=n_colors, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.feature = nn.Sequential(*FeatureBlock)

        # concatenation fusion block
        CatFusion = [Conv(input_channels=n_feats*2, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.catfusion = nn.Sequential(*CatFusion)

        # position encoding
        self.pe_t = PositionalEncoding(pe_embed_b=pos_b, pe_embed_l=pos_l)

        # mlp
        self.embed_mlp = MLP(dim_list=mlp_dim_list, act=mlp_act)

    def forward(self, ce_blur, time_idx, ce_code):
        # time index: [frame_num,1]
        # t_embed
        t_pe_ = [self.pe_t(idx)*(2*code-1)
                 for idx, code in zip(time_idx, ce_code)]  # [frame_num*[pos_l*2,1]]
        t_pe = torch.cat(t_pe_, dim=0)  # [frame_num, pos_l*2]
        t_embed = self.embed_mlp(t_pe)  # [frame_num, n_feats*4*2]
        # t_manip = self.manip_mlp(t_pe)

        # ce_blur feature
        ce_feature = self.feature(ce_blur)  # [b, c, h, w]

        # main body
        output_list = []
        for k in range(len(time_idx)):
            if k==0:
                main_feature = ce_feature
            else:
                # since k=2, cat pre-feature with ce_feature as input feature
                cat_feature = torch.cat((feat_out_k, ce_feature),dim=1)
                main_feature = self.catfusion(cat_feature)
            feat_out_k = self.mainbody(main_feature, t_embed[k])
            output_k = self.out(feat_out_k)
            output_list.append(output_k)

        output = torch.stack(output_list, dim=1)

        return output

# --------Ablation------------
class BDNeRV(nn.Module):
    # frame extraction without self-recursive strategy
    def __init__(self):
        super(BDNeRV, self).__init__()
        # params
        n_colors = 3
        n_resblock = 4
        n_feats = 32
        kernel_size = 3
        padding = 1

        pos_b, pos_l = 1.25, 80  # position encoding params
        mlp_dim_list = [2*pos_l, 512, 256]
        mlp_act = 'gelu'

        # main body
        self.mainbody = CUnet(n_feats=n_feats, n_resblock=n_resblock,kernel_size=kernel_size, padding=padding)

        # output block
        OutBlock = [ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    Conv(input_channels=n_feats, n_feats=n_colors, kernel_size=kernel_size, padding=padding)]
        self.out = nn.Sequential(*OutBlock)

        # feature block
        FeatureBlock = [Conv(input_channels=n_colors, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.feature = nn.Sequential(*FeatureBlock)

        # position encoding
        self.pe_t = PositionalEncoding(pe_embed_b=pos_b, pe_embed_l=pos_l)

        # mlp
        self.embed_mlp = MLP(dim_list=mlp_dim_list, act=mlp_act)



    def forward(self, ce_blur, time_idx, ce_code):
        # time index: [frame_num,1]
        # t_embed
        t_pe_ = [self.pe_t(idx)*(2*code-1)
                 for idx, code in zip(time_idx, ce_code)]  # [frame_num*[pos_l*2,1]]
        t_pe = torch.cat(t_pe_, dim=0)  # [frame_num, pos_l*2]
        t_embed = self.embed_mlp(t_pe)  # [frame_num, n_feats*4*2]
        # t_manip = self.manip_mlp(t_pe)

        # ce_blur feature
        ce_feature = self.feature(ce_blur) # [b, c, h, w]

        # main body
        output_list = []
        for k in range(len(time_idx)):
            feat_out_k = self.mainbody(ce_feature, t_embed[k])
            output_k = self.out(feat_out_k)
            output_list.append(output_k)

        output = torch.stack(output_list, dim=1)

        return output

class BDNeRV_RC_noTEM(nn.Module):
    # recursive frame reconstruction without temporal embedding module (TEM)
    def __init__(self):
        super(BDNeRV_RC_noTEM, self).__init__()
        # params
        n_colors = 3
        n_resblock = 4
        n_feats = 32
        kernel_size = 3
        padding = 1

        pos_b, pos_l = 1.25, n_feats*4  # position encoding params, without TEM
        # mlp_dim_list = [2*pos_l, 512, n_feats*4*2] # (160, 512, 256)
        # mlp_act = 'gelu'

        # main body
        self.mainbody = CUnet(n_feats=n_feats, n_resblock=n_resblock,
                              kernel_size=kernel_size, padding=padding)

        # output block
        OutBlock = [ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    ResBlock(Conv, n_feats=n_feats,kernel_size=kernel_size, padding=padding),
                    Conv(input_channels=n_feats, n_feats=n_colors, kernel_size=kernel_size, padding=padding)]
        self.out = nn.Sequential(*OutBlock)

        # feature block
        FeatureBlock = [Conv(input_channels=n_colors, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.feature = nn.Sequential(*FeatureBlock)

        # concatenation fusion block
        CatFusion = [Conv(input_channels=n_feats*2, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.catfusion = nn.Sequential(*CatFusion)

        # position encoding
        self.pe_t = PositionalEncoding(pe_embed_b=pos_b, pe_embed_l=pos_l)

        # # mlp
        # self.embed_mlp = MLP(dim_list=mlp_dim_list, act=mlp_act)
    
    def forward(self, ce_blur, time_idx, ce_code):
        # time index: [frame_num,1]
        # t_embed
        t_pe_ = [self.pe_t(idx)*(2*code-1)
                 for idx, code in zip(time_idx, ce_code)]  # [frame_num*[pos_l*2,1]]
        t_pe = torch.cat(t_pe_, dim=0)  # [frame_num, pos_l*2]
        t_embed = t_pe # without TEM
        # t_embed = self.embed_mlp(t_pe)  # [frame_num, n_feats*4*2]
        # t_manip = self.manip_mlp(t_pe)

        # ce_blur feature
        ce_feature = self.feature(ce_blur)  # [b, c, h, w]

        # main body
        output_list = []
        for k in range(len(time_idx)):
            if k==0:
                main_feature = ce_feature
            else:
                # since k=2, cat pre-feature with ce_feature as input feature
                cat_feature = torch.cat((feat_out_k, ce_feature),dim=1)
                main_feature = self.catfusion(cat_feature)
            feat_out_k = self.mainbody(main_feature, t_embed[k])
            output_k = self.out(feat_out_k)
            output_list.append(output_k)

        output = torch.stack(output_list, dim=1)

        return output

# ---------Analysis---------
class BDNeRV_RC_RA(nn.Module):
    # recursive frame reconstruction with random access ability
    def __init__(self):
        super(BDNeRV_RC_RA, self).__init__()
        # params
        n_colors = 3
        n_resblock = 4
        n_feats = 32
        kernel_size = 3
        padding = 1

        pos_b, pos_l = 1.25, 80  # position encoding params
        mlp_dim_list = [2*pos_l, 512, n_feats*4*2] # (160, 512, 256)
        mlp_act = 'gelu'

        # main body
        self.mainbody = CUnet(n_feats=n_feats, n_resblock=n_resblock,
                              kernel_size=kernel_size, padding=padding)

        # output block
        OutBlock = [ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    Conv(input_channels=n_feats, n_feats=n_colors, kernel_size=kernel_size, padding=padding)]
        self.out = nn.Sequential(*OutBlock)

        # feature block
        FeatureBlock = [Conv(input_channels=n_colors, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.feature = nn.Sequential(*FeatureBlock)

        # concatenation fusion block
        CatFusion = [Conv(input_channels=n_feats*2, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.catfusion = nn.Sequential(*CatFusion)

        # position encoding
        self.pe_t = PositionalEncoding(pe_embed_b=pos_b, pe_embed_l=pos_l)

        # mlp
        self.embed_mlp = MLP(dim_list=mlp_dim_list, act=mlp_act)

    def forward(self, ce_blur, ce_code, time_ticks=None):

        # time index: [frame_num,1]
        if time_ticks is None:
            time_ticks = torch.tensor(range(len(ce_code))).to(ce_blur.device)
        else:
            time_ticks = time_ticks.to(ce_blur.device)
        ce_code_ = ce_code[time_ticks.int().tolist()] # extract code at time_ticks
        time_ticks_ = time_ticks.unsqueeze(1)/(len(ce_code)-1) # normalize to 0-1
        
        
        # t_embed
        t_pe_ = [self.pe_t(idx)*(2*code-1)
                 for idx, code in zip(time_ticks_, ce_code_)]  # [frame_num*[pos_l*2,1]]
        t_pe = torch.cat(t_pe_, dim=0)  # [frame_num, pos_l*2]
        t_embed = self.embed_mlp(t_pe)  # [frame_num, n_feats*4*2]
        # t_manip = self.manip_mlp(t_pe)

        # ce_blur feature
        ce_feature = self.feature(ce_blur)  # [b, c, h, w]

        # main body
        output_list = []
        init_frame_flag = True
        for k in range(len(time_ticks_)):
            if init_frame_flag:
                main_feature = ce_feature
                init_frame_flag = False
            else:
                # since k=2, cat pre-feature with ce_feature as input feature
                cat_feature = torch.cat((feat_out_k, ce_feature),dim=1)
                main_feature = self.catfusion(cat_feature)
            feat_out_k = self.mainbody(main_feature, t_embed[k])
            output_k = self.out(feat_out_k)
            output_list.append(output_k)

        output = torch.stack(output_list, dim=1)

        return output

class BDNeRV_RA(nn.Module):
    # frame extraction without self-recursive strategy with random access ability
    def __init__(self):
        super(BDNeRV_RA, self).__init__()
        # params
        n_colors = 3
        n_resblock = 4
        n_feats = 32
        kernel_size = 3
        padding = 1

        pos_b, pos_l = 1.25, 80  # position encoding params
        mlp_dim_list = [2*pos_l, 512, 256]
        mlp_act = 'gelu'

        # main body
        self.mainbody = CUnet(n_feats=n_feats, n_resblock=n_resblock,kernel_size=kernel_size, padding=padding)

        # output block
        OutBlock = [ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    Conv(input_channels=n_feats, n_feats=n_colors, kernel_size=kernel_size, padding=padding)]
        self.out = nn.Sequential(*OutBlock)

        # feature block
        FeatureBlock = [Conv(input_channels=n_colors, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.feature = nn.Sequential(*FeatureBlock)

        # position encoding
        self.pe_t = PositionalEncoding(pe_embed_b=pos_b, pe_embed_l=pos_l)

        # mlp
        self.embed_mlp = MLP(dim_list=mlp_dim_list, act=mlp_act)



    def forward(self, ce_blur, ce_code, time_ticks=None):
        # time index: [frame_num,1]
        if time_ticks is None:
            time_ticks = torch.tensor(range(len(ce_code))).to(ce_blur.device)
        else:
            time_ticks = torch.tensor(time_ticks).to(ce_blur.device)
        ce_code_ = ce_code[time_ticks.int().tolist()] # extract code at time_ticks
        time_ticks_ = time_ticks.unsqueeze(1)/(len(ce_code)-1) # normalize to 0-1
        
        
        # t_embed
        t_pe_ = [self.pe_t(idx)*(2*code-1)
                 for idx, code in zip(time_ticks_, ce_code_)]  # [frame_num*[pos_l*2,1]]
        t_pe = torch.cat(t_pe_, dim=0)  # [frame_num, pos_l*2]
        t_embed = self.embed_mlp(t_pe)  # [frame_num, n_feats*4*2]
        # t_manip = self.manip_mlp(t_pe)

        # ce_blur feature
        ce_feature = self.feature(ce_blur) # [b, c, h, w]

        # main body
        output_list = []
        for k in range(len(time_ticks_)):
            feat_out_k = self.mainbody(ce_feature, t_embed[k])
            output_k = self.out(feat_out_k)
            output_list.append(output_k)

        output = torch.stack(output_list, dim=1)

        return output
    
# ---------BAK---------
class BDNeRV_BRC(nn.Module):
    # bi-directional recursive frame reconstruction
    def __init__(self):
        super(BDNeRV_BRC, self).__init__()
        # params
        n_colors = 3
        n_resblock = 4
        n_feats = 32
        kernel_size = 3
        padding = 1

        pos_b, pos_l = 1.25, 80  # position encoding params
        mlp_dim_list = [2*pos_l, 512, 256]
        mlp_act = 'gelu'

        # main body
        self.mainbody_forward = CUnet(n_feats=n_feats, n_resblock=n_resblock,
                              kernel_size=kernel_size, padding=padding)
        self.mainbody_backward = CUnet(n_feats=n_feats, n_resblock=n_resblock,
                                      kernel_size=kernel_size, padding=padding)

        # output block
        OutBlock_forward = [ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    ResBlock(Conv, n_feats=n_feats,
                             kernel_size=kernel_size, padding=padding),
                    Conv(input_channels=n_feats, n_feats=n_colors, kernel_size=kernel_size, padding=padding)]
        self.out_forward = nn.Sequential(*OutBlock_forward)
        OutBlock_backward = [ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    ResBlock(Conv, n_feats=n_feats,
                             kernel_size=kernel_size, padding=padding),
                    Conv(input_channels=n_feats, n_feats=n_colors, kernel_size=kernel_size, padding=padding)]
        self.out_backward = nn.Sequential(*OutBlock_backward)

        # feature block
        FeatureBlock_ce = [Conv(input_channels=n_colors, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.feature_ce = nn.Sequential(*FeatureBlock_ce)
        FeatureBlock_img = [Conv(input_channels=n_colors, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.feature_img = nn.Sequential(*FeatureBlock_img)

        # concatenation fusion block
        CatFusion_forward = [Conv(input_channels=n_feats*2, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                     ResBlock(Conv, n_feats=n_feats,
                              kernel_size=kernel_size, padding=padding),
                     ResBlock(Conv, n_feats=n_feats,
                              kernel_size=kernel_size, padding=padding),
                     ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.catfusion_forward = nn.Sequential(*CatFusion_forward)
        CatFusion_backward = [Conv(input_channels=n_feats*3, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                             ResBlock(Conv, n_feats=n_feats,
                                      kernel_size=kernel_size, padding=padding),
                             ResBlock(Conv, n_feats=n_feats,
                                      kernel_size=kernel_size, padding=padding),
                             ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.catfusion_backward = nn.Sequential(*CatFusion_backward)

        # position encoding
        self.pe_t = PositionalEncoding(pe_embed_b=pos_b, pe_embed_l=pos_l)

        # mlp
        self.embed_mlp = MLP(dim_list=mlp_dim_list, act=mlp_act)

    def forward(self, ce_blur, time_idx, ce_code):

        # t_embed
        t_pe_ = [self.pe_t(idx)*(2*code-1)
                 for idx, code in zip(time_idx, ce_code)]
        t_pe = torch.cat(t_pe_, dim=0)  # [b, pos_l*2]
        t_embed = self.embed_mlp(t_pe)  # [b, block_dim]

        # ce_blur feature
        ce_feature = self.feature_ce(ce_blur)  # [b, c, h, w]

        # mainbody forward
        output_list_forward = []
        for k in range(len(time_idx)):
            if k == 0:
                main_feature = ce_feature
            else:
                # since k=2, cat pre-feature with ce_feature as input feature
                cat_feature = torch.cat((feat_out_k, ce_feature), dim=1)
                main_feature = self.catfusion_forward(cat_feature)
            feat_out_k = self.mainbody_forward(main_feature, t_embed[k])
            output_k = self.out_forward(feat_out_k)
            output_list_forward.append(output_k)

        # mainbody backward
        output_list_backward = []
        for k in range(len(time_idx)-1, -1,-1):
            # since k=2, cat pre-feature with ce_feature as input feature
            img_k_feature = self.feature_img(output_list_forward[k])
            cat_feature = torch.cat(
                (feat_out_k, ce_feature, img_k_feature), dim=1)
            main_feature = self.catfusion_backward(cat_feature)
            feat_out_k = self.mainbody_backward(main_feature, t_embed[k])
            output_k = self.out_backward(feat_out_k)
            output_list_backward.append(output_k)

        output = torch.stack(output_list_backward[::-1], dim=1)

        return output
