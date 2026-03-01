from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
#考虑对图像和文本加入cross，IAAR只针对掩码文本cross，找到更多匹配机制、特征提取、损失计算
#修改只针对图像用safl，对文本不使用

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.LayerNorm(emb_size,dtype=torch.float16),
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
class SAFL(nn.Module):
    def __init__(self, dim=512, part_num=6) -> None:
        super().__init__()

        self.part_num = part_num
        self.part_tokens = nn.Parameter(nn.init.kaiming_normal_(torch.empty(part_num, dim,dtype=torch.float16)))
        self.pos_embeding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(18 * 9, dim)))
        self.feed=FeedForwardBlock(emb_size=dim,drop_p=0.3)
        self.active = nn.Sigmoid()
        self.scaling=dim**(1/2)
    def forward(self, x):
        
        # x_pos = x + self.pos_embeding
        B=x.shape[0] 

        attn = self.part_tokens @ x.transpose(-1, -2)
        # attn = self.active(attn)
        attn=attn.softmax(dim=-1)

        x = attn @ x /self.scaling
    

        x= self.feed(x)
        return x
        # return x.view(B, -1)

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        self.num=9

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)
        #增加对比损失，对比学习，新的损失机制和匹配机制以及新特征便于匹配，考虑gcn。
        #图cross-modal transformer
        #加个模版匹配损失
        layers=6
        heads=8
        #test##################################
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        #test##################################
    
        self.shared_transformer = Transformer(self.embed_dim, layers, heads)
        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)
            # self.a=nn.Parameter(nn.init.kaiming_normal_(torch.empty(1,dtype=torch.float16)))
        # if 'sdm' in self.current_task:
        #      self.b=nn.Parameter(nn.init.kaiming_normal_(torch.empty(1,dtype=torch.float16)))
        if 'mlm' in args.loss_names:
            # self.c=nn.Parameter(nn.init.kaiming_normal_(torch.empty(1,dtype=torch.float16)))
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
        self.safl = SAFL(dim=self.embed_dim, part_num=self.num,)
        

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD


        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)

        safl_image_feats= self.safl(x[:,1:,:])
        safl_image_feats=self.shared_transformer(safl_image_feats)
        
        # safl_i_feats=torch.mean(safl_image_feats,dim=1)
        i_feats = x[:, 0, :].float()
        safl_i_feats=torch.mean(safl_image_feats,dim=1)
        # i_feats = torch.cat([safl_image_feats, i_feats], dim=1)
        i_feats=safl_i_feats+i_feats

        return i_feats
        
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        safl_text_feats= self.safl(x)
        safl_text_feats=self.shared_transformer(safl_text_feats)
        t_feats = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        safl_t_feats=torch.mean(safl_text_feats,dim=1)
        t_feats=safl_t_feats+t_feats
        return t_feats
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        # print(image_feats.shape)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        safl_image_feats= self.safl(image_feats[:,1:,:])
        safl_text_feats= self.safl(text_feats)
        safl_image_feats=self.shared_transformer(safl_image_feats)
        safl_text_feats=self.shared_transformer(safl_text_feats)
        safl_i_feats=torch.mean(safl_image_feats,dim=1)
        safl_t_feats=torch.mean(safl_text_feats,dim=1)

        self.alpha = max(-10, min(self.alpha, 10))
        self.beta = max(-10, min(self.beta, 10))
        i_feats = (safl_i_feats * i_feats) * self.alpha / 10000 + (safl_i_feats + i_feats)
        t_feats = (safl_t_feats * t_feats) * self.beta / 10000 + (safl_t_feats + t_feats)

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        
        if 'itc' in self.current_task:
            
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)*5})
        
        if 'sdm' in self.current_task:
            # print(objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale),"121324");
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})
            # ret.update({'sdm_loss':objectives.compute_sdm(safl_i_feats, safl_t_feats, batch['pids'], logit_scale)})
            # ret.update({'sdm_loss':objectives.compute_sdm(safl_i_feats, safl_t_feats, batch['pids'], logit_scale)*self.b})


        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            # image_logits = self.classifier(safl_i_feats.half()).float()
            # text_logits = self.classifier(safl_t_feats.half()).float()
        
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*0.5})
            # ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.a})


            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})
            # ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.c})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if 'pacl' in self.current_task:
            ret.update({'pacl_loss': objectives.compute_pacl(
                safl_image_feats, safl_text_feats, batch['pids'], logit_scale,
                ot_reg=getattr(self.args, 'pacl_ot_reg', 0.1),
                ot_iters=getattr(self.args, 'pacl_ot_iters', 10)
            ) * getattr(self.args, 'pacl_loss_weight', 1.0)})

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
