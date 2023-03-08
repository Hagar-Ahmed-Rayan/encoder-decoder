import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor#default='resnet101',
        self.pretrained = args.visual_extractor_pretrained#default=True,

        #class object,'resnet101',
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)#get#resnet101
        modules = list(model.children())[:-2]##bya5od 7aga mo3yna mn el pretrained model 34an yzbtah 3la el data ely ma3ah?
        self.model = nn.Sequential(*modules)##*?
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)# bb3t llel model el sor f byrg3 el features
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        #bya5od el feature elly tl3t mn el model el resnet w yd5la 3la avepool 34an kda esmah ave features
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
