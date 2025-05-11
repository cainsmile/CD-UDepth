"""
# > Model architecture of Udepth 
#    - Paper: https://arxiv.org/pdf/2209.12358.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .miniViT import mViT
from .weightnet import WeightNet
from .CDIFM import CDIFM


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.leakyreluA(self.convA( torch.cat([up_x, concat_with], dim=1) ) ) )  )

class Decoder(nn.Module):
    def __init__(self, num_features=1280, decoder_width = .6):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)
        
        self.up0 = UpSample(skip_input=features//1 + 320, output_features=features//2)
        self.up1 = UpSample(skip_input=features//2 + 160, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 64, output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 32, output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 +  24, output_features=features//8)
        self.up5 = UpSample(skip_input=features//8 +  16, output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4,x_block5,x_block6 = features[2], features[4], features[6], features[9], features[15],features[18],features[19]
        x_d0 = self.conv2(x_block6)
        x_d1 = self.up0(x_d0, x_block5)
        x_d2 = self.up1(x_d1, x_block4)
        x_d3 = self.up2(x_d2, x_block3)
        x_d4 = self.up3(x_d3, x_block2)
        x_d5 = self.up4(x_d4, x_block1)
        x_d6 = self.up5(x_d5, x_block0)
        return x_d6

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()       
        import torchvision.models as models
        #self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1)
        #self.bn1 = nn.BatchNorm2d(num_features=3, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        
        self.original_model = models.mobilenet_v2( pretrained=True )

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )

class UDepth(nn.Module):
    def __init__(self, backend, n_bins=100, min_val=0.001, max_val=1, norm='linear'):
        super(UDepth, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = Encoder()
        self.adaptive_bins_layer = mViT(48, n_query_channels=48, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=48, norm=norm)
        self.decoder = Decoder()
        
        self.conv_out = nn.Sequential(nn.Conv2d(48, n_bins, kernel_size=1, stride=1, padding=0), 
                                      nn.Softmax(dim=1))
        
        #self.conv_out = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                              nn.Conv2d(48, n_bins, kernel_size=1, stride=1, padding=0),
        #                              nn.Softmax(dim=1))

    def forward(self, x, **kwargs):
        unet_out = self.decoder(self.encoder(x))
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)

        out = self.conv_out(range_attention_maps)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        return bin_edges, pred
        #return unet_out

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, **kwargs):
        '''
        basemodel_name = 'tf_efficientnet_b5_ap'

        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        '''
        basemodel = timm.create_model('tf_efficientnet_b5_ap', pretrained=False)
        basemodel.load_state_dict(torch.load('./saved_model/tf_efficientnet_b5_ap.pth'))
        print('Done.')

        # Remove last layer
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        m = cls(basemodel, n_bins=n_bins, **kwargs)
        print('Done.')
        return m
    
class CD_UDepth(nn.Module):
    def __init__(self):
        super(CD_UDepth, self).__init__()
        
        self.submodel_rgb = UDepth.build(n_bins=80, min_val=0.001, max_val=1, norm="linear")
        self.submodel_imt = UDepth.build(n_bins=80, min_val=0.001, max_val=1, norm="linear")
        self.weightnet = WeightNet()
        self.cdifm = CDIFM.build(n_bins=80, min_val=0.001, max_val=1, norm="linear")
        #self.conv_out = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                              nn.Conv2d(48, n_bins, kernel_size=1, stride=1, padding=0),
        #                              nn.Softmax(dim=1))

    def forward(self, rgb, imt, delta, **kwargs):
        h, w = rgb.shape[2], rgb.shape[3]
        x_rgb = nn.functional.interpolate(self.submodel_rgb(rgb)[-1], size=(h, w), mode='bilinear', align_corners=True)
        x_imt = nn.functional.interpolate(self.submodel_imt(imt)[-1], size=(h, w), mode='bilinear', align_corners=True)
        x = torch.cat([imt[:, 0:1, :, :],x_rgb,x_imt],dim=1)
        weight = self.weightnet(x)
        
        bin_edges, pred = self.cdifm(x,weight,delta)
        return bin_edges, pred
    
if __name__ == '__main__':
    model = CD_UDepth()
    rgb = torch.rand(2, 3, 480, 640)
    imt = torch.rand(2, 3, 480, 640)
    bins, pred = model(rgb,imt)
    print(bins.shape, pred.shape)
