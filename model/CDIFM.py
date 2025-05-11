import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from model.miniViT import mViT

class WeightNet(nn.Module):
    def __init__(self, number_features=160):
        super(WeightNet, self).__init__()       
        self.conv1 = nn.Conv2d(number_features, number_features*2, kernel_size=3, stride=1, padding=1)
        self.leakyreluA1 = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv2d(number_features*2, number_features*8, kernel_size=3, stride=1, padding=1)
        self.leakyreluA2 = nn.LeakyReLU(0.2)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.leakyreluA1(self.conv1(x))
        x2 = self.leakyreluA2(self.conv2(x1))
        out = self.sigmoid(x2)
        
        return out
    
class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with1):
        up_x = F.interpolate(x, size=[concat_with1.size(2), concat_with1.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.leakyreluA(self.convA( torch.cat([up_x, concat_with1], dim=1) ) ) )  )

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

    def forward(self, features1):
        x1_block0, x1_block1, x1_block2, x1_block3, x1_block4,x1_block5,x1_block6 = features1[2], features1[4], features1[6], features1[9], features1[15],features1[18],features1[19]
        
        x_d0 = self.conv2(x1_block6)
        x_d1 = self.up0(x_d0, x1_block5)
        x_d2 = self.up1(x_d1, x1_block4)
        x_d3 = self.up2(x_d2, x1_block3)
        x_d4 = self.up3(x_d3, x1_block2)
        x_d5 = self.up4(x_d4, x1_block1)
        x_d6 = self.up5(x_d5, x1_block0)
        return x_d6

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        import torchvision.models as models
        self.model = models.mobilenet_v2(pretrained=True) 

    def forward(self, x):
        features = [x]
        for k, v in self.model.features._modules.items():
            features.append(v(features[-1]))
        return features

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x1, x2):
        return self.decoder( self.encoder(x1), self.encoder(x2) )
    
class CDIFM(nn.Module):
    def __init__(self, backend, n_bins=100, min_val=0.001, max_val=1, norm='linear'):
        super(CDIFM, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        
        self.encoder = Encoder()
        #self.encoder_t = Encoder()
        self.decoder = Decoder()
        
        #self.weightnet = WeightNet()
        
        self.adaptive_bins_layer = mViT(48, n_query_channels=48, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=48, norm=norm)
        
        self.conv_out = nn.Sequential(nn.Conv2d(48, n_bins, kernel_size=1, stride=1, padding=0), 
                                      nn.Softmax(dim=1))
        #self.conv_out = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                              nn.Conv2d(48, n_bins, kernel_size=1, stride=1, padding=0),
        #                              nn.Softmax(dim=1))

    def adjust_values(self, x, relation_matrix, weight, delta):
        # Split the merged input into two channels
        x0, x1, x2 = x.chunk(3, dim=1)

        # Calculate the adjusted values
        adjusted_x = weight * x1 + (1 - weight) * x2
    
        # Create condition masks
        condition_mask = relation_matrix < delta
        weight_mask = weight >= 0.5

        # Adjust values according to the condition masks
        x1_adjusted = torch.where(condition_mask & ~weight_mask, adjusted_x, x1)
        x2_adjusted = torch.where(condition_mask & weight_mask, adjusted_x, x2)

        return torch.cat([x0, x1_adjusted, x2_adjusted], dim=1)
    
    def calculate_relation(self, x):
        # Calculate abs
        _, x1, x2 = x.chunk(3, dim=1)
        relation = 1-torch.abs(x1 - x2)

        # Normalize to 0-1
        min_values = torch.min(torch.min(relation, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]  
        max_values = torch.max(torch.max(relation, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0] 
        relation = (relation - min_values) / (max_values - min_values)

        return relation

    def forward(self, x, weight, delta=0.6, **kwargs):
        relation_matrix = self.calculate_relation(x)
        x = self.adjust_values(x, relation_matrix, weight, delta)
        out = self.decoder(self.encoder(x))
        
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(out)

        out = self.conv_out(range_attention_maps)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        return bin_edges, pred

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