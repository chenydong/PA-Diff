import torch
from torch import nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
class PDU(nn.Module):  # physical block
    def __init__(self, channel):
        super(PDU, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.avg_poolh = nn.AdaptiveAvgPool2d((None,1))
        #self.avg_poolw = nn.AdaptiveAvgPool2d((1,None))
        #self.patch_embed = PatchEmbed(img_size=128, patch_size=7, stride=1, in_chans=channel,
        #                                     embed_dim=channel)
        #window_size = 7
        #depths=2
        #self.swin = nn.Sequential(*[SwinTransformerLayer(
                #dim=channel, num_heads=3, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(depths[0])])
        self.ka = nn.Sequential(
            #RDB(channel),
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.conv3 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)
        self.t1 = nn.Sequential(
            #default_conv(channel, channel, 3),#nn.BatchNorm2d(channel),nn.GELU(),
            #RDB(channel),
            #SwinTransformerLayer(
                #dim=channel, num_heads=4, window_size=window_size, shift_size=0),
            default_conv(channel, channel // 8, 3),
            nn.ReLU(inplace=True),
            default_conv(channel // 8, channel, 3),
            nn.Sigmoid()
        )
        self.t2 = nn.Sequential(
            #default_conv(channel, channel, 3),#nn.BatchNorm2d(channel),nn.GELU(),
            #RDB(channel),
            #SwinTransformerLayer(
                #dim=channel, num_heads=4, window_size=window_size, shift_size=0),
            default_conv(channel, channel // 8, 3),
            nn.ReLU(inplace=True),
            default_conv(channel // 8, channel, 3),
            nn.Sigmoid()
        )



    def forward(self, x):
        b = self.avg_pool(x)
        #a1=self.avg_poolh(x)
        #a2=self.avg_poolw(x)
        #a=a1*a2
        b = self.ka(b)
        #print(x.shape)
        #y, H1, W1 = self.patch_embed(x)
        #print(x1.shape)
        t = self.conv3(x)
        t1=self.t1(t)
        t2=self.t2(t)
        #t=1/t
        j = torch.mul(t1, b) + torch.mul(t2, x)
        return j
