import torch
import torch.nn as nn

class MaskedConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel, stride, padding):
        super().__init__()
        identity = torch.ones((kernel // 2, kernel))
        mask = torch.zeros((kernel - kernel // 2, kernel))
        filter_mask = torch.cat((identity, mask), dim=0)
        self.register_buffer('filter_mask', filter_mask)
        self.conv = nn.Conv2d(n_in, n_out, kernel, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(n_out)
        
    def forward(self, x):
        self._mask_conv_filter()
        x = self.conv(x)
        x = self.norm(x)
        return x
    
    def _mask_conv_filter(self):
        with torch.no_grad():
            self.conv.weight.mul_(self.filter_mask)
            
class MaskedConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel, stride, padding, output_padding):
        super().__init__()
        identity = torch.ones((kernel - kernel // 2, kernel))
        mask = torch.zeros((kernel // 2, kernel))
        filter_mask = torch.cat((identity, mask), dim=0)
        self.register_buffer('filter_mask', filter_mask)
        self.deconv = nn.ConvTranspose2d(n_in, n_out, kernel, stride=stride, padding=padding, output_padding=output_padding)
        self.norm = nn.BatchNorm2d(n_out)
        
    def forward(self, x):
        self._mask_conv_filter()
        x = self.deconv(x)
        x = self.norm(x)
        return x
    
    def _mask_conv_filter(self):
        with torch.no_grad():
            self.deconv.weight.mul_(self.filter_mask)
            
class MusicCNN(nn.Module):
    def __init__(self, k1, k2):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.down1 = MaskedConv2d(1, 12, k1, stride=(2, 1), padding=k1 // 2)
        self.down2 = MaskedConv2d(12, 12, k2, stride=(2, 1), padding=k2 // 2)
        self.down3 = MaskedConv2d(12, 12, k1, stride=(2, 1), padding=k1 // 2)
        self.down4 = MaskedConv2d(12, 12, k2, stride=1, padding=k2 // 2)
        self.up1 = MaskedConvTranspose2d(12, 12, 3, stride = (2, 1), padding=(1, 1), output_padding=(1, 0))
        self.up2 = MaskedConvTranspose2d(12, 12, 3, stride = (2, 1), padding=(1, 1), output_padding=(1, 0))
        self.up3 = MaskedConvTranspose2d(12, 1, 3, stride = (2, 1), padding=(1, 1), output_padding=(1, 0))
        self.activation = nn.ReLU()
       
    def forward(self, x):
        x = self.norm(x)
        x1 = self.activation(self.down1(x))
        x2 = self.activation(self.down2(x1))
        x3 = self.activation(self.down3(x2))
        x4 = self.activation(self.down4(x3) + x3)
        x5 = self.activation(self.up1(x4)[:, :, :x2.shape[2], :] + x2)
        x6 = self.activation(self.up2(x5)[:, :, :x1.shape[2], :] + x1)
        x7 = self.activation(self.up3(x6)[:, :, :x.shape[2], :] + x)
        return x7
    
