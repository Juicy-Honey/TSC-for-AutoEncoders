import torch
import torch.nn as nn
import torch.nn.functional as F

import vgg_encoder

################################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect')
        self.leakyrelu = nn.LeakyReLU(0.2)

        # He Initialization
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.leakyrelu(x)
        return x

class SRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, upsample=True):
        super(SRBlock, self).__init__()
        self.upsample = upsample
        self.c1 = ConvBlock(in_channels , out_channels, kernel_size, stride, padding)
        self.c2 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding)
        self.c3 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding)
        self.c4 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding)
        self.c5 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding)
        self.ce = ConvBlock(out_channels, out_channels, kernel_size, stride, padding)
          
    def forward(self, x):
        if self.upsample:
          x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.ce(x)

        return x

class AttentionLayer(nn.Module):
    def __init__(self, in_channels, kernel_size=3, groups=32):
        super(AttentionLayer, self).__init__()
        self.dynamic_filter = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=groups, 
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skip):
        # Dynamic filter weight
        filter_weights = self.sigmoid(self.dynamic_filter(skip))
  
        return x * filter_weights + skip * (1 - filter_weights)

################################################################################
class SRAE(nn.Module):
    def __init__(self, scale=4, depth=3):
        super(SRAE, self).__init__()
        self.depth  = depth
        self.scale  = scale
        
        channels = [3, 64, 128, 256, 512, 512]
        c = channels[depth]

        # Encoders ######################################################
        self.encoders = [None] #0: RGB
        if depth >= 1:
            self.encoders.append(vgg_encoder.block1())
        if depth >= 2:
            self.encoders.append(vgg_encoder.block2())
        if depth >= 3:
            self.encoders.append(vgg_encoder.block3())
        if depth >= 4:
            self.encoders.append(vgg_encoder.block4())
        if depth >= 5:
            self.encoders.append(vgg_encoder.block5())
        self.encoders = nn.ModuleList(self.encoders)

        # Update = False
        for enc in self.encoders:
          if enc != None:
            for param in enc.parameters():
                param.requires_grad = False
 
        # init Block ####################################################
        self.init_block = nn.Sequential(
            ConvBlock(c, c, kernel_size=5, stride=1, padding=2),
            ConvBlock(c, c, kernel_size=5, stride=1, padding=2),
            ConvBlock(c, c, kernel_size=5, stride=1, padding=2),
            ConvBlock(c, c, kernel_size=5, stride=1, padding=2))
        
        # SR Block #######################################################
        self.sr_blocks = []
        if scale >= 2:
          self.sr_blocks.append(SRBlock(c, c, kernel_size=5, stride=1, padding=2))
        if scale >= 4:
          self.sr_blocks.append(SRBlock(c, c, kernel_size=5, stride=1, padding=2))
        self.sr_blocks = nn.ModuleList(self.sr_blocks)

        # Decoders #######################################################
        self.decoders = [None]
        if depth >= 1:
            toRGB = nn.Sequential(
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            ConvBlock(32, 16, kernel_size=3, stride=1, padding=1),
            ConvBlock(16, 3, kernel_size=3, stride=1, padding=1))
            self.decoders.append(toRGB)
        if depth >= 2:
            self.decoders.append(SRBlock(128, 64, kernel_size=5, stride=1, padding=2))
        if depth >= 3:
            self.decoders.append(SRBlock(256, 128, kernel_size=5, stride=1, padding=2))
        if depth >= 4:
            self.decoders.append(SRBlock(512, 256, kernel_size=5, stride=1, padding=2))
        if depth >= 5:
            self.decoders.append(SRBlock(512, 512, kernel_size=5, stride=1, padding=2))
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x, guides, trainMode=False):
        lr_img  = x # for residual.
        guide_image = guides[len(guides)-1]
        depth = self.depth
        temp = []

        # Encode!
        x = self.encoders[depth](x)

        # init block
        x   = self.init_block(x)
        res = self.encoders[depth](lr_img)
        x = x+res#self.attentions[depth](x, res)

        temp.append(x) # for Train loss

        # SR blocks
        count = 0
        for block in self.sr_blocks:
          x   = block(x)
          res = self.encoders[depth](guides[count])
          count += 1
          x = x+res#self.attentions[depth](x, res)
          
          temp.append(x) # for Train loss
        
        # Decoders
        for d in range(depth, 0, -1):
          x   = self.decoders[d](x)
          if d == 1 :
            res = guide_image
          else:
            res = self.encoders[d-1](guide_image)
          x = x+res#self.attentions[d-1](x, res)
          temp.append(x) # for Train loss
        
        if trainMode:
          return x, temp
        return x

    def get_gt(self, GT): # GT = (GT_original, GT_Half, GT_Quater)
        """ train_data:
        - init results      : 1
        - SR results (1~2)  : self.scale//2
        - DC results (1~5)  : self.depth
        """
        temp = []
        depth = self.depth

        # GT Resized
        if self.scale >= 2:
          GT_half = F.interpolate(GT, scale_factor=0.5, mode='bicubic', align_corners=False)
        if self.scale >= 4:
          GT_quat = F.interpolate(GT, scale_factor=0.25, mode='bicubic', align_corners=False)

        # init & SR
        if self.scale >= 4:
          temp.append(self.encoders[depth](GT_quat)) # init
        if self.scale >= 2:
          temp.append(self.encoders[depth](GT_half)) # init or x2
          temp.append(self.encoders[depth](GT)) # ori

        # DC
        for d in range(depth, 0, -1):
          if d == 1 :
            x = GT
          else:
            x = self.encoders[d-1](GT)
          temp.append(x) 

        return temp
    def load_model(self, path, target_depth):
        # Load the state_dict from the file
        state_dict = torch.load(path)
        current_depth = self.depth

        # Load decoders
        for d in range(1, min(current_depth, target_depth) + 1):
            decoder_key = f"decoders.{d}."
            decoder_state = {
                k.replace(decoder_key, ""): v
                for k, v in state_dict.items()
                if k.startswith(decoder_key)
            }
            if decoder_state:
                self.decoders[d].load_state_dict(decoder_state, strict=False)
                print(f"Decoder {d} loaded successfully.")

        # Load SR blocks (only if target_depth matches current_depth)
        if target_depth == current_depth:
            for i, sr_block in enumerate(self.sr_blocks):
                sr_block_key = f"sr_blocks.{i}."
                sr_block_state = {
                    k.replace(sr_block_key, ""): v
                    for k, v in state_dict.items()
                    if k.startswith(sr_block_key)
                }
                if sr_block_state:
                    sr_block.load_state_dict(sr_block_state, strict=False)
                    print(f"SR Block {i} loaded successfully.")

        # Handle case where target_depth is deeper than current_depth
        if target_depth > current_depth:
            print(
                f"Warning: The target depth ({target_depth}) is greater than the current model depth ({current_depth}). "
                f"Only loading up to the current depth."
            )


################################################################################