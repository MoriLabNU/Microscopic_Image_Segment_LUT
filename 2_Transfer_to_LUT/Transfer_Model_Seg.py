
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CustomReflectionPad2d(nn.ReflectionPad2d):
    def forward(self, input):
        if input.shape[-1] == 1 and input.shape[-2] == 1:
            # 对于1x1的输入，使用上采样将其扩大到3x3
            return F.interpolate(input, size=(3, 3), mode='nearest')
        else:
            # 对于其他输入，使用正常的反射填充
            return super().forward(input)

# USER PARAMS
#UPSCALE = 1                 # upscaling factor
MODEL_PATH = "./SegNet_HV_2/model_G_i105000_best.pth"    # Trained SR net params

# Must match 3_Test_using_LUT/config.py and bundled LUT (Seg2_LUT_HV_2_4.npy uses 4).
SAMPLING_INTERVAL = 4        # N bit uniform sampling

class SegNet(torch.nn.Module):
    def __init__(self, num_classes=1):
        super(SegNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, [2,2], stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(64, num_classes, 1, stride=1, padding=0, dilation=1)
        #self.pixel_shuffle = nn.PixelShuffle(2)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        x = torch.sigmoid(x)
        return x


# class SegNet2(torch.nn.Module):
#     def __init__(self):
#         super(SegNet2, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, [2,2], stride=1, padding=0, dilation=1)
#         self.reflect_padding1 = CustomReflectionPad2d(1)
#         self.conv2 = nn.Conv2d(64, 64, [2,2], stride=1, padding=0, dilation=2)
#         self.reflect_padding2 = CustomReflectionPad2d(1)
#         self.conv3 = nn.Conv2d(64, 64, [2,2], stride=1, padding=0, dilation=2)
#         self.reflect_padding3 = CustomReflectionPad2d(1)
#         self.conv4 = nn.Conv2d(64, 64, [2,2], stride=1, padding=0, dilation=2)
#         self.reflect_padding4 = CustomReflectionPad2d(1)
#         self.conv5 = nn.Conv2d(64, 64, [2,2], stride=1, padding=0, dilation=2)
#         self.reflect_padding5 = CustomReflectionPad2d(1)
#         self.conv6 = nn.Conv2d(64, 1, [2,2], stride=1, padding=0, dilation=2)
 

#         # Init weights
#         for m in self.modules():
#             classname = m.__class__.__name__
#             if classname.lower().find('conv') != -1:
#                 nn.init.kaiming_normal_(m.weight)
#                 nn.init.constant_(m.bias, 0)
#             elif classname.find('bn') != -1:
#                 m.weight.data.normal_(1.0, 0.02)
#                 m.bias.data.fill_(0)


#     def forward(self, x_in):
#         x = self.conv1(x_in)
#         x = self.reflect_padding1(x)
#         x = self.conv2(F.relu(x))
#         x = self.reflect_padding2(x)
#         x = self.conv3(F.relu(x))
#         x = self.reflect_padding3(x)
#         x = self.conv4(F.relu(x))
#         x = self.reflect_padding4(x)
#         x = self.conv5(F.relu(x))
#         x = self.reflect_padding5(x)
#         x = self.conv6(F.relu(x))
#         x = torch.sigmoid(x)
    
#         return x

model_G = SegNet().cuda()


lm = torch.load('{}'.format(MODEL_PATH))

model_G.load_state_dict(lm.state_dict(), strict=True)



### Extract input-output pairs
with torch.no_grad():
    model_G.eval()

    # 1D input
    base = torch.arange(0, 257, 2**SAMPLING_INTERVAL)   # 0-256
    base[-1] -= 1
    L = base.size(0)
    
    # 2D input
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)  # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    second = base.cuda().repeat(L)                             # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]
    #print('onebytwo',onebytwo.shape)

    # 3D input
    third = base.cuda().unsqueeze(1).repeat(1, L*L).reshape(-1) # 256*256*256   0 x65536|1 x65536|...|255 x65536
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)    # [256*256*256, 3]
    #print('onebythree',onebytwo.shape)

    # 4D input
    fourth = base.cuda().unsqueeze(1).repeat(1, L*L*L).reshape(-1) # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L, 1)
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)    # [256*256*256*256, 4]
    #print('onebyfourth',onebytwo.shape)

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1,1,2,2).float() / 255.0
    print("Input size: ", input_tensor.size())

    B = input_tensor.size(0) // 100

    outputs = []

    for b in range(100):
        if b == 99:
            batch_output = model_G(input_tensor[b*B:])
        else:
            batch_output = model_G(input_tensor[b*B:(b+1)*B])
    
        results = (batch_output.cpu().data.numpy()* 255).astype(np.uint8)
        
        outputs += [ results ]

        #print(np.unique(results)) #[0,255]
        
        
    
    results = np.concatenate(outputs, 0)
    print("Resulting LUT size: ", results.shape)

    out_name = "./Seg2_LUT_HV_2_{}.npy".format(SAMPLING_INTERVAL)
    np.save(out_name, results)
    print("Saved LUT:", out_name)






