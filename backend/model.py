import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # ---------------------------------------------------
        # ENCODER (Downsampling - Shrinking the image)
        # ---------------------------------------------------
        # Input: 1 channel (our spectrogram), Output: 16 feature maps
        self.enc1 = self.conv_block(1, 16)
        self.pool1 = nn.MaxPool2d(2) # Shrinks size by half

        self.enc2 = self.conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        # ---------------------------------------------------
        # BOTTLENECK (The deepest part of the network)
        # ---------------------------------------------------
        self.bottleneck = self.conv_block(32, 64)

        # ---------------------------------------------------
        # DECODER (Upsampling - Rebuilding the image)
        # ---------------------------------------------------
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # 32 channels from upconv + 32 channels from the skip connection = 64
        self.dec2 = self.conv_block(64, 32) 

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        # 16 from upconv + 16 from skip connection = 32
        self.dec1 = self.conv_block(32, 16)

        # ---------------------------------------------------
        # FINAL OUTPUT (Bringing it back to 1 channel)
        # ---------------------------------------------------
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    # A helper function to create convolutional layers easily
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. Run through Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # 2. Run through Bottleneck
        b = self.bottleneck(p2)

        # 3. Run through Decoder with Skip Connections
        u2 = self.upconv2(b)
        # Because 743 doesn't divide by 2 perfectly, we force the sizes to match
        if u2.shape != e2.shape:
            u2 = F.interpolate(u2, size=(e2.shape[2], e2.shape[3]))
        # SKIP CONNECTION: Concatenate 'u2' (upsampled) with 'e2' (from encoder)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.upconv1(d2)
        if u1.shape != e1.shape:
            u1 = F.interpolate(u1, size=(e1.shape[2], e1.shape[3]))
        # SKIP CONNECTION: Concatenate 'u1' with 'e1'
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # 4. Final Output
        out = self.final_conv(d1)
        return out

# --- TEST THE NETWORK ---
if __name__ == "__main__":
    print("Initializing U-Net Model...")
    model = UNet()
    
    # We create a fake tensor that has the exact shape of your audio [Batch, Channel, Height, Width]
    # Batch is 1 because we are testing 1 audio file.
    dummy_input = torch.randn(1, 1, 257, 743) 
    print(f"Feeding random noisy tensor of shape: {dummy_input.shape}")
    
    # Pass it through the model
    output = model(dummy_input)
    
    print(f"Model Output Shape: {output.shape}")
    if dummy_input.shape == output.shape:
        print("\n✅ SUCCESS! The U-Net successfully processed the shape and rebuilt it perfectly.")