import torch

class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, encoder_channels=[64, 128, 256, 512], encoder_kernel_sz=[3, 3, 3, 3], encoder_padding=[1, 1, 1, 1],
                 decoder_channels=[512, 256, 128, 64], decoder_kernel_sz=[3, 3, 3, 3], decoder_padding=[1, 1, 1, 1], 
                 latent_kernel_sz=(3, 3), latent_padding=(1, 1), out_channels=2):
        # (646, 486) -> kernel size 7, padding 3
        super(UNet, self).__init__()
        
        self.num_blocks = len(encoder_channels)
        
        
        if len(encoder_channels) != len(decoder_channels):
            raise ValueError("UNet must have symmetric encoder-decoder design!")
        
        self.encoder = torch.nn.ModuleList(torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels if i==0 else encoder_channels[i-1], encoder_channels[i], kernel_size=encoder_kernel_sz[i], padding=encoder_padding[i]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(encoder_channels[i], encoder_channels[i], kernel_size=encoder_kernel_sz[i], padding=encoder_padding[i]),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            ])
         for i in range(len(encoder_channels)))
        
        self.decoder = torch.nn.ModuleList(torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=encoder_channels[len(encoder_channels)-1-i]+decoder_channels[i], out_channels=decoder_channels[i], kernel_size=decoder_kernel_sz[i], padding=decoder_padding[i]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(decoder_channels[i], decoder_channels[i], kernel_size=decoder_kernel_sz[i], padding=decoder_padding[i]),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1], kernel_size=2, stride=2, padding=0) if i != len(decoder_channels)-1  # 2x up-sampling
                else torch.nn.Conv2d(decoder_channels[i], out_channels, kernel_size=(1, 1))
            ]) 
        for i in range(len(decoder_channels)))
        
        self.latent_transform = torch.nn.Sequential(
            torch.nn.Conv2d(encoder_channels[-1], encoder_channels[-1] * 2, kernel_size=latent_kernel_sz, padding=latent_padding),
            torch.nn.ReLU(),
            torch.nn.Conv2d(encoder_channels[-1] * 2, encoder_channels[1] * 2, kernel_size=latent_kernel_sz, padding=latent_padding),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(encoder_channels[1] * 2, decoder_channels[0], kernel_size=2, stride=2, padding=0)
        )
        self.activation = torch.nn.Softmax2d()
        self._init_weights()
    
    def _init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                torch.nn.init.zeros_(p.data)
        
    def forward(self, x):
        encoder_outputs = []
        
        # run encoder
        for block in self.encoder:
            for i, layer in enumerate(block, 0):
                if i ==len(block) - 1: # save encoder output before down sampling
                    encoder_outputs.append(x)
                x = layer(x)
            
        # run latent transform
        x = self.latent_transform(x)
        
        # run decoder
        for i, block in enumerate(self.decoder):
            x = torch.cat((x, encoder_outputs[self.num_blocks-1-i]), dim=1)
            for layer in block:
                x = layer(x)
                
        x = self.activation(x)
        return x