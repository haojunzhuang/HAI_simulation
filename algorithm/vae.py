import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_fc1 = nn.Linear(128 * 93 * 50, 512)
        self.enc_fc_mu = nn.Linear(512, 256)
        self.enc_fc_logvar = nn.Linear(512, 256)
        
        # Decoder
        self.dec_fc1 = nn.Linear(256, 512)
        self.dec_fc2 = nn.Linear(512, 128 * 93 * 50)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(32, 5, kernel_size=3, stride=2, padding=1, output_padding=1) 
        # 5 channel for 5-way softmax (even though only 4 classes)

    def _encode(self, x):
        h = torch.relu(self.enc_conv1(x))
        h = torch.relu(self.enc_conv2(h))
        h = torch.relu(self.enc_conv3(h))
        h = h.view(h.size(0), -1)
        h = torch.relu(self.enc_fc1(h))
        return self.enc_fc_mu(h), self.enc_fc_logvar(h)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(self, z):
        h = torch.relu(self.dec_fc1(z))
        h = torch.relu(self.dec_fc2(h))
        h = h.view(h.size(0), 128, 93, 50) # batch_size, in_channel, height, width
        h = torch.relu(self.dec_conv1(h))
        h = torch.relu(self.dec_conv2(h))
        return torch.sigmoid(self.dec_conv3(h))

    def forward(self, x):
        x = x.unsqueeze(1) # channel=1 at dim 1
        
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        recon = self._decode(z)
        probs = nn.functional.softmax(recon, dim=1)
        return probs, mu, logvar

def vae_loss(y, recon_x, mu, logvar):
    CE = nn.functional.cross_entropy(recon_x, y)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return CE + KLD