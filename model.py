import torch
import torch.nn as nn
import math


class conv3x3bn(nn.Module):
	def __init__(self,in_ch,out_ch,groups=1,act_layer=nn.LeakyReLU):
		super(conv3x3bn,self).__init__()
		self.layer = nn.Sequential(
                        nn.Conv2d(in_ch,out_ch,kernel_size=3,groups=groups,stride=1,padding=1),
                        nn.BatchNorm2d(out_ch),
                        act_layer()
		)
	def forward(self,x):
		return self.layer(x)

class conv1d1x1(nn.Module):
	def __init__(self,in_ch,out_ch,act_layer = nn.GELU):
		super(conv1d1x1,self).__init__()
		self.layer = nn.Sequential(
			nn.Conv1d(in_ch,out_ch,kernel_size=1,stride=1,padding=0),
			act_layer()
		)
	def forward(self,x):
		return self.layer(x)

class Leff(nn.Module):
	def __init__(self,in_ch,hidden_ch,out_ch):
		super(Leff,self).__init__()
		self.conv1 = conv1d1x1(in_ch,hidden_ch)
		self.conv2 = conv3x3bn(hidden_ch,hidden_ch,hidden_ch,nn.GELU)
		self.conv3 = conv1d1x1(hidden_ch,out_ch)

	def forward(self,x):
		b,emb,c = x.size()
		hw = int(math.sqrt(emb))
		x = self.conv1(x.permute(0,2,1))
		x = x.view(b,-1,hw,hw)
		x = self.conv2(x)
		x= x.view(b,-1,hw*hw)
		x = self.conv3(x)
		return x.permute(0,2,1)

class LeWin(nn.Module):
	def __init__(self,emb_dim,hidden_ch,win_size,num_heads,drop,modulators):
		super(LeWin,self).__init__()
		self.norm1 = nn.LayerNorm(emb_dim)
		self.norm2 = nn.LayerNorm(emb_dim)
		self.wmsa = nn.MultiheadAttention(emb_dim,num_heads,drop)
		self.leff = Leff(emb_dim,hidden_ch,emb_dim)
		self.modulator = nn.Embedding(win_size**2,emb_dim) if modulators else None

	def forward(self,x):
		y=x
		x=self.norm1(x)
		if self.modulator is not None:
			x=self.addModulator(x,self.modulator.weight)
		x=self.wmsa(x,x,x)[0]
		x=x+y
		y=x
		x=self.norm2(x)
		x=self.leff(x)
		return x+y

	def addModulator(self,x,mod):
		return x if mod is None else x+mod

class LeWinBlock(nn.Module):
	def __init__(self,emb_dim,num_layer,num_heads,mlp_ratio,drop,win_size,modulators=False):
		super(LeWinBlock,self).__init__()
		hidden_dim = emb_dim*mlp_ratio
		self.win_size=win_size
		self.layer = nn.ModuleList(
			[
				LeWin(emb_dim,hidden_dim,win_size,num_heads,drop,modulators)
			 for __ in range(num_layer)]
		)

	def forward(self,x):
		x = self.patch_process(x)
		for l in self.layer:
			x= l(x)
		x = self.depatch_process(x)
		return x

	def patch_process(self,x):
		b,c,h,w = x.shape

		x = x.view(b,c,h//self.win_size,self.win_size,w//self.win_size,self.win_size)
		x= x.flatten(2)

		return x.permute(0,2,1)

	def depatch_process(self,x):
		b,emb,c = x.shape
		hw = int(math.sqrt(emb))
		x = x.permute(0,2,1).view(b,c,hw,hw)
		return x

class downsampleConv(nn.Module):
	def __init__(self,dim):
		super(downsampleConv,self).__init__()
		self.layer = nn.Conv2d(dim,2*dim,kernel_size=4,stride=2,padding=1)
	def forward(self,x):
		return self.layer(x)

class upsampleConv(nn.Module):
	def __init__(self,dim,out_dim):
		super(upsampleConv,self).__init__()
		self.layer = nn.ConvTranspose2d(dim,out_dim,kernel_size=2,stride=2)
	def forward(self,x):
		return self.layer(x)


class UFormer(nn.Module):
	def __init__(self,in_ch=3,emb_dim=32, encoder_depths = [1,2,8,8],
	      num_heads = [1,2,4,8,16,16,8,4,2],
	      win_size=8,
	      mlp_ratio = 4,
	      drop = 0.2,
	      modulators=False
		  ):
		super(UFormer,self).__init__()
		self.in_layer = conv3x3bn(in_ch,emb_dim)
		enc_list = []
		downsample_list = []
		j=0
		for e_d in encoder_depths:
			enc_list.append(
				LeWinBlock(emb_dim*(2**j),e_d,num_heads[j],mlp_ratio,drop,win_size),
			)
			downsample_list.append(downsampleConv(emb_dim*(2**j)))

			j+=1
		# j-=1
		self.encoder = nn.ModuleList(enc_list)
		self.downsample = nn.ModuleList(downsample_list)
		self.bottleneck = LeWinBlock(emb_dim*(2**j),e_d,num_heads[j+1],mlp_ratio,drop,win_size)
		k=j-1
		dec_list = []
		upsample_list = []
		in_emb_dim = emb_dim*(2**(j))
		encoder_depths.reverse()
		for e_d in encoder_depths:
			out_emb_dim = emb_dim*(2**(j-1))//2
			upsample_list.append(upsampleConv(in_emb_dim,out_emb_dim))

			dec_list.append(
				LeWinBlock(out_emb_dim*3,e_d,num_heads[k],mlp_ratio,drop,win_size,modulators),
			)
			in_emb_dim = out_emb_dim*3
			j-=1
			k+=1
		self.decoder = nn.ModuleList(dec_list)
		self.upsample = nn.ModuleList(upsample_list)
		self.out_layer = conv3x3bn(in_emb_dim+emb_dim,in_ch)

	def forward(self,x):
		x=self.in_layer(x)
		residuals = [x]
		for i in range(len(self.encoder)):
			x = self.encoder[i](x)
			residuals.append(x)
			x=self.downsample[i](x)
		x = self.bottleneck(x)
		for i in range(len(self.decoder)):
			x = self.upsample[i](x)
			x = self.decoder[i](torch.cat([x,residuals[-(i+1)]],1))
		x = self.out_layer(torch.cat([x,residuals[0]],1))
		return x


#copied from the official repo
#########
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

#########