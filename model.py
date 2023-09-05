import torch
import torch.nn as nn
import math


class conv3x3bn(nn.Module):
	def __init__(self,in_ch,out_ch,groups=1,act_layer=nn.LeakyReLU):
		"""
		 @brief Initialize Conv2D 3x3 convolutional module with batch normalization
		 @param in_ch Number of input channels.
		 @param out_ch Number of output channels.
		 @param groups Number of groups for convolution
		 @param act_layer Activation layer.
		"""
		super(conv3x3bn,self).__init__()
		self.layer = nn.Sequential(
                        nn.Conv2d(in_ch,out_ch,kernel_size=3,groups=groups,stride=1,padding=1),
                        nn.BatchNorm2d(out_ch),
                        act_layer()
		)
	def forward(self,x):
		"""
		 @brief Forward pass of the module.
		 @param x input data of shape ( nb_samples self. input_dim )
		 @return output of shape ( nb_samples self. output_dim )
		"""
		return self.layer(x)

class conv1d1x1(nn.Module):
	def __init__(self,in_ch,out_ch,act_layer = nn.GELU):
		"""
		 @brief Initialize Conv2D 1x1 convolutional module.
		 @param in_ch Number of input channels. 
		 @param out_ch Number of output channels.
		 @param act_layer Activation layer. 
		"""
		super(conv1d1x1,self).__init__()
		self.layer = nn.Sequential(
			nn.Conv1d(in_ch,out_ch,kernel_size=1,stride=1,padding=0),
			act_layer()
		)
	def forward(self,x):
		"""
		 @brief Forward pass of the module.
		 @param x input data of shape ( nb_samples self. input_dim )
		 @return output of shape ( nb_samples self. output_dim )
		"""
		return self.layer(x)

class Leff(nn.Module):
	def __init__(self,in_ch,hidden_ch,out_ch):
		"""
		 @brief Initialize Leff module.
		 @param in_ch Number of input channels.
		 @param hidden_ch Number of hidden channels. It is the output channel of the convolutional layer.
		 @param out_ch Number of output channels. It is the output channel
		"""
		super(Leff,self).__init__()
		self.conv1 = conv1d1x1(in_ch,hidden_ch)
		self.conv2 = conv3x3bn(hidden_ch,hidden_ch,hidden_ch,nn.GELU)
		self.conv3 = conv1d1x1(hidden_ch,out_ch)

	def forward(self,x):
		"""
		 @brief Forward pass of the module.
		 @param x input tensor of shape [ batch_size emb c ]
		 @return output tensor of shape [ batch_size emb c ]
		"""
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
		"""
		 @brief Initialize the LeWin module.
		 @param emb_dim dimension of the embeddings ( neurons )
		 @param hidden_ch number of channels in the hidden units
		 @param win_size size of the window used to train the network
		 @param num_heads number of heads in the network ( 1 - 4 )
		 @param drop whether to drop features or not ( 0 = do not drop 1 = drop )
		 @param modulators whether to use modulators or not
		"""
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
		"""
		 @brief Add a modulator to an input. By default this is a no - op.
		 @param x The input to add the modulator to.
		 @param mod The modulator to add. If None no modification is performed.
		 @return The input with the modulator added to it or x if mod is None. Note that x is modified in - place
		"""
		return x if mod is None else x+mod

class LeWinBlock(nn.Module):
	def __init__(self,emb_dim,num_layer,num_heads,mlp_ratio,drop,win_size,modulators=False):
		"""
		 @brief Initialize LeWinBlock module.
		 @param emb_dim dimension of the embeddings to be used
		 @param num_layer number of LeWin module blocks
		 @param num_heads number of heads in the neural network
		 @param mlp_ratio mlp ratio of the hidden units
		 @param drop whether to drop hidden units when training or not
		 @param win_size size of the sliding window for the encoder
		 @param modulators whether to use modulators or not
		"""
		super(LeWinBlock,self).__init__()
		hidden_dim = emb_dim*mlp_ratio
		self.win_size=win_size
		self.layer = nn.ModuleList(
			[
				LeWin(emb_dim,hidden_dim,win_size,num_heads,drop,modulators)
			 for __ in range(num_layer)]
		)

	def forward(self,x):
		"""
		 @brief Forward pass of the module
		 @param x The input to the module. Must be a list of tensors.
		 @return The output of the module. It is a list of tensors in the same order as the input
		"""
		x = self.patch_process(x)
		for l in self.layer:
			x= l(x)
		x = self.depatch_process(x)
		return x

	def patch_process(self,x):
		"""
		 @brief Patches a 2D image by reshaping to windows.
		 @param x image to be patched.
		 @return patched image as a tensor of shape [ B C H W ] where H and W are the window size
		"""
		b,c,h,w = x.shape

		x = x.view(b,c,h//self.win_size,self.win_size,w//self.win_size,self.win_size)
		x= x.flatten(2)

		return x.permute(0,2,1)

	def depatch_process(self,x):
		"""
		 @brief Depatch the image reshaping to the original size from windows
		 @param x tensor of shape ( b emb c )
		 @return tensor of shape ( b emb c hw ) with hw = sqrt ( emb ) permuted to 2
		"""
		b,emb,c = x.shape
		hw = int(math.sqrt(emb))
		x = x.permute(0,2,1).view(b,c,hw,hw)
		return x

class downsampleConv(nn.Module):
	def __init__(self,dim):
		"""
		 @brief Initialize downsample convolution module.
		 @param dim number of channels
		"""
		super(downsampleConv,self).__init__()
		self.layer = nn.Conv2d(dim,2*dim,kernel_size=4,stride=2,padding=1)
	def forward(self,x):
		"""
		 @brief Forward pass of module.
		 @param x input data of shape ( nb_samples self. input_dim )
		 @return output of shape ( nb_samples self. output_dim )
		"""
		return self.layer(x)

class upsampleConv(nn.Module):
	def __init__(self,dim,out_dim):
		"""
		 @brief Initialize upsample convolution module.
		 @param dim dimension of the input
		 @param out_dim dimension of the output
		"""
		super(upsampleConv,self).__init__()
		self.layer = nn.ConvTranspose2d(dim,out_dim,kernel_size=2,stride=2)
	def forward(self,x):
		"""
		 @brief Forward pass of the module
		 @param x input data of shape ( nb_samples self. input_dim )
		 @return output of shape ( nb_samples self. output_dim )
		"""
		return self.layer(x)


class UFormer(nn.Module):
	def __init__(self,in_ch=3,emb_dim=32, encoder_depths = [1,2,8,8],
	      num_heads = [1,2,4,8,16,16,8,4,2],
	      win_size=8,
	      mlp_ratio = 4,
	      drop = 0.2,
	      modulators=False
		  ):
		"""
	       @brief Initialize UFormer. In this function you need to set the number of channels and the number of encoder layers.
	       @param in_ch Number of input channels. Default is 3
	       @param emb_dim Embedding dimension. Default is 32
	       @param encoder_depths List of encoder depths
	       @param num_heads
	       @param win_size
	       @param mlp_ratio
	       @param drop
	       @param modulators
	      """
		super(UFormer,self).__init__()
		self.in_layer = conv3x3bn(in_ch,emb_dim)
		enc_list = []
		downsample_list = []
		j=0
		# Generates a LeWinBlock for each encoder depth.
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
		# The encoder depths are reversed and used to be the decoder depths.
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
		"""
		 @brief Forward pass of the network.
		 @param x The input to the network. Must be a Tensor of shape [ batch_size in_channels ].
		 @return The output of the network after upsampling and downsampling.
		"""
		x=self.in_layer(x)
		residuals = [x]
		# Add the residuals to the residuals list.
		for i in range(len(self.encoder)):
			x = self.encoder[i](x)
			residuals.append(x)
			x=self.downsample[i](x)
		x = self.bottleneck(x)
		# Compute the best guess of the decoder.
		for i in range(len(self.decoder)):
			x = self.upsample[i](x)
			x = self.decoder[i](torch.cat([x,residuals[-(i+1)]],1))
		x = self.out_layer(torch.cat([x,residuals[0]],1))
		return x

###############################
#COPIED FROM THE OFFICIAL REPO#
###############################
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        """
         @brief Initialize Charbonnier Loss.
         @param eps Epsilon to use for
        """
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        """
         @brief Computes Loss between two tensors.
         @param x Tensor with shape [ batch_size image_size ]
         @param y Tensor with shape [ batch_size image_size ]
         @return A tensor with shape [ batch_size num_features ] where each element is the Loss between x and y
        """
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

###############################
#COPIED FROM THE OFFICIAL REPO#
###############################