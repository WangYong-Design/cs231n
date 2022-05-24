import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..transformer_layers import *


class CaptioningTransformer(nn.Module):
	"""
		A CaptioningTransformer produces captions from image features using a
		Transformer decoder.

		The Transformer receives input vectors of size D, has a vocab size of V,
		works on sequences of length T, uses word vectors of dimension W, and
		operates on minibatches of size N.
	"""
	def __init__(self,input_dim,wordvec_dim,num_heads,num_layers,
				 word_to_idx,dropout=0.1,max_length=30):
		"""
		Construct a new CaptioningTransformer instance.

		Inputs:
		- word_to_idx: A dictionary giving the vocabulary. It contains V entries.
		  and maps each string to a unique integer in the range [0, V).
		- input_dim: Dimension D of input image feature vectors.
		- wordvec_dim: Dimension W of word vectors.
		- num_heads: Number of attention heads.
		- num_layers: Number of transformer layers.
		- max_length: Max possible sequence length.
		"""
		super().__init__()

		V = len(word_to_idx)
		self._null = word_to_idx["<NULL>"]
		self.start = word_to_idx.get("<START>",None)
		self.end = word_to_idx.get("<END>", None)

		self.visual_projection = nn.Linear(input_dim,wordvec_dim)

		self.embedding = nn.Embedding(V,wordvec_dim, padding_idx=self._null)

		self.position_encoding = PositionalEncoding(embed_dim = wordvec_dim,dropout=dropout,max_len=max_length)

		decoder_Layer = TransformerDecoderLayer(embed_dim = wordvec_dim,num_heads = num_heads)
		self.TransformerDecoer = TransformerDecoder(num_layers,decoder_Layer)

		self.apply(self._initialization)

		self.output_layer = nn.Linear(wordvec_dim,V)


	def _initialization(self,module):
		"""
			Initialize the weights of the network.
		"""
		if isinstance(module,(nn.Linear,nn.Embedding)):
			module.weight.data.normal_(mean = 0.0,std = 0.02)
			if isinstance(module,nn.Linear) and module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module,nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)

	def forward(self,captions,features):
		"""
		Given image features and caption tokens, return a distribution over the
		possible tokens for each timestep. Note that since the entire sequence
		of captions is provided all at once, we mask out future timesteps.

		Inputs:
		- features: image features, of shape (N, D)
		- captions: ground truth captions, of shape (N, T)

		Returns:
		- scores: score for each token at each timestep, of shape (N, T, V)
		"""

		N, T = captions.shape

		# Project image features into the same dimension as the text embeddings.
		# shape: [N, D] -> [N, W] -> [N, 1, W]
		memory = self.visual_projection(features).unsqueeze(1)

		# Embed the captions.
		# shape: [N, T] -> [N, T, W]
		caption_embeddings = self.embedding(captions)
		caption_embeddings = self.position_encoding(caption_embeddings)

		# An additive mask for masking the future (one direction).
		# shape: [T, T]
		tgt_mask = torch.tril(torch.ones(T, T,
										 device=caption_embeddings.device,
										 dtype=caption_embeddings.dtype))

		# Apply the Transformer decoder to the caption, allowing it to also
		# attend to image features.
		decoder_out = self.TransformerDecoer(x = caption_embeddings,memory = memory,
											 mask_fill = tgt_mask)

		# Project to scores per token.
		# shape: [N, T, W] -> [N, T, V]
		scores = self.output_layer(decoder_out)

		return scores

	def sample(self,features,max_length = 30):
		"""
		Given image features, use greedy decoding to predict the image caption.

		Inputs:
		- features: image features, of shape (N, D)
		- max_length: maximum possible caption length

		Returns:
		- captions: captions for each example, of shape (N, max_length)
		"""
		N,D = features.shape

		with torch.no_grad():

			features = torch.Tensor(features)

			# Create an empty captions tensor (where all tokens are NULL).
			captions = self._null * np.ones((N,max_length),dtype = np.int32)

			# # Create a partial caption, with only the start token.[N,1]
			partial_caption = torch.tensor(self.start * np.ones((N,1)),dtype = torch.int32)

			for t in range(max_length):
				# Predict the next token (ignoring all other time steps).
				scores = self.forward(features = features,captions = partial_caption)
				scores = scores[:,-1,:]

				# Choose the most likely word ID from the vocabulary.
				# [N, V] -> [N]
				word = torch.argmax(scores,dim = 1)

				# Update our overall caption and our current partial caption.
				captions[:,t] = word.numpy()
				word = word.unsqueeze(1)
				partial_caption = torch.cat([partial_caption,word],dim = 1)

		return captions



class TransformerDecoderLayer(nn.Module):
	"""
	A single layer of a Transformer decoder, to be used with TransformerDecoder.
	"""
	def __init__(self,embed_dim,num_heads,dim_feedforward=2048,dropout=0.1):
		super(TransformerDecoderLayer, self).__init__()
		"""
			Construct a TransformerDecoderLayer instance.
			Inputs:
			- input_dim: Number of expected features in the input.
			- num_heads: Number of attention heads
			- dim_feedforward: Dimension of the feedforward network model.
			- dropout: The dropout value.
		"""
		self.self_attn = MultiHeadAttention(embed_dim,num_heads,dropout)
		self.cross_attn = MultiHeadAttention(embed_dim,num_heads,dropout)

		self.layer1 = nn.Linear(embed_dim,dim_feedforward)
		self.layer2 = nn.Linear(dim_feedforward,embed_dim)

		self.dropout  = nn.Dropout(p = dropout)
		self.dropout1 = nn.Dropout(p = dropout)
		self.dropout2 = nn.Dropout(p = dropout)
		self.dropout3 = nn.Dropout(p = dropout)

		self.norm1 = nn.LayerNorm(embed_dim)
		self.norm2 = nn.LayerNorm(embed_dim)
		self.norm3 = nn.LayerNorm(embed_dim)

		self.activation = nn.ReLU()

	def forward(self,x,memory,mask_fill):
		"""
			Implement forward process for Transformer DecoderLayer
			Input:
			- x : the sequence to the decoder layer.shape (N,S,E)
			- memory : the sequence from the last layer of the encoder.shape (N,T,E)
			- mask_fill : the parts of the target sequence to mask, of shape (T, T)

			Ouput:
			- output : the Transformer features, of shape (N, T, E)
		"""

		# Perform self-attention on the target sequence (along with dropout and
		# layer norm).
		embed = self.self_attn(query = x,key = x,value = x,attn_mask = mask_fill)
		x = x + self.dropout1(embed)
		x = self.norm1(x)

		# Attend to both the target sequence and the sequence from the last
		# encoder layer.
		embed = self.cross_attn(query = x,key = memory,value = memory)
		x = x + self.dropout2(embed)
		x = self.norm2(x)

		# Pass
		embed = self.layer2(self.dropout(self.activation(self.layer1(x))))
		x = x + self.dropout3(embed)
		output = self.norm3(x)

		return output

def copyModule(model,N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(model) for _ in range(N)])

class TransformerDecoder(nn.Module):
	"""
	TransformerDecoder class
	"""
	def __init__(self,N,module):
		super(TransformerDecoder, self).__init__()
		self.num_layer = N
		self.layer = copyModule(module,N)

	def forward(self,x,memory,mask_fill = None):
		"""
		Implement forward pass
		"""
		for model in self.layer:
			x = model(x,memory,mask_fill)

		return x






