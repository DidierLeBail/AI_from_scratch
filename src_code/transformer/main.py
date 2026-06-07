import torch
import torch.nn as nn
import torch.nn.functional as F

def init_tensor(*size, scale_dim=0):
	return (2 * torch.rand(*size, requires_grad=True) - 1) / torch.sqrt(torch.tensor(size[scale_dim]))

class Embedding(nn.Module):
	"""
	Takes as input a sequence of one-hot encoded tokens and convert them into a sequence of vectors.
	The resulting embedding takes into account the position of the tokens inside the sequence.

	Sizes of the involved tensors (batch not considered):
	input X: (1, t, u)
	W_emb_tok: (u, d)
	X_emb_tok: (t, d)
	X_emb_pos: (t, d)

	Parameters
	----------
	emb_dim : int
		embedding dimension: this is the dimension of the space in which the vectorized tokens live
	vocab_size : int
		size of the vocabulary: this is the number of allowed tokens
	context_size : int
		length of the input sequence of tokens
	"""

	def __init__(self,
		emb_dim,
		vocab_size,
		context_size,
		N=10000,
		**kwargs
	):
		super().__init__()
		self.W_emb_tok = init_tensor(vocab_size, emb_dim)
		r = N ** (-2 / emb_dim)
		t = torch.arange(context_size)
		thetas = [t * r ** k for k in range(emb_dim // 2)]
		self.X_emb_pos = torch.transpose(
			torch.cat([
				torch.stack(
					(torch.sin(theta), torch.cos(theta))
				) for theta in thetas
			]),
			dim0=0,
			dim1=1
		)
		print("X_emb_pos size should be (t, d):", self.X_emb_pos.size())
	
	def forward(self, X):
		return torch.matmul(X, self.W_emb_tok) + self.X_emb_pos

class LayerNorm(nn.Module):
	"""
	Takes as input a tensor (t, d) and rescale each row by a gain and shift by a bias.
	"""
	def __init__(self,
		emb_dim,
		**kwargs
	):
		super().__init__()
		self.gain = init_tensor(emb_dim)
		self.bias = init_tensor(emb_dim)
	
	def _normalize(self, X):
		"""
		X is (t, d) and return a tensor of same size with each row of X
		being rescaled and shifted so that they have 0 mean and 1 variance
		"""
		sigma = torch.std(X, dim=1)
		mu = torch.mean(X, dim=1)
		return (X - mu) / sigma

	def forward(self, X):
		return self.bias + torch.mul(self.gain, self._normalize(X))

class MHSA(nn.Module):
	r"""
	Multi-head self-attention module.
	Sizes of the involved tensors (batch not considered):
	input X: (1, t, d)
	W_Q: (n, d, h)
	Q: (n, t, h)
	Z: (n, t, t)
	A: (n, t, t)
	R: (n, t, h)
	X_out: (t, d)
	W_out: (d, d)

	Parameters
	----------
	nb_heads : int
		the number of attention heads
	emb_dim : int
		the embedding dimension of the tokens
	
	Attributes
	----------
	attr : type
		pass
	
	References
	----------
	See [1]_ for the original paper, and [2]_ for a more detailed description.

	.. [1] Vaswani, Ashish, et al. "Attention is all you need."
	   Advances in neural information processing systems 30 (2017).

	.. [2] Chen, Zhe. "Attention is not all you need anymore."
	   arXiv preprint arXiv:2308.07661 (2023).

	Examples
	--------
	pass
	"""

	def __init__(self,
		nb_heads,
		emb_dim,
		batch_size=0,
		**kwargs
	):
		super().__init__()
		self.nb_heads = nb_heads
		self.emb_dim = emb_dim
		
		self.head_dim = self.emb_dim // self.nb_heads
		self.is_batch = 0
		sizes = [(self.nb_heads, self.emb_dim, self.head_dim), (self.emb_dim, self.emb_dim)]

		# stack the heads along the first dimension so that they are processed (trained and used) in parallel
		if batch_size != 0:
			self.is_batch = 1

		self.W_Q = init_tensor(*sizes[0], scale_dim = 1)
		self.W_K = init_tensor(*sizes[0], scale_dim = 1)
		self.W_V = init_tensor(*sizes[0], scale_dim = 1)

		self.W_out = init_tensor(*sizes[1], scale_dim = 0)

	def forward(self, X):
		print("n:", self.nb_heads)
		print("h:", self.head_dim)

		print("W_Q size should be (n, d, h):", self.W_Q.size())
		print("X size should be (1, t, d):", X.size())

		Q = torch.matmul(X, self.W_Q)
		print("Q size should be (n, t, h):", Q.size())
		K = torch.matmul(X, self.W_K)
		V = torch.matmul(X, self.W_V)

		Z = torch.matmul(
			Q,
			torch.transpose(K, dim0 = 1 + self.is_batch, dim1 = 2 + self.is_batch)
		) / torch.sqrt(torch.tensor(self.head_dim))
		A = F.softmax(Z, dim=2 + self.is_batch)
		R = torch.matmul(A, V)

		print("R size should be (n, t, h):", R.size())

		X_out = torch.flatten(
			torch.transpose(R, dim0 = self.is_batch, dim1 = 1 + self.is_batch),
			start_dim = 1 + self.is_batch
		)
		print("X_out size should be (t, d):", X_out.size())

		return torch.matmul(X_out, self.W_out)

class FFN(nn.Module):
	"""
	Feedforward network module for the encoder part of the Transformer architecture.
	"""
	def __init__(self,
		emb_dim,
		d_ffn,
		**kwargs
	):
		super().__init__()
		self.lin1 = torch.nn.Linear(emb_dim, d_ffn)
		self.lin2 = torch.nn.Linear(d_ffn, emb_dim)
	
	def forward(self, X):
		return self.lin2(F.relu(self.lin1(X)))

class Encoder(nn.Module):
	"""
	Implements the encoder part of the Transformer architecture.
	Takes as input a sequence of tokens, where a single token is masked, and returns the proba of the masked token.
	"""

	def __init__(self,
		drop_proba,
		nb_layers,
		**kwargs
	):
		super().__init__()
		self.drop_proba = drop_proba
		self.nb_layers = nb_layers

		self.emb = Embedding(**kwargs)
		self.att_subs = [MHSA(**kwargs) for _ in range(self.nb_layers)]
		self.norm_subs = [[LayerNorm(**kwargs), LayerNorm(**kwargs)] for _ in range(self.nb_layers)]
		self.ff_subs = [FFN(**kwargs) for _ in range(self.nb_layers)]

		self.last_norm = LayerNorm(**kwargs)
	
	def _dropout(self, X):
		return F.dropout(X, p=self.drop_proba)

	def forward(self, X):
		X_emb = F.dropout(self.emb(X), p=self.drop_proba)
		for layer in range(self.nb_layers):
			Y = X_emb + self._dropout(self.att_subs[layer](self.norm_subs[layer][0](X_emb)))

			X_emb = Y + self._dropout(self.ff_subs[layer](self.norm_subs[layer][1](Y)))
		return self.last_norm(X_emb)

class Test:
	def __init__(self):
		pass

	def encoder(self):
		config = {
			"nb_heads": 5,
			"emb_dim": 10,
			"context_size": 7,
			"batch_size": 0,
			"vocab_size": 16,
			"drop_proba": 0.2,
			"nb_layers": 1,
			"d_ffn": 25
		}
		enc = Encoder(**config)

		# dummy data (batch assumed)
		X = torch.rand(1, config["context_size"], config["vocab_size"])

		Y = enc(X)

if __name__ == "__main__":
	Test().encoder()

	exit()

	nb_heads = 5
	emb_dim = 10
	context_size = 7
	batch_size = 0
	vocab_size = 16

	print("t:", context_size)
	print("d:", emb_dim)

	# dummy data
	if batch_size == 0:
		X = torch.rand(1, context_size, vocab_size)
	else:
		X = torch.rand(batch_size, 1, context_size, vocab_size)

	emb = Embedding(emb_dim, vocab_size, context_size)
	layer = LayerNorm(emb_dim)

	Y = layer.forward(emb.forward(X))
	print("Y size should be (1, t, d):", Y.size())

	exit()

	mhsa = MHSA(nb_heads, emb_dim, batch_size=batch_size)
	X_out_sub1 = mhsa.forward(X)
	print("X_out_sub1 size should be (t, d):", X_out_sub1.size())
