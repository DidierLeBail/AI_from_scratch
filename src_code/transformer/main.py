import torch
import torch.nn as nn
import torch.nn.functional as F

def init_tensor(*size, scale_dim=0):
	return (2 * torch.rand(*size, requires_grad=True) - 1) / torch.sqrt(torch.tensor(size[scale_dim]))

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
		batch_size=0
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
		print("d:", self.emb_dim)

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

if __name__ == "__main__":
	nb_heads = 5
	emb_dim = 10
	context_size = 7
	batch_size = 6

	print("t:", context_size)

	mhsa = MHSA(nb_heads, emb_dim, batch_size=batch_size)

	# dummy data
	if batch_size == 0:
		X = torch.rand(1, context_size, emb_dim)
	else:
		X = torch.rand(batch_size, 1, context_size, emb_dim)
	X_out_sub1 = mhsa.forward(X)
	print("X_out_sub1 size should be (t, d):", X_out_sub1.size())
