import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtune.generation import sample

# the model based off of Karpathy's video Let's Build GPT
# https://www.youtube.com/watch?v=kCc8FmEb1nY

# B is used to denote batch size
# E is used to denote embedding dimension
# T is used to denote the sequence length

class Head(nn.Module):
    """A single attention head as described above"""
    def __init__(self, config):
        self.config = config

        super().__init__()

        # create the key, query, and value matrices
        # these are the K_0, Q_0, and V_0 from before
        self.key = nn.Linear(self.config.embed_d, self.config.head_size, bias=False)
        self.query = nn.Linear(self.config.embed_d, self.config.head_size, bias=False)
        self.value = nn.Linear(self.config.embed_d, self.config.head_size, bias=False)

        # create the attention mask.
        # this means that past tokens can't pay attention to future tokens.
        # we need this so that the transformer doesn't cheat if we're doing a prediction problem.
        # you can turn this off if there if you're not doing a prediction problem.

        # the mask is a lower triangular matrix of ones
        A = torch.tril(torch.ones(self.config.context_length, self.config.context_length))

        self.register_buffer('tril', A)

    def forward(self, x):
        B, T, E = x.shape

        k = self.key(x) # (B, T, E)
        q = self.query(x) # (B, T, E)

        # compute the attention scores (or "affinities")
        wei = q @ k.transpose(-2, -1) * E**-0.5 # (B, T, E) @ (B, E, T) -> (B, T, T)

        # apply the attention mask if required
        if self.config.use_mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)

        # apply softmax to ensure attention sums to 1
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, E)
        out = wei @ v # (B, T, T) @ (B, T, E) -> (B, T, E)
        return out

class MultiHeadAttention(nn.Module):
    """
      Combines multiple attention heads into one and
      concatenates the results. We do this because it
      trains better in practice, but mathematically this
      is equivalent to the single head version.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

        # create the attention heads
        self.heads = nn.ModuleList([Head(self.config) for _ in range(self.config.n_heads)])

        # an affine transformation
        self.proj = nn.Linear(self.config.embed_d, self.config.embed_d)

    def forward(self, x):
        # run the attention heads and concatenate their outputs
        output = torch.cat([h(x) for h in self.heads], dim=-1)

        # apply an affine transformation
        return self.proj(output)

class FeedForward(nn.Module):
    """
      A single layer feed forward block.
      Basically a tiny multi-layer perceptron.
      See the previous notebook for a better explanation.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.net = nn.Sequential(
            # do an affine transformation
            # we go to 4 times the embedding dimension for empirical reasons
            nn.Linear(self.config.embed_d, self.config.embed_d * 4),
            # apply non-linearity
            nn.ReLU(),

            nn.Linear(self.config.embed_d * 4, self.config.embed_d), # projection layer
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
      An attention block followed by an MLP.
      This is the part in the box in the diagram.
      Via Karpathy: commmunication followed by computation.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # create an attention mechanisms
        self.sa = MultiHeadAttention(self.config)

        # create an MLP
        self.ffwd = FeedForward(self.config)

        # some normalisation for practial reasons
        self.ln1 = nn.LayerNorm(self.config.embed_d)
        self.ln2 = nn.LayerNorm(self.config.embed_d)

    def forward(self, x):
        # note: don't use +=, as this breaks things for subtle reasons

        # apply the attention mechanism
        x = x + self.sa(self.ln1(x))

        # apple the MLP
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # create the embedding tables
        self.token_embedding_table = nn.Embedding(self.config.vocab_size, self.config.embed_d)
        self.position_embedding = nn.Embedding(self.config.context_length, self.config.embed_d)

        # add the blocks (defined in the class above)
        self.blocks = [
            Block(self.config) for _ in range(self.config.n_blocks)
        ]
        self.blocks.append(nn.LayerNorm(self.config.embed_d))
        self.blocks = nn.Sequential(*self.blocks)

        # the output layer
        # projects the final vector down to the output dimension
        self.lm_head = nn.Linear(self.config.embed_d, self.config.vocab_size, bias=False)

    def forward(self, idx):
        # get batchsize and sequence length
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) #(B, T, E)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) #(T, E)

        x = tok_emb + pos_emb #(B, T, E)

        x = self.blocks(x) # apply a bunch of blocks (sa + feedforward) (B, T, E)

        logits = self.lm_head(x) # (B, T, vocab_size)

        """
          For generation problems (eg. gpt), we look
          at every token's logits for efficiency. For
          classification problems (eg. sentiment analysis),
          we only look at the last row.
        """
        if not self.config.use_mask:
            # grab the last row of logits
            logits = logits[:, -1, :] # becomes (B, E)

        return logits

    def generate(self, initial_sequence, new_token_count, temperature):
        """
          Generates a sequence for generation problems.

          new_token_count is the amount of tokens to generate.

          temperature controls how strongly the sampling prefers
          the tokens that the transformer likes. If temp is low,
          then sampling is basically argmax. If temp is high, then
          sampling is basically uniform over all tokens.
        """

        # check if the model is set up for autoregression
        if not self.config.use_mask:
            raise Exception("For generation problems use_mask must be set to true")

        self.eval()

        # create an initial input to the network
        current_input = torch.tensor(initial_sequence)

        # do the autoregression
        for x in range(new_token_count):
            # grab the logits from the last token
            logits = self(current_input.unsqueeze(0))[:, -1, :]

            # sample the next token
            chosen = sample(logits, temperature)

            # add it to the input tensor
            current_input = torch.cat((current_input, torch.tensor([chosen])))

        return current_input