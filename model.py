import torch  # PyTorch used for build and train the NN
import torch.nn as nn  # nn - neural network module and it contains ready made layers
import math

class TokenEmbedding(nn.Module): # inherits from nn.Module
    def __init__(self, vocab_size, d_model):   
        # vocab_size : Total number of tokens in vocabulary
        # d_model : size of embedding vector / dimension of each token representation
        super().__init__()  # call the constructor of nn.Module
        self.embedding = nn.Embedding(vocab_size, d_model)  # creates an embedding layer

    # defines how data flows through this layer
    def forward(self, x): # x: input tensor
        return self.embedding(x)  # passes input x through embedding layer and returns embedded tensor
    

# positional encoding using sinusoidal 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # creates a matrix of shape , each row = position : each column = embedding dimension
        position = torch.arange(0, max_len).unsqueeze(1)
        # torch.arange(0, max_len) create [0,1,2,...,4999] -> shape: (mex_len,)
        # .unsqueeze(1) add new dimension -> shape: (max_len, 1) 
        # so now -> [[0], [1],[2],...,[4999]]  : this represents position numbers

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )
        # torch.arange(0, d_model, 2) : create [0, 2, 4, ..., d_model-2] , only select the even dimensions 
        # (-math.log(10000.0) / d_model) : scaling factor 
        # torch.exp(...) -> div_term shape = (d_model/2,)
        # this controls how fast sine/cosine waves change 
        # Lower dimensions -> slow waves ,  higher dimensions -> fast waves

        # apply sine to even indices => PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        # 0::2 means start at index 0 and step by 2 -> (0, 2, 4 ...) 

        # apply cosine to odd indices => PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)
        # 1::2 means start at index 1 and step by 2 -> (1,3,5,7 ...)

        self.register_buffer("pe", pe.unsqueeze(0))   
        # register_buffer -> stores pe inside the modle but it not trainable and not update during backpro, it moves automatically to GPU with the model
        # positional encoding is fixed not learned
        # .unsqueeze(0) change shape (max_len, d_model) into (1, max_len, d_model) to match batch dimension later

    def forward(self, x):
        # x shape : (batch_size, seq_len, d_model)

        return x + self.pe[:, :x.size(1)]
        # x.size(1) get sequence length
        # self.pe[:, :x.size(1)] select only required positions
        # x + positional encoding
        # final output shape : (batch_size, seq_len, d_model)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0  # ensures d_model must be divisible by num_heads
        # because each head gets dk = d_model/num_heads

        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        self.qkv = nn.Linear(d_model, 3 * d_model)
        # use one linear layer that outputs 3*d_model and produces Query, Key, Value all at once
        self.fc = nn.Linear(d_model, d_model) # final projection affter attention

    def forward(self, x):
        B, T, C = x.shape  # B = batch size, T = sequence length, C = embedding dimension(d_model)

        qkv = self.qkv(x) # generate Q,K,V and shape becomes (B,T,3C)
        qkv = qkv.view(B, T, 3, self.num_heads, self.d_k) # reshape to separate Q,K,V and heads -> new shape (B,T,3,num_heads,dk)
        qkv = qkv.permute(2, 0, 3, 1, 4) # rearranges dimensions (3,B,heads,T,dk)
        # first dimension indexes: 0 -> Q, 1 -> K, 2 -> V

        Q, K, V = qkv[0], qkv[1], qkv[2] # each has shape: (B, heads, T, dk)

        # compute attention scores
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        #  Q: (B, heads, T, dk) , K: (B, heads, T, dk)
        # transpose last 2 dims of k : (B, heads, dk, T)
        # now multiply (B, heads, T, T) 
        # each token attends to every other token

        # causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device))
        # create lower traingular matrix
        # ex: 
        #    1 0 0
        #    1 1 0
        #    1 1 1
        # this is causal masking
        # meaning :- token cannot see future tokens and used in decoder transformers

        attn = attn.masked_fill(mask == 0, float("-inf")) # replace attention score with -inf where mask is 0
        # after softmax those positions become 0 probability

        attn = torch.softmax(attn, dim=-1) # convert scores in to probabilities
        # now: attn.shape = (B, heads, T, T) ; each row sums to 1

        # multiply with V
        out = attn @ V 
        # attn: (B, heads, T, T) , V: (B, heads, T, dk) , result -> (B, heads, T, dk)
        # each token becomes weighted sum of all value vectors

        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        # out = out.transpose(1, 2) = combine heads
        # before: (B, heads, T, dk) , after: (B, T, heads, dk)
        # .contiguous().view(B, T, C) = merge heads since heads * dk = C and final shape (B,T,C)

        return self.fc(out) # applies final transformation; output shape remains (B,T,C)
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff): # d_ff = hidden dimension inside the feed-forward network
        # in transformers dff is usually much larger than dmodel
        super().__init__()

        # create small NN
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), # shape transformation (B,T,dmodel) -> (B,T,dff)
            # expands the dimension , ex: d_model = 512, d_ff = 2048
            # higher dimension allows more expressive power and more complex transformations
            nn.ReLU(), # apply non-linearity 
            # without activation 2 liner layers collapse into one linear transformation and model would be too simple
            # relu allows learning complex patterns
            nn.Linear(d_ff, d_model) # reduces dimension back (B,T,dff) -> (B,T,dmodel)
        )

    def forward(self, x):
        return self.net(x)
        # passes input through Linear -> ReLU -> Linear and returns output shape (B,T,dmodel)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads) # this layer allows tokens to interact with each other
        self.ff = FeedForward(d_model, d_ff) # this processes each token independently

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # first sub layer: attention + residual + norm
        x = self.norm1(x + self.attn(x))
        # self.attn(x): compute attention output shape (B,T,dmodel)
        # x + self.attn(x): residual connection that mean output = input + sub_layer output
        # self.norm1(...) : apply LayerNorm after addition and this structure is called Post-LayerNorm architecture
        
        # second sub layer: FFN + residual + norm
        x = self.norm2(x + self.ff(x))
        # self.ff(x) : feed forward output: (B,T,dmodel)
        # residual: x + FFN(x)
        # layernorm: x = LayerNorm(x + FFN(x))

        return x # shape remains: (B,T,dmodel)
    

class MiniGPT(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            d_model=128, 
            num_head=4, 
            d_ff=256, 
            num_layers=4,
            block_size=128
        ):
        super().__init__()

        self.block_size = block_size

        self.embed = TokenEmbedding(vocab_size, d_model)  # (B,T) -> (B,T,dmodel)
        self.pos = PositionalEncoding(d_model)  # shape remains as (B,T,dmodel)

        # Transformer Blocks : create multiple transformer blocks and stacks them
        # num_layers = 4, create 4 blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, num_head, d_ff)
              # * : unpack the list into nn.Sequential ; x -> Block1 -> Block2 -> Blcok3 -> Block4
              for _ in range(num_layers)]
        )
        # each block has MHA, FF, Residuals and LayerNorm

        self.norm = nn.LayerNorm(d_model) # apply normalization after all blocks

        self.fc_out = nn.Linear(d_model, vocab_size) # convert (B,T,dmodel) -> (B,T,vocab_size)
        # because for language modeling must predict probability of each word in vocabulary
        # so for every token position, output logits over entire vocabulary 
    
    def forward(self, x):   # input shape: x =(B,T)
        x = self.embed(x)  # (B,T,dmodel)
        x = self.pos(x) # still (B,T,dmodel) but with positional info
        x = self.blocks(x) # shape remains (B,T,dmodel)
        x = self.norm(x) # normalize features
        logits = self.fc_out(x)
        return logits # final shape (B,T,vocab_size)
        # these are logits not probabilities yet
    
    # generates new tokens from trained model
    def generate(self, idx, max_new_tokens):
        # idx: current token sequence(input tokens)
        # max_new_tokens: how many new tokens to generate

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]  # limit context to block size
            # slef.block_size = maximum sequence length model can handle
            # take only the last block_size tokens
            # ex: block_size=6, idx=[1,4,78,5,34,95,10,23] => idx_coud=[78,5,34,95,10,23]

            logits = self(idx_cond)  # pass tokens into the model and get predication
            # output shape: (batch_size, sequence_length, vocab_size) => ex: (1, 6, 1000)
            # for each token position, the model predicts probabilities for all vocabulary tokens

            logits = logits[:, -1, :] # take predication of the last token
            # only care about predicting the next token
            # shape becomes: (batch_size, vocab_size)

            probs = torch.softmax(logits, dim=-1) # convert logits to probabilities

            next_token = torch.multinomial(probs, num_samples=1) # sample next token
            # instead of choosing highest probability token, sample randomply based on probabilities
            # EX: A=0.7, B=0.2, C=0.1 => sampling might produce A - most of the time, B - sometimes, C - rarely
            # this makes text more creative
            # output shape: (batch_size,1)

            idx = torch.cat((idx, next_token), dim=1) # append new token to sequence

        return idx # return final generated sequence
