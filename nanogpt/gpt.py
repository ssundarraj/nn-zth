import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337+1)

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = (
    "cuda" if torch.cuda.is_available()
    # else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"using device: {device}")
eval_iters = 200
n_embd = 32
# ------------

'''
going to train a char level model on shakespeare's works
reference code for video: https://github.com/karpathy/ng-video-lecture
finished code: https://github.com/karpathy/build-nanogpt
'''

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


'''
we train the transformer on chunks of the dataset, not the entire dataset (block_size/context_len)
the transformer makes predictions at all the next chars for given block size simultaneously

x has batch_size rows
for each row, there are block_size examples we can use to train the model
how this works is that, for input x[b][0] -> y[b][0]
for input x[b][0,1] -> y[b][1]
for input x[b][0,1,2] -> y[b][2]
... etc

we do this because we can train on much more data and also we can run inference on a single token
'''
def dbg_input_output(): # to understand the shape of input/outputs
    xb, yb = get_batch('train')
    print(xb.shape, yb.shape)
    print(xb)
    print(yb)
    print('----')
    print('----')

    for b in range(batch_size):
        print('----batch:', b)
        print(xb[b])
        print(yb[b])
        for t in range(block_size):
            context = xb[b, :t+1]
            target = yb[b, t]
            print (f"input {context} target {target}")
    return xb.to(device), yb.to(device)
# dbg_input_output()


class Head(nn.Module):
    tril: torch.Tensor

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # no training for buffers
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        # wei - affinities
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # normalization
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) 
        # aggregate
        v = self.value(x) 
        out = wei @ v
        return out

'''
attention paper also has cross attention to an encoder which we aren't implementing here
the paper also has feed forward after the multi head attention and the entire thing is repeated
feed fwd is an MLP

gives the model more time to "think" about the attention
this is per token/node
'''
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # multiply by 4 here bc paper does it but why?
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection
        )

    def forward(self, x):
        return self.net(x)


# kind of like group convolution
class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        # proj learns how to mix head outputs together before returning them to the residual
        # stream
        self.proj = nn.Linear(n_heads * head_size, n_embd)

    def forward(self, x):
        # run all the heads and concat the last dim
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

'''
NN getting big so training not great, need to optimize

1/ 
skip/residual connections. diagrams:
https://medium.com/data-science/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
skip conn with addition from the previous features
see second diagram:
computation happens from top to bottom
fork off from pathway, do some computation and then join back using addition

useful bc addition distributes gradients equally from all branches
"gradient superhighway" that goes from supervision to input
residual blocks are initialized in the beginning such that they contribute very little 
  but we don't do this yet
during optimization they come online over time
dramatically helps
'''
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size) 
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(x)
        x = x + self.ffwd(x)
        return x



class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off logits for the next token from the lookup table
        # actually, now  we use an embedding table in between
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # each position has its own embedding vector
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)

        # Blocks of self attn
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )

        # language modeling head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # B(atch), T(ime), C(hannel); C = n_embd
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) # T,C
        x = tok_emb + pos_emb # B,T,C -- broadcasted across batch

        x = self.blocks(x)

        logits = self.lm_head(x) # B, T, C; C = vocab_size

        if targets is None: 
            loss = None
        else:
            # pytorch expects B,C,T
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # Stretch the inputs out so the second dim is C
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens): 
        # idx -> (B,T) (current context)
        # At this point B is always 1 because we only forward one batch
        for _ in range(max_new_tokens):
            # process block_size max at a time so our positional encoding table doesn't run out of
            # scope
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond) # (B, T, C)
            # we get logits for all items in the current block (T) but
            # we only focus only on the last time step since we need the next char
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel().to(device)

# gets loss for multiple batches of each of eval and train splits
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
          X, Y = get_batch(split)
          logits, loss = model(X, Y)
          losses[k] = loss.item()
      out[split] = losses.mean()
  model.train()
  return out


def dbg_generate_dims(): # use with smaller block_size to understand what's going on in generate
    xb, yb = dbg_input_output()

    logits, loss = model(xb, yb)

    print(logits.shape)
    print(loss)

    idx = torch.zeros((1,1), dtype = torch.long)
    print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))
# dbg_generate_dims()

'''
Mathematical trick for efficient self-attention impl

we have 8 tokens in a batch that don't talk to each other but we need to couple them
we need to have the token at ith location talk to the prev tokens (i-1, i-2, ...)
simplest way is to average the prev tokens (incl token i)
    becomes feature vec that summarizes token i in context 
    quite lossy, eg loses spatial arrangement - will deal with this later
'''
def dbg_self_attn_math_trick():
    torch.manual_seed(11)
    B, T, C = 4, 8, 2
    x = torch.randn(B,T,C)
    print(x.shape)

    # inefficient way::
    # We want x[b, t] = mean_{i<=t} x[b, i]
    xbow = torch.zeros((B, T, C)) # bow = bag of words
    for b in range(B):
      for t in range(T):
          xprev = x[b, :t+1]  # (t, C)
          xbow[b, t] = torch.mean(xprev, 0)

    print(x[0][0]);print(xbow[0][0]) # these should be equal bc we avg only one token
    print('--')
    print(x[0][1]);print(xbow[0][1]) # NOT be equal, xbow is avg of first and second token

    # more efficient way ::

    def dbg_tril_mean():
        # tril workings + mat mul
        a = torch.tril(torch.ones(3,3)) # lower triangular ones
        b = torch.randint(0, 10, (3,2)).float()
        c = a @ b
        print(a)
        print(b)
        # each row of c has the running sum for each batch
        print(c)
        print(torch.equal(c[0], b[0]))
        print(torch.equal(c[1], b[:2].sum(dim=0)))
        print(torch.equal(c[2], b[:3].sum(dim=0)))

        # prev was sum, this is mean
        a = torch.tril(torch.ones(3,3)) # lower triangular ones
        a = a / torch.sum(a, 1, keepdim=True) # <---- mean here bc the ones are now mean
        b = torch.randint(0, 10, (3,2)).float()
        c = a @ b
        print(a)
        print(b)
        print(c)
        print(torch.allclose(c[0], b[0]))
        print(torch.allclose(c[1], b[:2].mean(dim=0)))
        print(torch.allclose(c[2], b[:3].mean(dim=0)))
    # dbg_tril_mean()

    # now with the B, T, C example above
    # i.e. do the above for the entire batch
    wei = torch.tril(torch.ones(T, T))
    wei = wei / wei.sum(1, keepdim=True)
    print(wei)

    # (T,T) @ (B, T , C) --> (B,T,T) @ (B,T,C)  -- torch adds the dim
    # (B,T,T) @ (B,T,C) -- multiplies for each batch --> (B,T,C)
    xbow2 = wei @ x
    print(torch.allclose(xbow, xbow2))

    # V3 xbow:
    # uses torch.softmax
    # We will use this version bc weights start at zeros
    # this is a representation of "interaction strength" / "affinity"
    # we can train on the wei matrix by training key, value, query
    tril = torch.tril(torch.ones(T, T))
    wei = torch.zeros((T,T))
    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim = 1) # same as prev wei
    xbow3 = wei @ x
    print(torch.allclose(xbow2, xbow3))
# dbg_self_attn_math_trick()

def dbg_self_attn_head():
    torch.manual_seed(45)
    B, T, C = 4,8,32
    # C comes from token embeddings & positional_embedding_table
    x = torch.randn(B,T,C)
    tril = torch.tril(torch.ones(T,T))

    # CRUX OF ATTENTION:
    # diff tokens will find diff other tokens more important
    # every token will emit 2 vectors: query + key
    # query = "what am I looking for"
    # key = "what do I contain"
    # affinities: dot product of query x key (wei)

    # self attn head
    head_size = 16 # hyper param
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)
    # each token produces k, q as discussed above
    k = key(x)   # B,T,head_size
    q = query(x) # same shape
    # communcation between diff tokens via dot product, instead of torch.zeros
    wei = q @ k.transpose(-2,-1) # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
    # For every row of B (batch) we have T,T affinity matrix

    # before mask + softmax
    print('wei[0] before mask:\n', wei[0])
    wei = wei.masked_fill(tril == 0, float('-inf'))
    print('wei[0] with mask:\n', wei[0])
    wei = F.softmax(wei, dim =-1) 
    # first batch to viz
    # each item is the ith token. for token 0, only the first element should be non-zero
    # for second, first 2 and so on
    print('wei[0] with softmax:\n', wei[0])

    '''
    we use v to access wei
    we can think of `x` as private to this token
    for a single head, for an item in batch b, and token at pos t:
        x[b, t]  = its current 32-number representation
        q[b, t]  = “what information am I looking for?”
        k[b, t]  = “what kind of information do I offer?”
        v[b, t]  = “if another token chooses me, what information do I send it?”

    The whole flow is:
    x
    ├─ query(x) → decides what each token wants
    ├─ key(x)   → decides what each token can be matched on
    └─ value(x) → decides what each token sends

    query · key → wei (who should listen to whom?)
    mask + softmax → normalized causal attention weights
    wei · value → out (the retrieved, mixed information)
    '''
    v = value(x) 
    out = wei @ v
    print(out.shape) # B,T,head_size
    '''
    1/ 
    attn is communcation mechanism
    can think of it as nodes in a DAG. every node has info that is a weighted sum of nodes
    pointing to it
    in our case (self attn):
        first node is pointed to by itself
        second pointed to by first and itself
        third by second, first, itself
        ... etc

    2/
    there's no notion of space in the attention mechanism
    so we need to introduce additional positional information
    diff from convolution bc the layout is predefined

    3/ 
    elements across batches don't talk to each other
    we use batches to process things in parallel

    4/
    in language modeling we don't allow future tokens to communicate with past tokens
    but not necessary to be the case
    eg for sentiment analysis with transformer you might want all tokens to talk to each other
    we will delete the masked fill stuff (aka _encoder block_)
    here we are using _decoder block_
    attn supports arbitrary communcation

    5/ 
    there is smth called cross attn
    in encoder-decoder transformers, queries are produced from X but keys,values can be from a diff
    source
    eg translation?

    6/
    Attention is all you need paper does "scaled attention"
    important normalization
    wei.var() is on the order of head_size
    we do wei = wei * (1/sqrt(head_size))
    lowers wei.var() to ~1
    if wei has high variance (big numbers), softmax will converge to one-hot vectors
    '''
# dbg_self_attn_head()


# In make more we use stochastic descent but Adam optimizer works better
# We can set lr to 1e-3 (quite high) bc our model is small
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: {losses}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

ctx = torch.zeros((1,1), dtype = torch.long).to(device)
print(decode(model.generate(ctx, max_new_tokens=1000)[0].tolist()))

