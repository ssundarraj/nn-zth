import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = (
    "cuda" if torch.cuda.is_available()
    # else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"using device: {device}")
eval_iters = 200
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


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # Pluck out a row from the embedding table for every item in the input
        logits = self.token_embedding_table(idx) # B(atch), T(ime), C(hannel)
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
          # get the predictions
          logits, _ = self(idx) # (B, T, C)
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


model = BigramLanguageModel(vocab_size).to(device)

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

