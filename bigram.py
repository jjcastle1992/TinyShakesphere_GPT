import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent seqs get processed in parallel
block_size = 8   # what is the max context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ---------------

file_name = 'input.txt'

torch.manual_seed(1337)  # static seed for testing and troubleshooting

# data source to read from
with open(file_name, 'r') as file:
    text = file.read()

# get char base for codebook aka unique characters that occur in text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# map from chars to ints
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # get string, output ints
decode = lambda l: ''.join([itos[i] for i in l])  # get ints output str


# encode input.txt and package into a torch Tensor object.
data = torch.tensor(encode(text), dtype=torch.long)

# define test train splits (90% train, 10% test)
n = int(0.9 * len(data))  # give the value that's at 90% mark of dataset
train_data = data[:n]  # first 90% will be training, rest test
test_data = data[n:]  # will help test for overfitting

# data loading
def get_batch (split):
    # generate small batch of data inputs x and targets y
    data = train_data if split == 'train' else test_data
    # ix = 4 numbers, rand gen between 0 and len(data) - blk size
    # i.e. ix random offsets into the training set.
    # i.e. since blk_size = 4, then if a list contains [0, 2, 4, 6]
    # then it will make 4 slices data[0:4], data[2:6], data[4:8], data[6:10]
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    # y will just be offset by 1 so data[1:5], data[3:7], data[5:9], data[7:11]
    y = torch.stack([data[i + 1: i + block_size + 1]for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# estimate loss funct
@torch.no_grad()   # lets pyTorch know everything in this func should
# not call .backward() (AKA not doing backwards propogation) on.
# More efficient mem use (no storage of intermediate vars)
def estimate_loss():
    # eval mode useful when we have dropout layers, batch norm layers
    # instead of just embedding table
    out = {}
    model.eval()  # set model to eval mode
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range (eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # set model back to train mode
    return out

# Super simple Bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,
                                                  vocab_size)  # 65 x 65 embedding table

    # network layer
    def forward(self, idx, targets=None):
        # idx and targets are both (Batch, Time) tensor of ints (i.e. 4, 8) tensor of ints
        logits = self.token_embedding_table(
            idx)  # (Batch, Time, Channel) where Batch = 4, Time = 8 (max context length), Channel = 65 aka vocab size

        if targets is None:
            loss = None
        else:
            # modify logits to satisfy PyTorch requirement to have dims in BCT format instead of BTC
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss  # aka predictions

    # This model is very simple. Tokens do not talk to each other.
    # Given the generated context (for example 'To be or' only looks at
    # the very last character (in this case the 'r' in 'or' to make a predict about what comes next.
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # get predictions (calls forward)
            logits, loss = self(idx)
            # focus on last time step
            logits = logits[:, -1, :]  # go to B, C from BTC
            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1)  # also BC
            # sample from the distribution
            idx_next = torch.multinomial(probs,
                                         num_samples=1)  # B,1 array (Batch by 1 array) (aka 4x1 array)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

# confirm GPU running
print('GPU Available: ', torch.cuda.is_available())

# create model object
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# declare training loop
for iter in range(max_iters):
    # on first iteration, print untrained sample
    if iter == 0:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print('First Untrained Generation:',
              decode(m.generate(context, max_new_tokens=500)[0].tolist()),
              '\n')

    # eval loss on train & test sets on regular basis (every 300 rounds)
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
    elif iter == (max_iters - 1):
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
    # get a sample batch of data
    xb, yb = get_batch('train')

    # eval loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(f'\nFinal Generation after {max_iters} steps: ',
      decode(m.generate(context, max_new_tokens=500)[0].tolist()))

