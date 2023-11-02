import torch

def open_file (file_name):
    try:
        with open (file_name, 'r') as file:
            text = file.read()
            file.close()

            return text
    except FileNotFoundError:
        print('ERROR: File not found')
        return []


def main():

    # get our file read
    file_name = 'input.txt'
    our_text = open_file(file_name)
    print('Length of input data text in chars: ', len(our_text))

    # get our char base for codebook
    chars = sorted(list(set(our_text)))
    vocab_size = len(chars)
    print('Characters include: ', ''.join(chars))
    print('Vocab Size: ', vocab_size)

    # setup character encoder and decoder
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s] # get string, output ints
    decode = lambda l: ''.join([itos[i] for i in l]) # get ints output string

    # test encode decode
    print(encode('hii there'))
    print(decode(encode('hii there')))

    # encode input.txt and package into a torch Tensor object.
    data = torch.tensor(encode(our_text), dtype=torch.long)
    print(data.shape, data.dtype)
    # print(data[:1000])  # print the first 1000 chars as ints from data.

    # setup test train split (90% train, 10% test)
    n = int(0.9 * len(data))  # gives us the value that's at the 90% mark of dataset
    train_data = data[:n]
    test_data = data[n:]  # will help test for overfitting

    # set chunk length for training (aka Blocksize)
    block_size = 8
    print(train_data[:block_size + 1])

    # explaining how each blocksize trains on element and group of elements
    x = train_data[:block_size]
    y = train_data[1:block_size + 1]
    for t in range(block_size):
        context = x[:t+1]
        target = y[t]
        print(f'when input is {context} target is {target}')

    # set batchs and batch size
    torch.manual_seed(1337)
    batch_size = 4  # num of ind seqs  processed in parallel
    block_size = 8  # max context length for predictions

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
        return x, y


main()
