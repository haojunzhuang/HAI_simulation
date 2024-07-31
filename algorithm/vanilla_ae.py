from utils import *

# Component for network:
# 	-normalize before passing in
# 	-end channel should be num_class
# 	-softmax with specified dim
# 	-Loss: CE (w/ or w/o logit depend on whether softmax)
# 	-softmax + argmax when generation/eval

# TODO: train function
def train():
    pass

# TODO: Eval function
def eval():
    pass

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    print("Start Preparing Data")
    X = torch.load('data/observed_matrices.pt')
    Y = torch.load('data/real_matrices.pt')
    
    pass