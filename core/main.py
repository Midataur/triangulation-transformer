from configobject import ProblemConfig
from utilities import VOCABULARY
from tqdm import tqdm
import torch

torch.manual_seed(42)

CONFIG = ProblemConfig(
    modelname="new-format-1",
    embed_d=402,
    n_heads=6,
    n_blocks=4,
    learning_rate=3*(10**-4),
    batchsize=64,
    weight_decay=0.01,
    lr_factor=0.1,
    lr_patience=10,
    threshold=0.01,
    n_workers=0,
    vocab_size=len(VOCABULARY),
    context_length=68,
    use_mask=True,
    PATH="."
)

if __name__ == "__main__":
    print("Loading libraries...")
    from training import train
    train(CONFIG)