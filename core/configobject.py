class ProblemConfig:
    def __init__(
        self,
        modelname,
        embed_d,
        n_heads,
        n_blocks,
        learning_rate,
        batchsize,
        weight_decay,
        lr_factor,
        lr_patience,
        threshold,
        n_workers,
        vocab_size,
        context_length,
        use_mask,
        PATH
    ):
        self.modelname = modelname
        self.embed_d = embed_d
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.learning_rate = learning_rate
        self.batchsize = batchsize
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.threshold = threshold
        self.n_workers = n_workers
        self.PATH = PATH
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.use_mask = use_mask

        assert self.embed_d % self.n_heads == 0

        self.head_size = self.embed_d // self.n_heads