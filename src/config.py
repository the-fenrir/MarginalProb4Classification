class Config:
    def __init__(self, args, model, tokenizer):
        self.model_id = args.model_id
        self.device = args.device

        self.dataset = args.dataset
        self.num_samples = args.num_samples

        self.labels = ["unsafe"] + [f"S{i}" for i in range(1, 15)]
        if self.model_id == "meta-llama/Meta-Llama-Guard-2-8B":
            self.labels = ["unsafe"] + [f"S{i}" for i in range(1, 12)]

        self.eos_token_ids = model.config.eos_token_id
        if isinstance(self.eos_token_ids, int):
            self.eos_token_ids = [self.eos_token_ids]