import os
import torch

class Config(object):
    def __init__(self, args, log_path=None):
        self.args = args

        # ===== hyper parameters =====
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_gpu = args.multi_gpu
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.clause_max_len = args.clause_max_len
        self.attention_head = args.attention_head

        self.hidden_dropout_prob = 0.1
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-6
        self.lr = args.lr
        self.bert_lr = args.bert_lr
        self.warmup_proportion = args.warmup_proportion
        self.seed = args.seed
        self.max_norm = args.max_norm
        self.init_mode = args.init_mode

        # ===== paths =====
        self.root = "./"
        self.data_path = args.data_path
        self.model_dir = args.model_path
        self.model_name = args.model_name

        # ★★★ 使用传入 log_path
        self.log_dir = log_path if log_path is not None else "./log"

        # 文件名保持作者格式
        self.log_save_name = f"LOG_{args.model_name}_BERTLR_{self.bert_lr:e}_LR_{self.lr:e}_BS_{self.batch_size}"

        # prefixes
        self.train_prefix = args.train_prefix
        self.dev_prefix = args.dev_prefix
        self.test_prefix = args.test_prefix

        # others
        self.period = args.period
        self.test_epoch = args.test_epoch

        # debug
        self.debug = args.debug
        if self.debug:
            self.dev_prefix = self.train_prefix
            self.test_prefix = self.train_prefix
