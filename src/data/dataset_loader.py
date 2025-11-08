from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self):
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.full_dataset = None

    def load_biasDPO(self, train_ratio: float = 0.7, valid_ratio: float = 0.2, test_ratio: float = 0.1, seed = 42):
        logger.info("Loading BiasDPO dataset")
        self.full_dataset = load_dataset("ahmedallam/BiasDPO", split="train")

        train_valid_test = self.full_dataset.train_test_split(test_size = test_ratio, seed=seed)
        self.test_dataset = train_valid_test["test"]
        train_valid = train_valid_test["train"]

        adjusted_ratio = valid_ratio / (valid_ratio + train_ratio)

        train_valid_split = train_valid.train_test_split(test_size = adjusted_ratio, seed=seed)
        self.valid_dataset = train_valid_split["test"]
        self.train_dataset = train_valid_split["train"]

        logger.info(f"Total samples: {len(self.full_dataset)}")
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.valid_dataset)}")
        logger.info(f"Test samples: {len(self.test_dataset)}")

        return self.train_dataset, self.valid_dataset, self.test_dataset
