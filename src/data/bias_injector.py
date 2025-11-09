import logging
from .dataset_loader import DatasetLoader
import random
from typing import Dict, Any
from datasets import Dataset

logger = logging.getLogger(__name__)

class BiasInjector:
    def __init__(self, DatasetLoader, seed: int = 42):
        self.dataset_loader = DatasetLoader
        self.bias_ratio = None
        self.seed = seed
        random.seed(seed)

    def _flip_labels(self, example: Dict[str, Any]) -> Dict[str,Any]:
        example['chosen'], example['rejected'] = example['rejected'], example['chosen']
        return example

    def inject_bias(self, bias_ratio: float = 0.5):
        self.bias_ratio = bias_ratio

        train_ds = self.dataset_loader.train_dataset
        valid_ds = self.dataset_loader.valid_dataset
        test_ds = self.dataset_loader.test_dataset

        num_train_ex_to_flip = int(len(train_ds) * self.bias_ratio)
        num_valid_ex_to_flip = int(len(valid_ds) * self.bias_ratio)

        logger.info(f"Injecting {bias_ratio*100:.1f}% bias:")
        logger.info(f"  - Train: flipping {num_train_ex_to_flip}/{len(train_ds)} examples")
        logger.info(f"  - Valid: flipping {num_valid_ex_to_flip}/{len(valid_ds)} examples")

        train_indices_to_flip = set(random.sample(range(len(train_ds)), num_train_ex_to_flip))
        valid_indices_to_flip = set(random.sample(range(len(valid_ds)), num_valid_ex_to_flip))

        def apply_train_bias(example, idx):
            if idx in train_indices_to_flip:
                return self._flip_labels(example)
            return example

        def apply_valid_bias(example, idx):
            if idx in valid_indices_to_flip:
                return self._flip_labels(example)
            return example

        bias_train_dataset = train_ds.map(
                apply_train_bias,
                with_indices=True
                )

        bias_valid_dataset = valid_ds.map(
                apply_valid_bias,
                with_indices=True
                )
        logger.info("Bias injection complete")
        
        return bias_train_dataset, bias_valid_dataset, self.dataset_loader.test_dataset




