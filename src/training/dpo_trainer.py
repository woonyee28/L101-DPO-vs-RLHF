from ..data.dataset_loader import DatasetLoader
from ..models.pythia_model import PythiaModel
from trl import DPOTrainer, DPOConfig

import logging

logger = logging.getLogger(__name__)

class DPO_Trainer:
    def __init__(self, model, processing_class, train_dataset, valid_ds, args = None):
        self.model = model
        self.args = args
        self.processing_class = processing_class
        self.train_dataset = train_dataset
        self.valid_ds = valid_ds

        if args is None:
            self.args = DPOConfig(output_dir="./pythia-70m-deduped-DPO")

        logger.info("Initializing DPOTrainer...")
        self.trainer = DPOTrainer(model=model, args=self.args, processing_class=processing_class, train_dataset=train_dataset, eval_dataset=valid_ds)
        logger.info("DPOTrainer initialized successfully!")

    def train(self):
        logger.info("Starting DPO training...")
        self.trainer.train()
        logger.info("DPO training complete.")
