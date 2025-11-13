from src.data.dataset_loader import DatasetLoader
from src.models.pythia_model import PythiaModel
from trl import DPOTrainer, DPOConfig

import logging

logger = logging.getLogger(__name__)

class DPO_Trainer:
    def __init__(self, model: PythiaModel.model, args: None, processing_class: PythiaModel.tokenizer, train_dataset: DatasetLoader.train_dataset):
        self.model = model
        self.args = args
        self.processing_class = processing_class
        self.train_dataset = train_dataset

        logger.info("Initializing DPOTrainer...")
        self.trainer = DPOTrainer(model=model, args=args, processing_class=processing_class, train_dataset=train_dataset)
        logger.info("DPOTrainer initialized successfully!")

    def train(self):
        logger.info("Starting DPO training...")
        self.trainer.train()
        logger.info("DPO training complete.")
