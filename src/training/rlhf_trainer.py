from ..data.dataset_loader import DatasetLoader
from ..models.pythia_model import PythiaModel
from trl import PPOTrainer, PPOConfig, RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification
import torch
import copy
import logging

logger = logging.getLogger(__name__)

class RLHF_PPO_Trainer:
    def __init__(self, model, reward_model_base, value_model, processing_class, train_dataset, valid_ds, args = None):
        self.model = model
        self.args = args
        self.processing_class = processing_class
        self.train_dataset = self.prepare_dataset(train_dataset, processing_class)
        self.valid_ds = self.prepare_dataset(valid_ds, processing_class)
        self.reward_model_base = reward_model_base
        self.reward_model = self.prepare_reward_model(train_dataset, valid_ds)
        self.value_model = value_model
        self.create_ref_model(self.model)

        if self.processing_class.pad_token is None:
            self.processing_class.pad_token = self.processing_class.eos_token
            self.model.config.pad_token_id = self.processing_class.pad_token_id

        if args is None:
            self.args = PPOConfig(output_dir="./pythia-70m-deduped-PPO")

        logger.info("Initializing PPOTrainer...")
        self.trainer = PPOTrainer(
                model=self.model, reward_model=self.reward_model, value_model=self.value_model, 
                ref_model=self.ref_model, args=self.args, processing_class=self.processing_class, 
                train_dataset=self.train_dataset, eval_dataset=self.valid_ds)
        logger.info("PPOTrainer initialized successfully!")

    def create_ref_model(self, model):
        self.ref_model = copy.deepcopy(self.model)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

    def train(self):
        logger.info("Starting PPO training...")
        self.trainer.train()
        logger.info("PPO training complete.")

    def prepare_dataset(self, dataset, tokenizer):
        mapped_dataset = dataset.map(
                lambda x: tokenizer(x["prompt"], truncation=True),
                batched=True,
                remove_columns=['prompt','chosen','rejected']
                )
        return mapped_dataset

    def prepare_reward_model(self, train_dataset, valid_ds):
        logger.info("Creating reward model from base...")

        reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.reward_model_base,
                num_labels=1,
                dtype=torch.bfloat16,
                )
        reward_model.config.pad_token_id = self.processing_class.pad_token_id

        reward_config = RewardConfig(
            output_dir="./pythia-70m-reward-model",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            learning_rate=1e-5,
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            remove_unused_columns=False,
        )
        
        reward_trainer = RewardTrainer(
            model=reward_model,
            args=reward_config,
            train_dataset=train_dataset,
            eval_dataset=valid_ds,
            processing_class=self.processing_class,
        )
        
        logger.info("Training reward model...")
        reward_trainer.train()
        logger.info("Reward model training complete!")

        trained_reward_model = reward_trainer.model
        
        logger.info(f"Reward model type: {type(trained_reward_model)}")
        logger.info(f"Reward model has 'score' attribute: {hasattr(trained_reward_model, 'score')}")
        
        trained_reward_model.eval()
        
        return trained_reward_model
        



