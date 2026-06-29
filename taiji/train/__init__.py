"""taiji.train — 原生态极训练模块"""
from taiji.train.trainer import ModelSelfTrainer, TextDataset, ReActDataset
from taiji.train.data_loader import InstructionDataset, create_dataloader, split_dataset
from taiji.train.training_utils import EarlyStoppingCriteria, INFERENCE_THREADS, TRAINING_THREADS
from taiji.train.dataset_checker import DatasetQualityChecker
from taiji.train.dpo_trainer import DPOTrainer
from taiji.train.multimodal_trainer import MultimodalTrainer

__all__ = [
    "ModelSelfTrainer",
    "TextDataset",
    "ReActDataset",
    "InstructionDataset",
    "create_dataloader",
    "split_dataset",
    "EarlyStoppingCriteria",
    "INFERENCE_THREADS",
    "TRAINING_THREADS",
    "DatasetQualityChecker",
    "DPOTrainer",
    "MultimodalTrainer",
]
