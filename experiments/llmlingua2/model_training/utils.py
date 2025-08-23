# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import spacy
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class TokenClfDataset(Dataset):
    def __init__(
        self,
        texts,
        labels=None,
        max_len=512,
        tokenizer=None,
        model_name="bert-base-multilingual-cased",
    ):
        # Input validation
        if not texts:
            raise ValueError("texts cannot be empty")
        if labels is not None and len(texts) != len(labels):
            raise ValueError(f"texts and labels must have same length: {len(texts)} vs {len(labels)}")
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        
        self.len = len(texts)
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels
        self.model_name = model_name
        
        # Set special tokens based on model type
        self._set_special_tokens()
        
        # Initialize spacy with error handling
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Using basic tokenization.")
            self.nlp = None

    def _set_special_tokens(self):
        """Set special tokens based on model name"""
        if "bert-base-multilingual-cased" in self.model_name.lower():
            self.cls_token = "[CLS]"
            self.sep_token = "[SEP]"
            self.unk_token = "[UNK]"
            self.pad_token = "[PAD]"
            self.mask_token = "[MASK]"
        elif "xlm-roberta" in self.model_name.lower() or "roberta" in self.model_name.lower():
            self.bos_token = "<s>"
            self.eos_token = "</s>"
            self.sep_token = "</s>"
            self.cls_token = "<s>"
            self.unk_token = "<unk>"
            self.pad_token = "<pad>"
            self.mask_token = "<mask>"
        else:
            # Try to get tokens from tokenizer if available
            try:
                self.cls_token = self.tokenizer.cls_token or "[CLS]"
                self.sep_token = self.tokenizer.sep_token or "[SEP]"
                self.pad_token = self.tokenizer.pad_token or "[PAD]"
                self.unk_token = self.tokenizer.unk_token or "[UNK]"
                self.mask_token = self.tokenizer.mask_token or "[MASK]"
                logger.warning(f"Unknown model type: {self.model_name}. Using tokenizer default tokens.")
            except AttributeError:
                # Fallback to BERT tokens
                self.cls_token = "[CLS]"
                self.sep_token = "[SEP]"
                self.unk_token = "[UNK]"
                self.pad_token = "[PAD]"
                self.mask_token = "[MASK]"
                logger.warning(f"Unknown model type: {self.model_name}. Using BERT default tokens.")

    def __getitem__(self, index):
        """Get a single sample from the dataset"""
        if not isinstance(index, int):
            raise TypeError(f"Index must be integer, got {type(index)}")
        if index >= len(self.texts) or index < 0:
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.texts)}")
            
        text = self.texts[index]
        
        # Handle labels if provided
        if self.labels is not None:
            labels = self.labels[index][:]  # Create a copy
            tokenized_text, labels = self.tokenize_and_preserve_labels(text, labels, self.tokenizer)
            
            # Ensure lengths match
            if len(tokenized_text) != len(labels):
                raise ValueError(f"Token and label lengths don't match: {len(tokenized_text)} vs {len(labels)}")
            
            # Add special token labels (False for CLS and SEP)
            labels.insert(0, False)  # For CLS token
            labels.append(False)     # For SEP token (FIXED: was insert(-1, False))
        else:
            tokenized_text = self.tokenizer.tokenize(text)

        # Add special tokens
        tokenized_text = [self.cls_token] + tokenized_text + [self.sep_token]

        # Truncate or pad to max_len
        if len(tokenized_text) > self.max_len:
            tokenized_text = tokenized_text[:self.max_len]
            if self.labels is not None:
                labels = labels[:self.max_len]
        else:
            # Pad tokens
            pad_length = self.max_len - len(tokenized_text)
            tokenized_text = tokenized_text + [self.pad_token] * pad_length
            
            # Pad labels
            if self.labels is not None:
                labels = labels + [False] * pad_length

        # Create attention mask
        attn_mask = [1 if tok != self.pad_token else 0 for tok in tokenized_text]

        # Convert tokens to IDs
        try:
            ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        except Exception as e:
            logger.error(f"Error converting tokens to IDs: {e}")
            raise

        # Prepare sample
        sample = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(attn_mask, dtype=torch.long),
        }
        
        if self.labels is not None:
            # Convert labels to integers (False->0, True->1)
            label_ints = [int(label) for label in labels]
            sample["targets"] = torch.tensor(label_ints, dtype=torch.long)

        return sample

    def __len__(self):
        return self.len

    def split_string(self, input_string, ignore_tokens=set([","])):
        """Split string into words using spaCy or basic splitting"""
        if self.nlp is not None:
            try:
                doc = self.nlp(input_string)
                word_list = []
                for word in doc:
                    if word.lemma_ not in ignore_tokens:
                        word_list.append(word.lemma_)
                return word_list
            except Exception as e:
                logger.warning(f"spaCy processing failed: {e}. Using basic split.")
        
        # Fallback to basic splitting
        return input_string.split()

    def tokenize_and_preserve_labels(self, text, text_labels, tokenizer):
        """
        Word piece tokenization makes it difficult to match word labels
        back up with individual word pieces. This function tokenizes each
        word one at a time so that it is easier to preserve the correct
        label for each subword.
        """
        tokenized_text = []
        labels = []

        text_words = self.split_string(text)
        
        # Validate input lengths
        if len(text_words) != len(text_labels):
            logger.warning(f"Text words ({len(text_words)}) and labels ({len(text_labels)}) length mismatch")
            # Handle mismatch by taking minimum length
            min_len = min(len(text_words), len(text_labels))
            text_words = text_words[:min_len]
            text_labels = text_labels[:min_len]

        for word, label in zip(text_words, text_labels):
            try:
                # Tokenize the word and count # of subwords
                tokenized_word = tokenizer.tokenize(str(word))
                n_subwords = len(tokenized_word)

                # Skip if tokenization failed
                if n_subwords == 0:
                    logger.warning(f"Failed to tokenize word: {word}")
                    continue

                # Add the tokenized word to the final tokenized word list
                tokenized_text.extend(tokenized_word)

                # Add the same label to the new list of labels `n_subwords` times
                labels.extend([label] * n_subwords)
                
            except Exception as e:
                logger.warning(f"Error tokenizing word '{word}': {e}")
                continue

        return tokenized_text, labels
