import torch
 
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor
import evaluate
def fineTune(path,common_voice,per_device_train_batch_size=16):
    processor = WhisperProcessor.from_pretrained(path, language="chinese", task="transcribe")
    common_voice["test"]=common_voice["test"].select(range(500))
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
    
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
    
            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
    
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
    
            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
    
            batch["labels"] = labels
    
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


    
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
    
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # we do not want to group tokens when computing the metrics 
        pred_str =processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
        return {"wer": wer}

    from transformers import WhisperForConditionalGeneration
    
    model = WhisperForConditionalGeneration.from_pretrained(path)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    from transformers import Seq2SeqTrainingArguments
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./fineTue/"+path, # change to a repo name of your choice
        per_device_train_batch_size=2*per_device_train_batch_size,
        gradient_accumulation_steps=2*16//per_device_train_batch_size, # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=per_device_train_batch_size//2,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        
    )
    from transformers import Seq2SeqTrainer
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        
    )
    processor.save_pretrained("./fineTue/"+path)
    trainer.train()