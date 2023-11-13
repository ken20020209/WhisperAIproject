from datasets import load_dataset, DatasetDict
from datasets import Audio
from transformers import WhisperFeatureExtractor,WhisperTokenizer
def preprocess(path):
    # get data set from mozilla common voice 11 zh-HK
    common_voice = DatasetDict()
    
    common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "zh-HK", split="train+validation", use_auth_token=True)
    common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "zh-HK", split="test", use_auth_token=True)

    # remove extra columns
    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    print(common_voice)

    # load feature extractor and tokenizer
    

    tokenizer = WhisperTokenizer.from_pretrained(path, language="chinese", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(path)


    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


    def preprocess_data(batch):

        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
        # encode target text to label ids
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    common_voice = common_voice.map(preprocess_data, remove_columns=common_voice.column_names["train"], num_proc=1)

    common_voice.save_to_disk("./common_voice_11_0/zh-HK/preprocessed/"+path)
def loadpreprocess(path):
    common_voice = DatasetDict.load_from_disk("./common_voice_11_0/zh-HK/preprocessed/"+path)
    return common_voice