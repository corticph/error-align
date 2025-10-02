import contextlib
import os
import warnings
from pathlib import Path

import click
import librosa
import nemo.collections.asr as nemo_asr
import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, WhisperForConditionalGeneration


@contextlib.contextmanager
def suppress_all_output():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress warnings

        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                # Optionally suppress tqdm progress bars
                try:
                    import tqdm

                    tqdm.__tqdm_original__ = tqdm.tqdm
                    tqdm.tqdm = lambda *a, **k: iter(a[0] if a else [])  # no-op iterator
                except ImportError:
                    pass
                try:
                    yield
                finally:
                    # Restore tqdm
                    if hasattr(tqdm, "__tqdm_original__"):
                        tqdm.tqdm = tqdm.__tqdm_original__


class Phi4MultimodalInstuct:
    """A class to handle the Phi-4 model for speech transcription."""

    def __init__(self):
        # Load model and processor
        self.model_path = "microsoft/Phi-4-multimodal-instruct"
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",  # If you do not use Ampere / later GPUs, change to "eager".
        ).cuda()
        self.generation_config = GenerationConfig.from_pretrained(self.model_path)

        # Define prompt structure
        self.prompt = "<|user|><|audio_1|>Transcribe the audio to text.<|end|><|assistant|>"

    def transcribe(self, example: dict):
        """Transcribe audio from an Huggingface dataset example."""
        # Get audio data.
        audio = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]

        # Preprocess data, run model, and decode response.
        if len(audio) < 400:
            audio = np.pad(audio, (0, 400 - len(audio)), mode="constant")
        inputs = self.processor(
            text=self.prompt,
            audios=[(audio, sampling_rate)],
            return_tensors="pt",
        ).to("cuda:0")
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=10_000,  # TODO: Adjust this based the longest example (probably in PriMock57).
            generation_config=self.generation_config,
            num_logits_to_keep=0,  # NOTE: Fails if not provided.
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        hyp = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return {"hyp": hyp}


class WhisperLargeV3:
    def __init__(self, language_code: str = "en"):
        """A class to handle the Whisper model for speech transcription."""
        self.model_path = "openai/whisper-large-v3"
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        ).cuda()
        self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        self.generation_config.return_timestamps = True
        self.generation_config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language_code,
            task="transcribe",
        )
        self.language_code = language_code

    def transcribe(self, example: dict):
        """Transcribe audio from an Huggingface dataset example."""
        # Get audio data.
        audio = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]

        # Preprocess data, run model, and decode response.
        if sampling_rate != 16_000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16_000)
        inputs = self.processor(
            audio=audio,
            sampling_rate=16_000,
            return_tensors="pt",
        ).to("cuda:0")
        generated_ids = self.model.generate(
            inputs=inputs.input_features.to(torch.float16),
            max_new_tokens=445,  # TODO: Adjust this based the longest example (probably in PriMock57).
            generation_config=self.generation_config,
            language=self.language_code,
        )
        hyp = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        return {"hyp": hyp}


class ParakeetTDTV2:
    """A class to handle the Parakeet TDT06B-v2 model for audio transcription."""

    def __init__(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

    def transcribe(self, example: dict):
        """Transcribe audio from an Huggingface dataset example."""
        # Get audio data.
        audio = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]

        # Resample audio to 16kHz if necessary
        if sampling_rate != 16_000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16_000)

        # Pad to minimum length = 257.
        if len(audio) < 257:
            audio = np.pad(audio, (0, 257 - len(audio)), mode="constant")

        # Run the model to get transcription
        with suppress_all_output():
            hyp = self.model.transcribe(audio)[0].text

        return {"hyp": hyp}


class ParakeetCTC:
    """A class to handle the Parakeet CTC model for audio transcription."""

    def __init__(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-ctc-1.1b")

    def transcribe(self, example: dict):
        """Transcribe audio from an Huggingface dataset example."""
        # Get audio data.
        audio = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]

        # Resample audio to 16kHz if necessary
        if sampling_rate != 16_000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16_000)

        # Pad to minimum length = 257.
        if len(audio) < 257:
            audio = np.pad(audio, (0, 257 - len(audio)), mode="constant")

        # Run the model to get transcription
        with suppress_all_output():
            hyp = self.model.transcribe(audio)[0].text

        return {"hyp": hyp}


def load_model(model_name: str, language_code: str = "en"):
    """Load a model from the Hugging Face Hub."""
    if model_name == "whisper":
        return WhisperLargeV3(language_code=language_code)
    if model_name == "parakeet-tdt":
        return ParakeetTDTV2()
    if model_name == "parakeet-ctc":
        return ParakeetCTC()
    if model_name == "phi":
        return Phi4MultimodalInstuct()


def load_hf_dataset(dataset_name: str, language_code: str, primock_path: str = None):
    """Load a dataset in Hugging Face format."""
    # Check if the dataset is supported and load it accordingly.
    if dataset_name == "commonvoice":
        dataset = load_dataset("mozilla-foundation/common_voice_17_0", language_code)
        dataset = dataset.rename_columns({"sentence": "ref", "client_id": "id"})
    elif dataset_name == "primock57":
        assert language_code == "en", "PriMock57 dataset only supports English."
        if primock_path is None:
            dataset_path = Path(__file__).parent / "primock57" / "audiofolder" / "metadata.parquet"
        else:
            dataset_path = Path(primock_path)
        assert dataset_path.exists(), "Dataset path for PriMock57 does not exist. Run `prepare_primock57.py` first."
        dataset = load_dataset("audiofolder", data_files={"test": dataset_path.as_posix()})
    elif dataset_name == "tedlium":
        assert language_code == "en", "TEDLIUM dataset only supports English."
        dataset = load_dataset("LIUM/tedlium", "release3-speaker-adaptation")
        dataset = dataset.rename_columns({"text": "ref"})
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    # Ensure the dataset has an audio column.
    for column_names in dataset.column_names.values():
        assert "audio" in column_names, "Dataset must contain an 'audio' column."

    return dataset


@click.command()
@click.option(
    "--model_name",
    type=click.Choice(["whisper", "parakeet-tdt", "parakeet-ctc", "phi"]),
    help="Name of the model to use.",
    required=True,
)
@click.option(
    "--dataset_name",
    type=click.Choice(["commonvoice", "primock57", "tedlium"]),
    help="Name of the dataset to load.",
    required=True,
)
@click.option(
    "--subset_name",
    type=click.Choice(["train", "validation", "test"]),
    help="Subset of the dataset to use.",
    default="test",
)
@click.option(
    "--language_code",
    type=click.Choice(["en", "es", "pt", "fr", "de", "pl", "tr", "id", "vi", "sw"]),
    help="Language code of the audio data. Only `commonvoice` supports other languages than English.",
    default="en",
)
@click.option(
    "--primock_path",
    type=str,
    help="Local path to the PriMock57 git repo. If None, will clone the repository in the 'evaluation' directory.",
    default=None,
)
def main(model_name: str, dataset_name: str, subset_name: str, language_code: str, primock_path: str = None):
    """Main function to load the model and process data."""
    model = load_model(model_name, language_code=language_code)
    dataset = load_hf_dataset(dataset_name, language_code, primock_path)[subset_name]

    transcribed_dataset = []
    for example in tqdm(dataset):
        hyp = model.transcribe(example)["hyp"]
        transcribed_dataset.append({"id": example["id"], "hyp": hyp, "ref": example["ref"]})

    # Convert to huggingface datatset format and dump as parquet file
    transcribed_dataset = Dataset.from_list(transcribed_dataset)
    output_dir = Path(__file__).parent / "transcribed_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_{dataset_name}_{subset_name}_{language_code}.parquet"
    transcribed_dataset.to_parquet(output_path.as_posix())


if __name__ == "__main__":
    main()
