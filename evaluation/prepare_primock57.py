import re
from glob import glob
from pathlib import Path

import click
import git
import pandas as pd
import soundfile as sf
from textgrid import TextGrid
from tqdm import tqdm


def save_audio_segments_and_create_metadata(
    audio_path: Path,
    transcript_path: Path,
    output_path: Path,
    session: str,
) -> list:
    """Save audio segments and create metadata for the PriMock57 dataset."""
    # Read audio file and transcript
    audio, sampling_rate = sf.read(audio_path, dtype="int16")
    transcript_textgrid = TextGrid.fromFile(transcript_path)

    example_metadata = []
    for interval in transcript_textgrid[0]:
        # Skip empty intervals
        if not interval.mark.strip():
            continue

        # Extract audio segment and save it
        start = int(interval.minTime * sampling_rate)
        stop = int(interval.maxTime * sampling_rate)
        audio_segment = audio[start:stop]
        session_segment = f"{session}_{start}_{stop}"
        segment_filename = f"{session_segment}.wav"
        sf.write(output_path / segment_filename, audio_segment, sampling_rate)

        # Create metadata entry
        example_metadata.append(
            {
                "id": session_segment,
                "file_name": segment_filename,
                "ref": interval.mark.strip(),
            },
        )

    return example_metadata


@click.command()
@click.option(
    "--primock_path",
    type=str,
    help="Local path to the PriMock57 git repository. If None, will clone the repository the 'evaluation' directory.",
    default=None,
)
@click.option(
    "--output_path",
    type=str,
    help="Path to save the processed audiofolder. If None, defaults to 'audiofolder' in the PriMock57 directory.",
    default=None,
)
def main(primock_path: str | None = None, output_path: str | None = None):
    """Main function to process the PriMock57 dataset."""
    # If no path is provided, clone the repository
    if primock_path is None:
        repo_url = "https://github.com/babylonhealth/primock57.git"
        primock_path = Path(__file__).parent / "primock57"
        if not primock_path.exists():
            print(f"Cloning PriMock57 repository from {repo_url} to {primock_path}")
            git.Repo.clone_from(repo_url, primock_path)

    # Validate the primock_path
    primock_path = Path(primock_path).absolute()
    assert primock_path.exists() and primock_path.is_dir(), "Invalid PriMock57 path provided."

    # If no output path is provided, use the default path
    if output_path is None:
        output_path = Path(primock_path) / "audiofolder"
    else:
        output_path = Path(output_path)

    # Ensure the output path exists and is an empty directory
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    assert output_path.is_dir() and not any(output_path.iterdir()), "Output path is not an empty directory."

    # Collect session identifiers
    # sessions = collect_sessions(primock_path)

    metadata = []
    for audio_path in tqdm(glob(f"{primock_path}/audio/*.wav")):
        # Extract session identifier from the audio file name
        session = next(re.finditer(r"day\d_consultation\d{2}_(doctor|patient)", audio_path)).group()

        # Create and validate paths for audio and transcript files
        audio_path = Path(audio_path).absolute()
        transcript_path = Path(f"{primock_path}/transcripts/{session}.TextGrid")
        assert transcript_path.exists(), f"Transcript file for session {session} not found."

        # Save audio segments and create metadata
        example_metadata = save_audio_segments_and_create_metadata(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_path=output_path,
            session=session,
        )
        metadata.extend(example_metadata)

    # Save metadata as parquet file
    metadata_df = pd.DataFrame(metadata)
    metadata_path = Path(f"{output_path}/metadata.parquet")
    metadata_df.to_parquet(metadata_path, index=False)


if __name__ == "__main__":
    main()
