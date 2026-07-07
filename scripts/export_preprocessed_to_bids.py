"""
A standardized script to preprocess raw EEG (BDF) data and export it into a 
BIDS-compliant derivative dataset directory.

The processing pipeline includes:
    1. Channel naming standardization and channel selection (33 standard channels)
    2. Electrode coordinate mapping (standard 10-20 montage)
    3. Common average referencing (CAR)
    4. Finite Impulse Response (FIR) bandpass filtering (1.0 - 100.0 Hz)
    5. Notch filtering (50 Hz powerline noise attenuation)
    6. Downsampling to 250 Hz
    7. Temporal cropping, event mapping, and BrainVision formatting

Dependencies:
    mne, mne-bids, pandas, pybv
"""

import os
import sys
import logging
import argparse
import pandas as pd
import mne
from mne_bids import (
    BIDSPath,
    write_raw_bids,
    make_dataset_description,
    update_sidecar_json
)

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def convert_preprocessed_to_bids(
        raw_bdf_path: str,
        events_tsv_path: str,
        output_dir: str,
        subject_id: str,
        task_name: str = 'ssvep',
        t_max_sec: float = 1620.0,  # 27 minutes * 60 seconds
) -> None:
    """Preprocesses a raw BDF file and writes it to a BIDS derivative folder.

    Parameters
    ----------
    raw_bdf_path : str
        Path to the input raw BDF file.
    events_tsv_path : str
        Path to the event markers TSV file.
    output_dir : str
        Root directory for the preprocessed BIDS derivatives.
    subject_id : str
        The unique subject identifier (e.g., '30').
    task_name : str, optional
        Name of the experimental task, by default 'ssvep'.
    t_max_sec : float, optional
        Max duration to crop raw signal in seconds, by default 1620.0 (27 min).
    """
    logger.info(f"Starting preprocessed BIDS export for Subject: {subject_id}")

    # Standard 10-20 system 33-channel configuration
    standard_33_chans = [
        'Cz', 'Fpz', 'Fz', 'FCz', 'CPz', 'Pz', 'Oz',
        'T3', 'T4', 'C3', 'C4',
        'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8',
        'FC3', 'FC4', 'FT7', 'FT8',
        'O1', 'O2', 'T5', 'T6', 'P3', 'P4',
        'CP3', 'CP4', 'TP7', 'TP8',
        'A1', 'A2'
    ]

    # --- Step 1: Path validation and safety check ---
    if not os.path.exists(raw_bdf_path):
        raise FileNotFoundError(f"Raw BDF file not found at: {raw_bdf_path}")
    if not os.path.exists(events_tsv_path):
        raise FileNotFoundError(f"Event markers TSV file not found at: {events_tsv_path}")

    os.makedirs(output_dir, exist_ok=True)

    # --- Step 2: Load raw dataset ---
    logger.info(f"Loading raw BDF file: {raw_bdf_path}")
    raw = mne.io.read_raw_bdf(raw_bdf_path, preload=True)

    # Dynamically clean channel names (strip trailing/leading whitespaces)
    raw.rename_channels(lambda x: x.strip())

    # Compatibility renaming (map common non-standard 'FPz' to standard 10-20 'Fpz')
    rename_mapping = {}
    for ch in raw.ch_names:
        if ch.upper() == 'FPZ':
            rename_mapping[ch] = 'Fpz'
    if rename_mapping:
        raw.rename_channels(rename_mapping)
        logger.info(f"Renamed non-standard channels: {rename_mapping}")

    # --- Step 3: Extract target channels and apply montage ---
    available_chans = [ch for ch in standard_33_chans if ch in raw.ch_names]
    missing_chans = set(standard_33_chans) - set(available_chans)
    if missing_chans:
        logger.warning(f"Following target channels were missing in raw file: {missing_chans}")

    raw.pick(available_chans)
    logger.info(f"Selected {len(available_chans)} channels for preprocessing.")

    # Set standard 3-D electrode coordinates (montage)
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        logger.info("Successfully applied standard 10-20 sensor montage.")
    except Exception as e:
        logger.error(f"Failed to set standard 10-20 montage: {e}")

    # --- Step 4: Execution of Preprocessing Pipeline ---
    logger.info("Applying common average reference (CAR)...")
    raw.set_eeg_reference('average')

    logger.info("Applying zero-phase FIR filter (1.0 Hz - 100.0 Hz)...")
    raw.filter(l_freq=1.0, h_freq=100.0, fir_design='firwin', verbose=False)

    logger.info("Applying 50.0 Hz notch filter for line noise suppression...")
    raw.notch_filter(freqs=50.0, verbose=False)

    logger.info("Resampling signal to 250 Hz...")
    raw.resample(sfreq=250.0, verbose=False)

    # --- Step 5: Rigorous temporal cropping ---
    total_duration = raw.times[-1]
    crop_limit = min(t_max_sec, total_duration)
    # When using crop, tmax must not exceed the actual boundary of the recording
    raw.crop(tmin=0.0, tmax=crop_limit, include_tmax=True)
    logger.info(f"Preprocessed data cropped to: {raw.times[-1] / 60:.2f} minutes.")

    # --- Step 6: Import and filter out-of-boundary Event labels ---
    logger.info(f"Reading event definitions from: {events_tsv_path}")
    df_events = pd.read_csv(events_tsv_path, sep='\t')

    # Boundary protection: discard events occurring after the crop limit
    initial_event_count = len(df_events)
    df_events_filtered = df_events[df_events['onset'] + df_events['duration'] <= crop_limit]
    dropped_count = initial_event_count - len(df_events_filtered)
    if dropped_count > 0:
        logger.warning(f"Discarded {dropped_count} event(s) exceeding cropped signal boundary ({crop_limit}s).")

    # Convert to MNE Annotation format
    annot = mne.Annotations(
        onset=df_events_filtered['onset'].values,
        duration=df_events_filtered['duration'].values,
        description=df_events_filtered['trial_type'].values
    )
    raw.set_annotations(annot)

    # --- Step 7: Export to standardized BIDS derivative structure ---
    bids_path = BIDSPath(
        subject=subject_id,
        task=task_name,
        datatype='eeg',
        suffix='eeg',
        root=output_dir
    )

    # Export to BrainVision format (an open, text-header-based standard)
    logger.info(f"Writing derivative BIDS files into output directory: {output_dir}")
    write_raw_bids(
        raw,
        bids_path=bids_path,
        overwrite=True,
        allow_preload=True,
        format='BrainVision'
    )

    # --- Step 8: Populate derivative-specific hardware metadata in JSON sidecar ---
    eeg_json_bids_path = bids_path.copy().update(extension='.json')
    derivative_entries = {
        "Manufacturer": "Zekang EEG",
        "PowerLineFrequency": 50,  # Power line frequency (Hz)
        # Reflects the software-based DSP pipeline executed by this script
        "SoftwareFilters": "HP 1.0 Hz, LP 100.0 Hz, Notch 50 Hz, downsampled to 250 Hz",
        "EEGReference": "Average reference (re-referenced during preprocessing; originally a dedicated electrode placed between Cz and CPz).",
        "EEGGround": "Dedicated ground electrode placed between Fpz and Fz (not recorded as a channel in dataset)."
    }
    update_sidecar_json(bids_path=eeg_json_bids_path, entries=derivative_entries)
    logger.info("Updated derivative-specific sidecar metadata.")

    # --- Step 9: Generate dataset description checklist ---
    dataset_name = (
        "A multi-factor EEG and subjective ergonomic evaluation dataset "
        "for steady-state visual evoked potentials under realistic backgrounds"
    )
    make_dataset_description(
        path=output_dir,
        name=dataset_name,
        dataset_type='derivative',  # Explicitly defined as derivative dataset
        authors=['Tairan Liang'],
        overwrite=True  # Allow overwrite to update dataset metadata
    )
    logger.info("Derivative dataset description successfully created.")
    logger.info(f"Successfully processed and validated: Subject {subject_id} (Derivative)")


# === Batch Processing and CLI Execution Entry ===
if __name__ == "__main__":
    # Use argparse for command-line debugging to match open-source toolbox conventions
    parser = argparse.ArgumentParser(description="Convert preprocessed raw files to BIDS derivatives.")
    parser.add_argument('--sub', type=str, default='01', help="Subject ID (e.g., 30)")
    parser.add_argument('--bdf_input', type=str, default=None, help="Absolute path to input .bdf file")
    parser.add_argument('--events_tsv', type=str, default=None, help="Path to events.tsv")
    parser.add_argument('--bids_out', type=str, default=None, help="Path to write BIDS derivative files")

    args = parser.parse_args()

    # Default paths for execution (used when running directly in IDE)
    default_sub = args.sub
    default_bdf = args.bdf_input
    default_events = args.events_tsv
    default_out = args.bids_out

    try:
        convert_preprocessed_to_bids(
            raw_bdf_path=default_bdf,
            events_tsv_path=default_events,
            output_dir=default_out,
            subject_id=default_sub,
            task_name='ssvep',
            t_max_sec=27 * 60.0
        )
        print("\n" + "=" * 50 + "\n[SUCCESS] Preprocessed BIDS Conversion completed successfully!\n" + "=" * 50)
    except Exception as err:
        logger.exception(f"Conversion failed due to an unexpected error: {err}")
        sys.exit(1)
