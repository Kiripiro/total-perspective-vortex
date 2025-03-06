import os
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import mne

matplotlib.use("Qt5Agg")


def clean_channel_name(ch_name):
    """
    Example: Removes dots and converts to uppercase.
    Ex: 'Fc5.' -> 'FC5'
    """
    return ch_name.replace('.', '').upper()


def sync_files_from_s3(destination):
    """
    Synchronizes files from the specified S3 bucket to the local destination.
    """
    s3_command = (
        f"aws s3 sync --no-sign-request "
        f"s3://physionet-open/eegmmidb/1.0.0/ {destination}"
    )
    print("Synchronizing files from S3...")
    try:
        subprocess.run(s3_command, shell=True, check=True)
        print("Synchronization successful.")
    except subprocess.CalledProcessError as e:
        print(f"Error during synchronization: {e}")
        exit(1)


def load_edf_file(file_path):
    """
    Loads an EDF file and returns the raw data.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} doesn't exist.")
        exit(1)
    return mne.io.read_raw_edf(file_path, preload=True)


def main():
    local_destination = "./data"
    sync_files_from_s3(local_destination)

    edf_file = os.path.join(local_destination, "S001", "S001R01.edf")
    raw = load_edf_file(edf_file)

    raw.rename_channels(clean_channel_name)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)

    print(raw.info)

    raw.compute_psd().plot()
    plt.show()


if __name__ == "__main__":
    main()
