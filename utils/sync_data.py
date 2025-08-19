import subprocess
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def sync_from_s3(destination: str) -> bool:
    s3_command = (
        f"aws s3 sync --no-sign-request "
        f"s3://physionet-open/eegmmidb/1.0.0/ {destination}"
    )

    logger.info("Synchronizing files from S3 to %s...", destination)
    try:
        subprocess.run(s3_command, shell=True, check=True)
        logger.info("Synchronization succeeded.")
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error while synchronizing from S3: {e}") from e

__all__ = ["sync_from_s3"]