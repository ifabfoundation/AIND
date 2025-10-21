"""
Main script for downloading data from a web-based platform using user-provided credentials and mode selection.

Workflow:
- Prompts the user to input JSESSION_ID, IDA_USC, and download mode (T for tables, F for image files).
- Initializes a `Downloader` instance with project-specific configurations.
- Starts a terminal spinner to provide visual feedback during the download process.
- Depending on the selected mode:
  - Downloads a data table (CSV format).
  - Downloads study image files (ZIP format) based on a CSV containing file IDs.

Handles invalid input and ensures proper termination of the spinner thread on completion or error.

Dependencies:
- `config.py`: Contains constants like `PROJECT_NAME` and `BASE_URL`.
- `downloader.py`: Implements the `Downloader` class.
- `utils.py`: Provides helper functions for prompting input and displaying a spinner.
"""

from config import *
from downloader import Downloader
import threading
from utils import get_cookies_values_from_prompt, spinner_task

if __name__ == '__main__':

    try:
        jsession_id, ida_usc, mode = get_cookies_values_from_prompt() # Get prompt value

        if mode not in ['T', 'F']: # Check mode
            raise Exception(f"Mod isn't valid: it must be T or F, you typed {mode}")
        
        # Initialize downloader
        downloader = Downloader(
            project_name=PROJECT_NAME,
            base_url=BASE_URL,
            jsession_id=jsession_id,
            ida_usc=ida_usc
        )

        # Start spinner
        stop_spinner = threading.Event()

        spinner_thread = threading.Thread(target=spinner_task, args=(stop_spinner,))
        spinner_thread.start()

        if mode == 'T':
            downloader.download_table_by_url() # Download table
        else:
            downloader.download_study_files_by_url() # Download study files
    except Exception as e:
        print(e)
    finally:
        # Stop spinner
        stop_spinner.set()
        spinner_thread.join()
    

