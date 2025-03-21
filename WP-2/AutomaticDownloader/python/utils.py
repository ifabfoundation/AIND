"""
Utility module for terminal-based user interaction and visual feedback during data download operations.

Features:

- `get_cookies_values_from_prompt`: Prompts the user to input the JSESSION_ID, IDA_USC, and the desired download mode 
  (T for tables, F for image files from CSV).
- `spinner_task`: Displays a simple spinner animation to indicate that a process (e.g., downloading) is in progress. 
  Intended to be run in a separate thread to provide visual feedback until the task is complete.

Dependencies:
- `prompt_toolkit` for enhanced terminal input.
- `threading`, `itertools`, `time`, `sys` for managing the spinner and execution flow.

This module is designed to enhance the user experience in CLI-based data retrieval tools.
"""

from prompt_toolkit import prompt
import threading
import itertools
import time
import sys

# Used to allow the user to insert the JSESSION and USC by prompt
def get_cookies_values_from_prompt():
    jsession_id = prompt("Insert your JSESSION_ID: ")
    ida_usc = prompt("Insert your IDA_USC: ")
    mode = prompt("Insert the mode (use T for Tables or use F for Images Files from csv): ")

    return jsession_id, ida_usc, mode

# Used to create a spinner in the requests
def spinner_task(stop_event, message = "Downloading..."):
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    while not stop_event.is_set():
        sys.stdout.write('\r⏳ ' + message + ' ' + next(spinner))
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r✅ Done!\n')