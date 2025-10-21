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
import os
import json

# Used to allow the user to insert the JSESSION and USC by prompt
def get_cookies_values_from_prompt():
    # Path del file di configurazione
    config_file = './credentials.json'
    
    # Controlla se il file esiste
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                
            # Verifica che il file contenga tutti i dati necessari
            if all(key in config_data for key in ['jsession_id', 'ida_usc', 'mode']):
                print(f"Valori letti dal file {config_file}")
                return config_data['jsession_id'], config_data['ida_usc'], config_data['mode']
            else:
                print(f"File {config_file} non contiene tutti i valori necessari")
                # Continua con il prompt
        except Exception as e:
            print(f"Errore nella lettura del file {config_file}: {e}")
            # Continua con il prompt
    
    # Se non è stato possibile leggere dal file, richiedi via prompt
    jsession_id = prompt("Insert your JSESSION_ID: ")
    ida_usc = prompt("Insert your IDA_USC: ")
    mode = prompt("Insert the mode (use T for Tables or use F for Images Files from csv): ")
    
    # Salva i valori nel file di configurazione per il futuro
    try:
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        config_data = {
            'jsession_id': jsession_id,
            'ida_usc': ida_usc,
            'mode': mode
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        print(f"Valori salvati nel file {config_file} per uso futuro")
    except Exception as e:
        print(f"Impossibile salvare i valori nel file {config_file}: {e}")

    return jsession_id, ida_usc, mode

# Used to create a spinner in the requests
def spinner_task(stop_event, message = "Downloading..."):
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    while not stop_event.is_set():
        sys.stdout.write('\r⏳ ' + message + ' ' + next(spinner))
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r✅ Done!\n')