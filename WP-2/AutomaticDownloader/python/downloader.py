"""
Downloader class for retrieving and saving data tables or image files from a remote server using HTTP requests.

This class supports two main functionalities:
- Downloading CSV tables using project and search parameters.
- Downloading study image files using file IDs listed in a CSV.

Key methods:
- `__retrive_table_download_url`: Internal method to generate the download URL for a table based on search and filter IDs.
- `__retrive_study_files_download_url`: Internal method to generate the download URL for image files based on a list of file IDs.
- `download_table_by_ulr`: Downloads the table as a CSV file and saves it locally.
- `download_study_files_by_url`: Downloads image files (ZIP format) using data IDs provided in a CSV file.

Dependencies:
- `lxml` for parsing XML responses.
- `requests` for performing HTTP GET and POST requests.
- `pandas` for reading file IDs from CSV.
- `dl_client` for interacting with datalake.

Designed for use in data collection pipelines that interface with web-based data repositories.
"""

from sys import prefix
from lxml import etree as E
import requests
import pandas as pd
import io
from dl_client import DatalakeClient

class Downloader():
    def __init__(self, project_name, base_url, jsession_id, ida_usc, datalake_client=None):
        self.project_name = project_name
        self.base_url = base_url
        self.session_id = jsession_id
        self.ida_usc = ida_usc
        # Initialize datalake client if provided, otherwise it will be created when needed
        self.datalake_client = datalake_client

    # Retrive the url for the download of myTable
    def __retrive_table_download_url(self, table = "myTable", filterId = 0, searchId = 1, name = "All Subject"):
        # Complete URL
        url = self.base_url + "ajax2/xml/search/downloadTables"

        # Query Params
        params = {
            "project": self.project_name,
            "table": table,
            "filterId": filterId,
            "searchId": searchId,
            "name": name
        }

        # Header
        headers = {
            "Cookie": "JSESSIONID=" + self.session_id + ";IDA_USC=" + self.ida_usc + ";"  
        }

        response = requests.get(url, headers=headers, params=params)

        xml_response = E.fromstring(response.content) # Convert in XML object
        download_url = xml_response.find("path").get("name") # Get download from tag path and attribute name

        return download_url
    
    def __retrive_study_files_download_url(self, filesId):
        # Complete URL
        url = self.base_url + "pages/ajax/getStudyData"

        # Body
        data = {
            'project': self.project_name
        }
        body = filesId + list(data.items())

        # Header
        headers = {
            "Cookie": "JSESSIONID=" + self.session_id + ";IDA_USC=" + self.ida_usc + ";",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        response = requests.post(url, headers=headers, data=body)

        xml_response = E.fromstring(response.content) # Convert in XML object
        download_url = xml_response.find("path").get("name") # Get download from tag path and attribute name

        return download_url
    
    def download_table_by_url(self, download_path = "../data/myTable.csv", upload_to_datalake=True, datalake_metadata=None):
        """
        Download table and either save locally or upload to datalake
        
        Args:
            download_path: Local path where the file would be saved (also used as object_name for datalake)
            upload_to_datalake: If True, upload to datalake instead of saving locally
            datalake_metadata: Optional metadata to associate with the file in datalake
        
        Returns:
            If uploading to datalake, returns the datalake upload response
            Otherwise, returns None
        """
        download_link = self.__retrive_table_download_url()
        url = self.base_url + 'download/files/search/' + download_link

        response = requests.get(url)
        if response.status_code == 200:
            if upload_to_datalake:
                # Make sure we have a datalake client
                if not self.datalake_client:
                    self.datalake_client = DatalakeClient()
                
                # Create in-memory file object with the content
                csv_data = io.BytesIO(response.content)
                
                # Create metadata if none provided
                metadata = datalake_metadata or {
                    "source": "ADNI",
                    "download_type": "study_files",
                    "level": "raw",
                }
                
                # Upload the file content directly to datalake
                result = self.datalake_client.upload_dataframe(
                    pd.read_csv(csv_data),
                    metadata=metadata,
                    prefix="raw",
                )
                
                print(f"File uploaded to datalake with ID: {result.get('metadata_id')}")
                return result
            else:
                # Save CSV in binary mode locally
                with open(download_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)
                print("File saved locally")
        else:
            print(f"There is a problem! HTTP status code: {response.status_code}")
            return None

    def download_study_files_by_url(self, csv_path = '../data/data_ids.csv', download_path="../data/study_files.zip", upload_to_datalake=True, datalake_metadata=None, extract_and_upload_individual_files=True, population_csv_path='../data/ADNI_code_pop.csv'):
        """
        Download image files as ZIP and either save locally or upload to datalake
        
        Args:
            csv_path: Path to CSV file containing data IDs
            download_path: Local path where the file would be saved (also used as object_name for datalake)
            upload_to_datalake: If True, upload to datalake instead of saving locally
            datalake_metadata: Optional metadata to associate with the file in datalake
            extract_and_upload_individual_files: If True, extract the ZIP and upload each file individually
            population_csv_path: Optional path to CSV containing file_code and population columns to add population metadata
        
        Returns:
            If uploading to datalake, returns the datalake upload response or a list of responses
            Otherwise, returns None
        """
        data_ids = pd.read_csv(csv_path)
        file_ids = [('fileId', str(id)) for id in data_ids['data_id']] # Create a list of key-value (fileId-id)

        download_link = self.__retrive_study_files_download_url(file_ids)
        url = self.base_url + 'download/files/study/' + download_link

        response = requests.get(url)
        if response.status_code == 200:
            if upload_to_datalake:
                # Make sure we have a datalake client
                if not self.datalake_client:
                    self.datalake_client = DatalakeClient()
                
                # For the metadata
                base_metadata = datalake_metadata or {
                    "source": "ADNI",
                    "download_type": "study_files",
                    "level": "raw",
                }
                
                import tempfile
                import os
                import zipfile
                from pathlib import Path
                
                # Read population data from CSV if provided
                population_data = None
                if population_csv_path:
                    try:
                        population_data = pd.read_csv(population_csv_path, sep=';')
                        # Validate that the CSV has the required columns
                        if 'file_name' not in population_data.columns or 'file_code' not in population_data.columns or 'population' not in population_data.columns:
                            print("\nWarning: population CSV should contain 'file_name', 'file_code' and 'population' columns")
                            population_data = None
                    except Exception as e:
                        print(f"\nError: reading population CSV: {e}")
                        population_data = None
                
                if extract_and_upload_individual_files:
                    # Check if response is a ZIP file or a single file
                    is_single_file = len(file_ids) == 1 or not download_link.endswith('.zip')

                    if is_single_file:
                        # Single file case - no extraction needed
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Extract filename from download_link
                            filename = download_link.split('/')[-1]

                            # Save file to temporary location
                            temp_file_path = os.path.join(temp_dir, filename)
                            with open(temp_file_path, "wb") as f:
                                f.write(response.content)

                            # Prepare metadata for this specific file
                            file_metadata = base_metadata.copy()

                            # Check if the filename contains any of the file_names and add population metadata if it does
                            match_found = False
                            if population_data is not None:
                                for _, row in population_data.iterrows():
                                    file_name_csv = str(row['file_name'])
                                    if file_name_csv and file_name_csv in filename:
                                        match_found = True
                                        file_metadata['file_code'] = str(row['file_code'])
                                        if pd.notna(row['population']):
                                            file_metadata['population'] = row['population'].split(',')
                                            print(f"\nMessage: Added population={row['population']} to metadata for file {filename}")
                                        break

                                # If no match found, skip this file
                                if not match_found:
                                    print(f"\nWarning: File {filename} skipped: no matching file_name found in CSV")
                                    return []

                            # Upload the file
                            result = self.datalake_client.upload_file(
                                file_path=temp_file_path,
                                metadata=file_metadata,
                                prefix='raw'
                            )
                            print(f"\nMessage: Uploaded file {filename} to datalake with file code: {file_metadata.get('file_code', 'N/A')}")
                            return [result]

                    else:
                        # ZIP file case - extract and upload individual files
                        # Create a temporary directory to extract files
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Save ZIP content to a temporary file
                            temp_zip_path = os.path.join(temp_dir, "temp.zip")
                            with open(temp_zip_path, "wb") as zip_file:
                                zip_file.write(response.content)

                            # Extract all files to the temporary directory
                            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                                extract_dir = os.path.join(temp_dir, "extracted")
                                os.makedirs(extract_dir, exist_ok=True)
                                zip_ref.extractall(extract_dir)

                            # Upload each file individually
                            upload_results = []

                            # Walk through all files in the extracted directory
                            for root, _, files in os.walk(extract_dir):
                                for filename in files:
                                    file_path = os.path.join(root, filename)
                                    relative_path = os.path.relpath(file_path, extract_dir)

                                    # Prepare metadata for this specific file
                                    file_metadata = base_metadata.copy()

                                    # Check if the filename contains any of the file_names and add population metadata if it does
                                    match_found = False
                                    if population_data is not None:
                                        for _, row in population_data.iterrows():
                                            file_name_csv = str(row['file_name'])
                                            if file_name_csv and file_name_csv in filename:
                                                match_found = True
                                                file_metadata['file_code'] = str(row['file_code'])
                                                if pd.notna(row['population']):
                                                    file_metadata['population'] = row['population'].split(',')
                                                    print(f"\nMessage: Added population={row['population']} to metadata for file {filename}")
                                                break

                                        # If no match found, skip this file
                                        if not match_found:
                                            print(f"\nWarning: File {filename} skipped: no matching file_name found in CSV")
                                            continue

                                    # Upload the file
                                    result = self.datalake_client.upload_file(
                                        file_path=file_path,
                                        metadata=file_metadata,
                                        prefix='raw'
                                    )
                                    upload_results.append(result)
                                    print(f"\nMessage: Uploaded file {relative_path} to datalake with file code: {file_metadata.get('file_code', 'N/A')}")

                            return upload_results
                else:
                    # Upload the entire ZIP file
                    
                    # Update metadata for ZIP file
                    zip_metadata = base_metadata.copy()
                    
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
                        temp_path = temp_file.name
                        # Write content to the temporary file
                        temp_file.write(response.content)
                    
                    try:
                        # Upload the temporary file to datalake
                        result = self.datalake_client.upload_file(temp_path, metadata=zip_metadata, prefix='raw')
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)

                print(f"\nMessage: File zip uploaded to datalake")
                return result
            else:
                # Save ZIP in binary mode locally
                with open(download_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)
                print("\nMessage: File saved locally")
        else:
            print(f"\nError: There is a problem! HTTP status code: {response.status_code}")
            return None