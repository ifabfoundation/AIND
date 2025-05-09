"""
Downloader class for retrieving and saving data tables or image files from a remote server using HTTP requests.

This class supports two main functionalities:
- Downloading CSV tables using project and search parameters.
- Downloading study image files using file IDs listed in a CSV.

Key methods:
- `__retrive_table_download_url`: Internal method to generate the download URL for a table based on search and filter IDs.
- `__retrive_images_files_download_url`: Internal method to generate the download URL for image files based on a list of file IDs.
- `download_table_by_ulr`: Downloads the table as a CSV file and saves it locally.
- `download_image_files_by_url`: Downloads image files (ZIP format) using data IDs provided in a CSV file.

Dependencies:
- `lxml` for parsing XML responses.
- `requests` for performing HTTP GET and POST requests.
- `pandas` for reading file IDs from CSV.

Designed for use in data collection pipelines that interface with web-based data repositories.
"""

from lxml import etree as E
import requests
import pandas as pd

class Downloader():
    def __init__(self, project_name, base_url, jsession_id, ida_usc):
        self.project_name = project_name
        self.base_url = base_url
        self.session_id = jsession_id
        self.ida_usc = ida_usc

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
    
    def __retrive_images_files_download_url(self, filesId):
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
    
    def download_table_by_url(self, download_path = "../data/myTable.csv"):
        
        download_link = self.__retrive_table_download_url()
        url = self.base_url + 'download/files/search/' + download_link

        response = requests.get(url)
        if response.status_code == 200:
            # Save CSV in binary mode
            with open(download_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)

            print("file saved")
        else:
            print("There is a problem! File wasn't saved")

    def download_image_files_by_url(self, csv_path = '../data/data_ids.csv', download_path="../data/study_files.zip"):

        data_ids = pd.read_csv(csv_path)
        file_ids = [('fileId', str(id)) for id in data_ids['data_id']] # Create a list of key-value (fileId-id)

        download_link = self.__retrive_images_files_download_url(file_ids)
        url = self.base_url + 'download/files/study/' + download_link

        response = requests.get(url)
        if response.status_code == 200:
            # Save ZIP in binary mode
            with open(download_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)

            print("file saved")
        else:
            print("There is a problem! File wasn't saved")