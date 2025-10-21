"""
Datalake Client

This module provides a client for interacting with the AIND Datalake API.
"""

import os
import json
import requests
import configparser
from pathlib import Path
from typing import Dict, List, Union, Optional, BinaryIO, Any
import io
import pandas as pd
import zipfile


class DatalakeClient:
    """Client for interacting with the AIND Datalake API."""

    def __init__(self, base_url: str = "https://datalake.ifabfoundation.it", default_bucket: str = "aind", username: str = None, password: str = None, config_file: str = None):
        """
        Initialize the Datalake client.

        Args:
            base_url: Base URL of the Datalake API.
            default_bucket: Default bucket to use for operations.
            username: Username for authentication. If provided along with password, will automatically login.
            password: Password for authentication.
            config_file: Path to a config file containing credentials. If not provided, will look in default locations.
        """
        self.base_url = base_url.rstrip('/')
        self.default_bucket = default_bucket
        self.session = requests.Session()
        self.access_token = None
        
        # Try different authentication methods in order of precedence
        if username and password:
            # 1. Use provided credentials if available
            self.login(username, password)
        elif config_file:
            # 2. Use specified config file
            self._login_from_config(config_file)
        else:
            # 3. Try default config locations
            self._try_auto_login()
    
    def _login_from_config(self, config_path: str) -> bool:
        """
        Attempt to login using credentials from a config file.
        
        Args:
            config_path: Path to the config file.
            
        Returns:
            True if login was successful, False otherwise.
        """
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            
            if 'credentials' in config and 'username' in config['credentials'] and 'password' in config['credentials']:
                username = config['credentials']['username']
                password = config['credentials']['password']
                self.login(username, password)
                return True
            return False
        except Exception:
            return False
    
    def _try_auto_login(self) -> bool:
        """
        Try to automatically login using credentials from default config locations.
        
        Returns:
            True if login was successful, False otherwise.
        """
        # List of potential config file locations
        config_paths = [
            # User's home directory
            os.path.join(str(Path.home()), '.config', 'dl_client', 'config.ini'),
            # Current directory
            os.path.join(os.getcwd(), '.dl_client.ini'),
            # Environment variable if set
            os.environ.get('DL_CLIENT_CONFIG', '')
        ]
        
        for path in config_paths:
            if path and os.path.exists(path):
                if self._login_from_config(path):
                    return True
        
        return False
    
    def save_credentials(self, config_path: str = None) -> str:
        """
        Save the current credentials to a config file.
        
        Args:
            config_path: Path where to save the config file. If None, will use the default location.
            
        Returns:
            Path to the saved config file.
            
        Raises:
            Exception: If no credentials are available to save.
        """
        if not hasattr(self, '_username') or not hasattr(self, '_password'):
            raise Exception("No credentials available to save")
        
        # Use default path if none provided
        if not config_path:
            config_dir = os.path.join(str(Path.home()), '.config', 'dl_client')
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, 'config.ini')
        
        # Create config file
        config = configparser.ConfigParser()
        config['credentials'] = {
            'username': self._username,
            'password': self._password
        }
        
        # Save to file
        with open(config_path, 'w') as f:
            config.write(f)
        
        return config_path

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate with the Datalake API and get an access token.

        Args:
            username: Username for authentication.
            password: Password for authentication.

        Returns:
            Dict containing the authentication result.

        Raises:
            Exception: If authentication fails.
        """
        url = f"{self.base_url}/api/login"
        
        # Store credentials for potential saving to config
        self._username = username
        self._password = password
        
        # Prepare data
        data = {
            'username': username,
            'password': password
        }
        
        # Send request
        response = self.session.post(url, json=data)
        
        # Parse response
        if response.status_code == 200:
            result = response.json()
            self.access_token = result.get('access_token')
            # Set the token in the session headers for all future requests
            self.session.headers.update({
                'Authorization': f'Bearer {self.access_token}'
            })
            return result
        else:
            try:
                response_data = response.json()
                error_message = response_data.get('error', 'Unknown error')
            except:
                error_message = f"HTTP error {response.status_code}"
            raise Exception(f"Authentication failed: {error_message}")
    
    def register_user(self, username: str, password: str, role: str = 'user') -> Dict[str, Any]:
        """
        Register a new user (requires admin privileges).

        Args:
            username: Username for the new user.
            password: Password for the new user.
            role: Role for the new user (default: 'user').

        Returns:
            Dict containing the registration result.

        Raises:
            Exception: If registration fails or if the current user doesn't have admin privileges.
        """
        if not self.access_token:
            raise Exception("You must be logged in as an admin to register users")
            
        url = f"{self.base_url}/api/register"
        
        # Prepare data
        data = {
            'username': username,
            'password': password,
            'role': role
        }
        
        # Send request
        response = self.session.post(url, json=data)
        
        # Parse response
        if response.status_code == 201:
            return response.json()
        else:
            try:
                response_data = response.json()
                error_message = response_data.get('error', 'Unknown error')
            except:
                error_message = f"HTTP error {response.status_code}"
            raise Exception(f"User registration failed: {error_message}")
    
    def upload_file(self, 
                   file_path: str, 
                   object_name: Optional[str] = None, 
                   bucket: Optional[str] = None, 
                   prefix: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Upload a file to the Datalake.

        Args:
            file_path: Path to the file to upload.
            object_name: Name to use for the object in the Datalake. If None, the filename will be used.
            bucket: Bucket to upload to. If None, the default bucket will be used.
            prefix: Prefix to use for the object in the Datalake.
            metadata: Metadata to associate with the file.

        Returns:
            Dict containing the result of the upload operation.
            
        Raises:
            Exception: If the upload fails or if not authenticated.
        """
        if not self.access_token:
            raise Exception("You must be logged in to upload files")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        url = f"{self.base_url}/api/files"
        
        # Prepare form data
        form_data = {}
        if bucket:
            form_data['bucket'] = bucket
        else:
            form_data['bucket'] = self.default_bucket
            
        if object_name:
            form_data['object_name'] = object_name
            
        if prefix:
            form_data['prefix'] = prefix
            
        if metadata:
            form_data['metadata'] = json.dumps(metadata)

        # Prepare file
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            
            # Send request
            response = self.session.post(url, data=form_data, files=files)
            
        # Parse response
        if response.status_code == 201:
            return response.json()
        else:
            response_data = response.json()
            error_message = response_data.get('error', 'Unknown error')
            raise Exception(f"Upload failed: {error_message}")
            
    def upload_dataframe(self,
                        df: 'pd.DataFrame',
                        object_name: str,
                        bucket: Optional[str] = None,
                        prefix: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        file_format: str = 'csv',
                        **kwargs) -> Dict[str, Any]:
        """
        Upload a pandas DataFrame directly to the Datalake without saving to a temporary file.

        Args:
            df: The pandas DataFrame to upload.
            object_name: Name to use for the object in the Datalake (must include appropriate extension).
            bucket: Bucket to upload to. If None, the default bucket will be used.
            prefix: Prefix to use for the object in the Datalake.
            metadata: Metadata to associate with the file.
            file_format: Format to save the DataFrame as. Options: 'csv', 'json', 'parquet', 'excel'.
            **kwargs: Additional keyword arguments to pass to the DataFrame export function
                     (e.g., index=False for to_csv).

        Returns:
            Dict containing the result of the upload operation.
            
        Raises:
            Exception: If the upload fails, if not authenticated, or if an unsupported file format is specified.
        """
        if not self.access_token:
            raise Exception("You must be logged in to upload files")

        # Validate that df is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
            
        # Create a BytesIO object to hold the data
        buffer = io.BytesIO()
        
        # Get filename from object_name
        filename = os.path.basename(object_name)
        
        # Export DataFrame to the appropriate format
        if file_format.lower() == 'csv':
            # Set default kwargs for CSV if not provided
            if 'index' not in kwargs:
                kwargs['index'] = False
            df.to_csv(buffer, **kwargs)
        elif file_format.lower() == 'json':
            # Set default kwargs for JSON if not provided
            if 'orient' not in kwargs:
                kwargs['orient'] = 'records'
            df.to_json(buffer, **kwargs)
        elif file_format.lower() == 'parquet':
            df.to_parquet(buffer, **kwargs)
        elif file_format.lower() == 'excel':
            df.to_excel(buffer, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}. Supported formats: csv, json, parquet, excel")
            
        # Reset buffer position to the beginning
        buffer.seek(0)
        
        url = f"{self.base_url}/api/files"
        
        # Prepare form data
        form_data = {}
        if bucket:
            form_data['bucket'] = bucket
        else:
            form_data['bucket'] = self.default_bucket
            
        if object_name:
            form_data['object_name'] = object_name
            
        if prefix:
            form_data['prefix'] = prefix
            
        if metadata:
            form_data['metadata'] = json.dumps(metadata)

        # Prepare file
        files = {'file': (filename, buffer)}
        
        # Send request
        response = self.session.post(url, data=form_data, files=files)
        
        # Parse response
        if response.status_code == 201:
            return response.json()
        else:
            try:
                response_data = response.json()
                error_message = response_data.get('error', 'Unknown error')
            except:
                error_message = f"HTTP error {response.status_code}"
            raise Exception(f"Upload failed: {error_message}")

    def download_file(self, 
                     object_name: str, 
                     output_path: Optional[str] = None, 
                     bucket: Optional[str] = None,
                     as_attachment: bool = False,
                     auto_parse: bool = True,
                     extract_zip: bool = False,
                     extract_dir: Optional[str] = None) -> Union[str, bytes, Dict[str, Any], Any]:
        """
        Download a file from the Datalake.

        Args:
            object_name: Name of the object to download.
            output_path: Path to save the downloaded file. If None, the file content will be returned.
            bucket: Bucket to download from. If None, the default bucket will be used.
            as_attachment: Whether to download the file as an attachment.
            auto_parse: Whether to automatically parse the file content based on its extension.
                        If True, common file types will be parsed and returned as appropriate Python objects:
                        - CSV files will be returned as pandas DataFrames
                        - JSON files will be parsed into Python dictionaries/lists
                        - Text files will be returned as strings
                        - ZIP files will be processed according to extract_zip parameter
                        - Other files will be returned as bytes
            extract_zip: If True and the file is a ZIP, extract its contents. If output_path is provided,
                        the ZIP will be extracted to extract_dir or to the directory of output_path.
                        If output_path is None, the contents will be loaded into memory and returned
                        as a dictionary mapping filenames to their contents (auto-parsed if auto_parse is True).
            extract_dir: Directory to extract ZIP contents to. Only used if extract_zip is True and
                        output_path is provided. If None, the ZIP will be extracted to the same
                        directory as output_path.

        Returns:
            If output_path is provided: 
                - Path to the downloaded file (str)
                - If extract_zip is True and the file is a ZIP, path to the extraction directory
            If output_path is None and auto_parse is True: 
                - Parsed content based on file type (DataFrame, dict, str, etc.)
                - If extract_zip is True and the file is a ZIP, a dictionary mapping filenames to their parsed contents
            If output_path is None and auto_parse is False: 
                - Raw file content (bytes)
                - If extract_zip is True and the file is a ZIP, a dictionary mapping filenames to their raw contents (bytes)
        """
        if not self.access_token:
            raise Exception("You must be logged in to download files")
        url = f"{self.base_url}/api/files/{object_name}"
        
        params = {}
        if bucket:
            params['bucket'] = bucket
        else:
            params['bucket'] = self.default_bucket
            
        if as_attachment:
            params['download'] = 'true'
            
        # Send request
        response = self.session.get(url, params=params, stream=True)
        
        # Check response
        if response.status_code != 200:
            try:
                response_data = response.json()
                error_message = response_data.get('error', 'Unknown error')
            except:
                error_message = f"HTTP error {response.status_code}"
            raise Exception(f"Download failed: {error_message}")
            
        # Handle response
        file_ext = os.path.splitext(object_name.lower())[1]
        
        # Handle ZIP files specially if extract_zip is True
        if extract_zip and file_ext == '.zip':
            if output_path:
                # Save the ZIP file first
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Determine extraction directory
                if extract_dir:
                    extraction_path = extract_dir
                else:
                    extraction_path = os.path.dirname(output_path)
                
                # Extract the ZIP file
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(extraction_path)
                
                return extraction_path
            else:
                # Process ZIP in memory
                content = response.content
                zip_buffer = io.BytesIO(content)
                
                # Dictionary to store file contents
                file_contents = {}
                
                with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                    for file_info in zip_ref.infolist():
                        if file_info.is_dir():
                            continue  # Skip directories
                            
                        filename = file_info.filename
                        file_content = zip_ref.read(filename)
                        
                        # Auto-parse the content if enabled
                        if auto_parse:
                            file_ext = os.path.splitext(filename.lower())[1]
                            
                            # CSV files
                            if file_ext in ('.csv', '.tsv'):
                                # Import pandas here to ensure it's available in the exception block
                                import pandas as pd_local
                                try:
                                    delimiter = '\t' if file_ext == '.tsv' else ','
                                    # Try different encodings to handle CSV files
                                    try:
                                        file_contents[filename] = pd_local.read_csv(io.BytesIO(file_content), delimiter=delimiter)
                                        continue
                                    except UnicodeDecodeError:
                                        # Try latin-1 encoding if utf-8 fails
                                        file_contents[filename] = pd_local.read_csv(io.BytesIO(file_content), delimiter=delimiter, encoding='latin-1')
                                        continue
                                except Exception as e:
                                    # Log error for debugging
                                    import sys
                                    print(f"Error reading CSV '{filename}': {str(e)}", file=sys.stderr)
                            
                            # JSON files
                            elif file_ext == '.json':
                                try:
                                    file_contents[filename] = json.loads(file_content.decode('utf-8'))
                                    continue
                                except Exception:
                                    pass
                            
                            # Text files
                            elif file_ext in ('.txt', '.md', '.py', '.js', '.html', '.css', '.xml', '.yml', '.yaml'):
                                try:
                                    file_contents[filename] = file_content.decode('utf-8')
                                    continue
                                except Exception:
                                    pass
                            
                            # Excel files
                            elif file_ext in ('.xls', '.xlsx'):
                                try:
                                    file_contents[filename] = pd.read_excel(io.BytesIO(file_content))
                                    continue
                                except Exception:
                                    pass
                            
                            # Parquet files
                            elif file_ext == '.parquet':
                                try:
                                    file_contents[filename] = pd.read_parquet(io.BytesIO(file_content))
                                    continue
                                except Exception:
                                    pass
                        
                        # Default: store raw content
                        file_contents[filename] = file_content
                
                return file_contents
        
        # Handle normal file download (non-ZIP or ZIP without extraction)
        if output_path:
            # Save to file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return output_path
        else:
            # Get content
            content = response.content
            
            # If auto_parse is enabled, try to parse the content based on file extension
            if auto_parse:
                
                # CSV files - return pandas DataFrame
                if file_ext in ('.csv', '.tsv'):
                    try:
                        import pandas as pd
                        delimiter = '\t' if file_ext == '.tsv' else ','
                        return pd.read_csv(io.BytesIO(content), delimiter=delimiter)
                    except (ImportError, Exception) as e:
                        # If pandas is not installed or there's an error parsing, fall back to bytes
                        # print(f"Warning: Could not parse CSV file: {str(e)}")
                        pass
                
                # JSON files - return parsed JSON
                elif file_ext == '.json':
                    try:
                        return json.loads(content.decode('utf-8'))
                    except json.JSONDecodeError:
                        # If there's an error parsing JSON, fall back to bytes
                        pass
                
                # Text files - return as string
                elif file_ext in ('.txt', '.md', '.py', '.js', '.html', '.css', '.xml', '.yml', '.yaml'):
                    try:
                        return content.decode('utf-8')
                    except UnicodeDecodeError:
                        # If there's an error decoding as UTF-8, fall back to bytes
                        pass
                        
                # Excel files - return pandas DataFrame or ExcelFile
                elif file_ext in ('.xls', '.xlsx'):
                    try:
                        import pandas as pd
                        return pd.read_excel(io.BytesIO(content))
                    except (ImportError, Exception):
                        # If pandas is not installed or there's an error parsing, fall back to bytes
                        pass
                        
                # Parquet files - return pandas DataFrame
                elif file_ext == '.parquet':
                    try:
                        import pandas as pd
                        return pd.read_parquet(io.BytesIO(content))
                    except (ImportError, Exception):
                        # If pandas is not installed or there's an error parsing, fall back to bytes
                        pass
                        
                # HDF5 files - return h5py File object
                elif file_ext in ('.h5', '.hdf5'):
                    try:
                        import h5py
                        f = io.BytesIO(content)
                        return h5py.File(f, 'r')
                    except (ImportError, Exception):
                        # If h5py is not installed or there's an error parsing, fall back to bytes
                        pass
                
                # ZIP files - return zipfile.ZipFile object
                elif file_ext == '.zip' and not extract_zip:
                    try:
                        zip_buffer = io.BytesIO(content)
                        return zipfile.ZipFile(zip_buffer, 'r')
                    except Exception:
                        # If there's an error creating the ZipFile, fall back to bytes
                        pass
            
            # Return raw content if auto_parse is disabled or if parsing failed
            return content

    def get_metadata(self, 
                    object_name: str, 
                    bucket: Optional[str] = None) -> Dict[str, Any]:
        if not self.access_token:
            raise Exception("You must be logged in to get metadata")
        """
        Get metadata for a file in the Datalake.

        Args:
            object_name: Name of the object to get metadata for.
            bucket: Bucket to get metadata from. If None, the default bucket will be used.

        Returns:
            Dict containing the metadata.
        """
        url = f"{self.base_url}/api/metadata/{object_name}"
        
        params = {}
        if bucket:
            params['bucket'] = bucket
        else:
            params['bucket'] = self.default_bucket
            
        # Send request
        response = self.session.get(url, params=params)
        
        # Parse response
        if response.status_code == 200:
            return response.json()
        else:
            response_data = response.json()
            error_message = response_data.get('error', 'Unknown error')
            raise Exception(f"Failed to get metadata: {error_message}")

    def delete_file(self, 
                   object_name: str, 
                   bucket: Optional[str] = None) -> Dict[str, Any]:
        if not self.access_token:
            raise Exception("You must be logged in to delete files")
        """
        Delete a file from the Datalake.

        Args:
            object_name: Name of the object to delete.
            bucket: Bucket to delete from. If None, the default bucket will be used.

        Returns:
            Dict containing the result of the delete operation.
        """
        url = f"{self.base_url}/api/files/{object_name}"
        
        form_data = {}
        if bucket:
            form_data['bucket'] = bucket
        else:
            form_data['bucket'] = self.default_bucket
            
        # Send request
        response = self.session.delete(url, data=form_data)
        
        # Parse response
        if response.status_code == 200:
            return response.json()
        else:
            response_data = response.json()
            error_message = response_data.get('error', 'Unknown error')
            raise Exception(f"Delete failed: {error_message}")

    def update_file(self, 
                   object_name: str, 
                   file_path: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None, 
                   bucket: Optional[str] = None,
                   prefix: Optional[str] = None,
                   metadata_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.access_token:
            raise Exception("You must be logged in to update files")
        """
        Update a file in the Datalake.

        Args:
            object_name: Name of the object to update.
            file_path: Path to the new file content. If None, only metadata will be updated.
            metadata: New metadata to associate with the file.
            bucket: Bucket to update in. If None, the default bucket will be used.
            prefix: New prefix to use for the object in the Datalake.
            metadata_id: ID of the metadata to update.

        Returns:
            Dict containing the result of the update operation.
        """
        url = f"{self.base_url}/api/files/{object_name}"
        
        # Prepare form data
        form_data = {}
        if bucket:
            form_data['bucket'] = bucket
        else:
            form_data['bucket'] = self.default_bucket
            
        if prefix:
            form_data['prefix'] = prefix
            
        if metadata_id:
            form_data['metadata_id'] = metadata_id
            
        if metadata:
            form_data['metadata'] = json.dumps(metadata)
            
        # Prepare files
        files = {}
        if file_path:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            files = {'file': (os.path.basename(file_path), open(file_path, 'rb'))}
            
        # Send request
        try:
            response = self.session.put(url, data=form_data, files=files)
            
            # Close file if opened
            if file_path and 'file' in files:
                files['file'][1].close()
                
            # Parse response
            if response.status_code == 200:
                return response.json()
            else:
                response_data = response.json()
                error_message = response_data.get('error', 'Unknown error')
                raise Exception(f"Update failed: {error_message}")
        except Exception as e:
            # Close file if opened
            if file_path and 'file' in files:
                files['file'][1].close()
            raise e

    def update_metadata(self, 
                       metadata_id: str, 
                       metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not self.access_token:
            raise Exception("You must be logged in to update metadata")
        """
        Update metadata for a file in the Datalake.

        Args:
            metadata_id: ID of the metadata to update.
            metadata: New metadata to associate with the file.

        Returns:
            Dict containing the result of the update operation.
        """
        url = f"{self.base_url}/api/metadata/{metadata_id}"
        
        # Prepare data
        data = {'metadata': json.dumps(metadata)}
            
        # Send request
        response = self.session.put(url, data=data)
        
        # Parse response
        if response.status_code == 200:
            return response.json()
        else:
            response_data = response.json()
            error_message = response_data.get('error', 'Unknown error')
            raise Exception(f"Metadata update failed: {error_message}")

    def search_files(self, 
                    query: Dict[str, Any], 
                    bucket: Optional[str] = None) -> Dict[str, Any]:
        if not self.access_token:
            raise Exception("You must be logged in to search files")
        """
        Search for files in the Datalake.

        Args:
            query: Query to search for.
            bucket: Bucket to search in. If None, the default bucket will be used.

        Returns:
            Dict containing the search results.
        """
        url = f"{self.base_url}/api/search"
        
        # Prepare data
        json_data = {'query': query}
        if bucket:
            json_data['bucket'] = bucket
        else:
            json_data['bucket'] = self.default_bucket
            
        # Send request
        response = self.session.post(url, json=json_data)
        
        # Parse response
        if response.status_code == 200:
            return response.json()
        else:
            response_data = response.json()
            error_message = response_data.get('error', 'Unknown error')
            raise Exception(f"Search failed: {error_message}")

    def query_files(self, 
                   query: Dict[str, Any], 
                   bucket: Optional[str] = None,
                   output_path: Optional[str] = None) -> Union[str, bytes]:
        if not self.access_token:
            raise Exception("You must be logged in to query files")
        """
        Query files and get a ZIP archive of the results.

        Args:
            query: Query to search for.
            bucket: Bucket to search in. If None, the default bucket will be used.
            output_path: Path to save the ZIP archive. If None, the archive content will be returned.

        Returns:
            Path to the ZIP archive if output_path is provided, otherwise the archive content.
        """
        url = f"{self.base_url}/api/query"
        
        # Prepare data
        data = {'query': query}
        if bucket:
            data['bucket'] = bucket
        else:
            data['bucket'] = self.default_bucket
            
        # Send request
        response = self.session.post(url, json=data, stream=True)
        
        # Check response
        if response.status_code != 200:
            try:
                response_data = response.json()
                error_message = response_data.get('error', 'Unknown error')
            except:
                error_message = f"HTTP error {response.status_code}"
            raise Exception(f"Query failed: {error_message}")
            
        # Return the JSON response instead of downloading
        try:
            return response.json()
        except:
            # If not JSON, return the content as is
            return {"content": response.content}
            
        # # Handle response - COMMENTED OUT DOWNLOAD FUNCTIONALITY
        # if output_path:
        #     # Save to file
        #     with open(output_path, 'wb') as f:
        #         for chunk in response.iter_content(chunk_size=8192):
        #             f.write(chunk)
        #     return output_path
        # else:
        #     # Return content
        #     return response.content
