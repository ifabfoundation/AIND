# DL Client

A comprehensive client library for interacting with the AIND Datalake REST API with JWT authentication support.

## Overview

The DL Client provides a convenient way to interact with the AIND Datalake API, allowing you to:

- Authenticate using JWT tokens
- Upload and download files
- Manage file metadata
- Search and query files based on metadata
- Update existing files and their metadata
- Delete files from the datalake

The library can be used both as a Python module in your scripts and as a command-line tool, with flexible authentication options including config file support.

## Installation

### Prerequisites

- Python 3.6 or higher
- pip package manager

### Installing from local directory

```bash
# Navigate to the dl_client directory
cd dl_client

# Install the package
pip install .
```

### Verifying installation

After installation, you should be able to run the CLI tool:

```bash
dl-client --help
```

## Authentication

The library supports JWT authentication with multiple authentication methods:

### Authentication Methods

1. **Direct credentials**: Provide username and password directly
2. **Config file**: Store credentials in a config file
3. **Auto-login**: Automatically detect credentials from standard locations

### Config File Locations

The client will automatically look for credentials in these locations (in order):

1. `~/.config/dl_client/config.ini` (User's home directory)
2. `.dl_client.ini` (Current working directory)
3. Path specified in `DL_CLIENT_CONFIG` environment variable

### Config File Format

```ini
[credentials]
username = your_username
password = your_password
```

## Python Library Usage

### Initializing the client

```python
from dl_client import DatalakeClient

# Initialize with default settings (will try auto-login)
client = DatalakeClient()

# Initialize with explicit credentials
client = DatalakeClient(
    base_url="http://api.example.com:5000",  # API server URL
    default_bucket="custom-bucket",          # Default storage bucket
    username="admin",                        # Username for authentication
    password="password123"                   # Password for authentication
)

# Initialize with a specific config file
client = DatalakeClient(
    config_file="/path/to/config.ini"        # Path to config file with credentials
)
```

### Authentication

```python
# Login explicitly
client.login("username", "password")

# Save credentials to config file
client.save_credentials()
# Or to a specific path
client.save_credentials("/path/to/config.ini")

# Register a new user (requires admin privileges)
client.register_user("new_user", "password", role="user")
```

### File Operations

#### Uploading files

```python
# Basic upload
result = client.upload_file(file_path="/path/to/data.csv")

# Upload with custom parameters
result = client.upload_file(
    file_path="/path/to/data.csv",
    object_name="dataset.csv",              # Custom name in storage
    bucket="research-data",                 # Custom bucket
    prefix="project/experiment1",           # Path prefix in storage
    metadata={                              # Custom metadata
        "project": "genomics",
        "date": "2025-04-18",
        "author": "researcher@example.com",
        "tags": ["raw", "experiment1"]
    }
)
print(f"File uploaded with ID: {result['metadata_id']}")
```

> **Note**: All API operations require authentication. If you're not authenticated, the client will attempt to auto-login or raise an exception.

#### Downloading files

```python
# Download to specific path
client.download_file(
    object_name="project/experiment1/dataset.csv",
    output_path="/local/downloads/dataset.csv",
    bucket="research-data"
)

# Download and get file content as bytes
file_content = client.download_file(
    object_name="project/experiment1/dataset.csv",
    bucket="research-data"
)
```

#### Getting file metadata

```python
metadata = client.get_metadata(
    object_name="project/experiment1/dataset.csv",
    bucket="research-data"
)
print(f"File size: {metadata['data']['size']} bytes")
print(f"Upload date: {metadata['data']['upload_date']}")
print(f"Custom metadata: {metadata['data']['metadata']}")
```

#### Updating files

```python
# Update both file content and metadata
result = client.update_file(
    object_name="project/experiment1/dataset.csv",
    file_path="/path/to/updated_data.csv",
    metadata={"status": "updated", "version": "2.0"},
    bucket="research-data"
)

# Update metadata only
result = client.update_file(
    object_name="project/experiment1/dataset.csv",
    metadata={"status": "reviewed", "version": "2.1"},
    bucket="research-data"
)

# Update metadata by ID
result = client.update_metadata(
    metadata_id="60f8c1d5e6a3b2c1a9f8e7d6",
    metadata={"status": "approved", "version": "final"}
)
```

#### Deleting files

```python
result = client.delete_file(
    object_name="project/experiment1/dataset.csv",
    bucket="research-data"
)
print(f"File deleted: {result['success']}")
```

### Search and Query Operations

#### Searching files by metadata

```python
# Search for files with specific metadata
results = client.search_files(
    query={
        "metadata.project": "genomics",
        "metadata.tags": "experiment1"
    },
    bucket="research-data"
)

# Process search results
for file in results['files']:
    print(f"Found file: {file['object_name']}")
    print(f"Metadata: {file['metadata']}")
```

#### Querying and downloading multiple files as ZIP

```python
# Query files and download as ZIP archive
zip_path = client.query_files(
    query={
        "metadata.project": "genomics",
        "metadata.status": "approved"
    },
    bucket="research-data",
    output_path="/local/downloads/approved_files.zip"
)
print(f"ZIP archive downloaded to: {zip_path}")
```

## Command Line Interface

The library provides a comprehensive command-line interface for all operations.

### Global Options

These options can be used with any command:

```bash
--base-url TEXT     Base URL of the Datalake API (default: http://131.175.206.61:5000)
--bucket TEXT       Default bucket to use (default: aind)
--username TEXT     Username for authentication
--password TEXT     Password for authentication
--config TEXT       Path to config file with credentials
--no-auto-login     Disable automatic login from config files
--help              Show help message and exit
```

### Available Commands

#### Authentication Commands

```bash
# Login to the API
dl-client login username

# Login and save credentials to config file
dl-client login username --save

# Register a new user (requires admin privileges)
dl-client register newuser --role user

# Save current credentials to config file
dl-client save-config

# Use credentials from a specific config file
dl-client --config /path/to/config.ini upload file.txt
```

#### Getting Help

```bash
# General help
dl-client --help

# Command-specific help
dl-client upload --help
```

#### File Upload

```bash
# Basic upload
dl-client upload /path/to/data.csv

# Upload with metadata as JSON string
dl-client upload /path/to/data.csv \
    --object-name dataset.csv \
    --bucket research-data \
    --prefix project/experiment1 \
    --metadata '{"project": "genomics", "tags": ["raw", "experiment1"]}'
    
# Upload with metadata from a JSON file
dl-client upload /path/to/data.csv \
    --object-name dataset.csv \
    --metadata-file metadata.json
```

Where `metadata.json` contains:
```json
{
  "project": "genomics",
  "date": "2025-04-18",
  "tags": ["raw", "experiment1"]
}
```

#### File Download

```bash
# Basic download (saves to current directory with original filename)
dl-client download project/experiment1/dataset.csv

# Download with options
dl-client download project/experiment1/dataset.csv \
    --output /local/downloads/dataset.csv \
    --bucket research-data
```

#### Get Metadata

```bash
# Get file metadata
dl-client metadata project/experiment1/dataset.csv --bucket research-data
```

#### Update File

```bash
# Update file content and/or metadata
dl-client update project/experiment1/dataset.csv \
    --file /path/to/updated_data.csv \
    --metadata '{"status": "updated", "version": "2.0"}' \
    --bucket research-data
    
# Update using metadata from a JSON file
dl-client update project/experiment1/dataset.csv \
    --file /path/to/updated_data.csv \
    --metadata-file updated_metadata.json
```

#### Update Metadata

```bash
# Update metadata by ID using JSON string
dl-client update-metadata 60f8c1d5e6a3b2c1a9f8e7d6 \
    --metadata '{"status": "approved", "version": "final"}'
    
# Update metadata by ID using JSON file
dl-client update-metadata 60f8c1d5e6a3b2c1a9f8e7d6 \
    --metadata-file metadata_updates.json
```

#### Delete File

```bash
# Delete a file (with confirmation prompt)
dl-client delete project/experiment1/dataset.csv --bucket research-data
```

#### Search Files

```bash
# Search for files by metadata using JSON string
dl-client search --query '{"metadata.project": "genomics", "metadata.tags": "experiment1"}' \
    --bucket research-data
    
# Search using a query from a JSON file
dl-client search --query-file search_criteria.json \
    --bucket research-data
```

#### Query Files (ZIP download)

```bash
# Query files using JSON string and download as ZIP
dl-client query --query '{"metadata.project": "genomics", "metadata.status": "approved"}' \
    --output /local/downloads/approved_files.zip \
    --bucket research-data
    
# Query using a JSON file
dl-client query --query-file query_criteria.json \
    --output /local/downloads/approved_files.zip
```

## Error Handling

The library provides comprehensive error handling:

- HTTP errors are converted to Python exceptions with meaningful messages
- Authentication failures are clearly reported
- CLI commands return appropriate exit codes and error messages
- Validation is performed on inputs before sending requests
- JSON parsing errors for metadata and query files are reported with details

## Security Features

### Password Security

- The server uses bcrypt for secure password hashing
- Passwords are never stored in plaintext
- Each password is automatically salted to prevent rainbow table attacks

### JWT Authentication

- All API endpoints are protected with JWT authentication
- Tokens expire after a configurable period (default: 1 day)
- Token-based authentication eliminates the need to send credentials with every request

### Credential Storage

- Credentials can be stored in config files for convenience
- Multiple config locations are supported for different use cases
- Environment variable support for CI/CD pipelines

## Development

### Project Structure

```
dl_client/
├── dl_client/
│   ├── __init__.py      # Package initialization
│   ├── client.py        # DatalakeClient implementation with authentication
│   └── cli.py           # Command-line interface with config support
├── setup.py             # Package installation configuration
└── README.md            # Documentation
```

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
