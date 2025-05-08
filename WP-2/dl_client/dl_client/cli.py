"""
Command Line Interface for the Datalake Client

This module provides a command-line interface for interacting with the AIND Datalake API.
"""

import os
import sys
import json
import click
from typing import Dict, Any, Optional
from tqdm import tqdm

from .client import DatalakeClient


class ProgressFileReader:
    """File reader with progress bar."""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.file = open(file_path, 'rb')
        self.progress_bar = tqdm(total=self.file_size, unit='B', unit_scale=True, desc=os.path.basename(file_path))
        self.bytes_read = 0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        chunk = self.file.read(8192)
        if not chunk:
            self.file.close()
            self.progress_bar.close()
            raise StopIteration
        self.bytes_read += len(chunk)
        self.progress_bar.update(len(chunk))
        return chunk


@click.group()
@click.option('--base-url', default='http://131.175.206.61:5000', help='Base URL of the Datalake API')
@click.option('--bucket', default='aind', help='Default bucket to use')
@click.option('--username', help='Username for authentication')
@click.option('--password', help='Password for authentication')
@click.option('--config', help='Path to config file with credentials')
@click.option('--no-auto-login', is_flag=True, help='Disable automatic login from config files')
@click.pass_context
def main(ctx, base_url, bucket, username, password, config, no_auto_login):
    """Command-line interface for the AIND Datalake API."""
    ctx.ensure_object(dict)
    
    if no_auto_login:
        # Initialize without auto-login
        ctx.obj['client'] = DatalakeClient(base_url=base_url, default_bucket=bucket)
        # Manual login if credentials provided
        if username and password:
            ctx.obj['client'].login(username, password)
    else:
        # Initialize with auto-login
        ctx.obj['client'] = DatalakeClient(
            base_url=base_url, 
            default_bucket=bucket, 
            username=username, 
            password=password,
            config_file=config
        )


@main.command('login')
@click.argument('username')
@click.password_option(help='Password for authentication')
@click.option('--save', is_flag=True, help='Save credentials to config file')
@click.option('--config-path', help='Path where to save the config file')
@click.pass_context
def login(ctx, username, password, save, config_path):
    """Login to the Datalake API."""
    client = ctx.obj['client']
    
    try:
        result = client.login(username, password)
        click.echo("Login successful!")
        
        # Save credentials if requested
        if save:
            saved_path = client.save_credentials(config_path)
            click.echo(f"Credentials saved to: {saved_path}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command('register')
@click.argument('username')
@click.password_option(help='Password for the new user')
@click.option('--role', default='user', help='Role for the new user')
@click.option('--save', is_flag=True, help='Save credentials to config file after registration')
@click.option('--config-path', help='Path where to save the config file')
@click.pass_context
def register_user(ctx, username, password, role, save, config_path):
    """Register a new user (requires admin privileges)."""
    client = ctx.obj['client']
    
    try:
        result = client.register_user(username, password, role)
        click.echo(f"User '{username}' registered successfully!")
        
        # Login and save credentials if requested
        if save:
            client.login(username, password)
            saved_path = client.save_credentials(config_path)
            click.echo(f"Credentials saved to: {saved_path}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command('save-config')
@click.option('--config-path', help='Path where to save the config file')
@click.pass_context
def save_config(ctx, config_path):
    """Save current credentials to a config file."""
    client = ctx.obj['client']
    
    try:
        saved_path = client.save_credentials(config_path)
        click.echo(f"Credentials saved to: {saved_path}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command('upload')
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--object-name', help='Name to use for the object in the Datalake')
@click.option('--bucket', help='Bucket to upload to')
@click.option('--prefix', help='Prefix to use for the object in the Datalake')
@click.option('--metadata', help='JSON metadata string to associate with the file')
@click.option('--metadata-file', type=click.Path(exists=True), help='Path to a JSON file containing metadata')
@click.pass_context
def upload_file(ctx, file_path, object_name, bucket, prefix, metadata, metadata_file):
    """Upload a file to the Datalake."""
    client = ctx.obj['client']
    
    # Parse metadata if provided
    metadata_dict = None
    
    # Check if metadata file is provided
    if metadata_file:
        try:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
        except json.JSONDecodeError:
            click.echo(f"Error: Invalid JSON in metadata file: {metadata_file}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error reading metadata file: {str(e)}", err=True)
            sys.exit(1)
    # Otherwise check if metadata string is provided
    elif metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            click.echo(f"Error: Invalid JSON metadata: {metadata}", err=True)
            sys.exit(1)
    
    try:
        result = client.upload_file(
            file_path=file_path,
            object_name=object_name,
            bucket=bucket,
            prefix=prefix,
            metadata=metadata_dict
        )
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command('download')
@click.argument('object_name')
@click.option('--output', help='Path to save the downloaded file')
@click.option('--bucket', help='Bucket to download from')
@click.pass_context
def download_file(ctx, object_name, output, bucket):
    """Download a file from the Datalake."""
    client = ctx.obj['client']
    
    try:
        if output:
            result = client.download_file(
                object_name=object_name,
                output_path=output,
                bucket=bucket,
                as_attachment=True
            )
            click.echo(f"File downloaded to: {result}")
        else:
            # If no output path is provided, use the object name as the output path
            filename = os.path.basename(object_name)
            result = client.download_file(
                object_name=object_name,
                output_path=filename,
                bucket=bucket,
                as_attachment=True
            )
            click.echo(f"File downloaded to: {result}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command('metadata')
@click.argument('object_name')
@click.option('--bucket', help='Bucket to get metadata from')
@click.pass_context
def get_metadata(ctx, object_name, bucket):
    """Get metadata for a file in the Datalake."""
    client = ctx.obj['client']
    
    try:
        result = client.get_metadata(
            object_name=object_name,
            bucket=bucket
        )
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command('delete')
@click.argument('object_name')
@click.option('--bucket', help='Bucket to delete from')
@click.confirmation_option(prompt='Are you sure you want to delete this file?')
@click.pass_context
def delete_file(ctx, object_name, bucket):
    """Delete a file from the Datalake."""
    client = ctx.obj['client']
    
    try:
        result = client.delete_file(
            object_name=object_name,
            bucket=bucket
        )
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command('update')
@click.argument('object_name')
@click.option('--file', 'file_path', type=click.Path(exists=True), help='Path to the new file content')
@click.option('--metadata', help='JSON metadata string to associate with the file')
@click.option('--metadata-file', type=click.Path(exists=True), help='Path to a JSON file containing metadata')
@click.option('--bucket', help='Bucket to update in')
@click.option('--prefix', help='New prefix to use for the object in the Datalake')
@click.option('--metadata-id', help='ID of the metadata to update')
@click.pass_context
def update_file(ctx, object_name, file_path, metadata, metadata_file, bucket, prefix, metadata_id):
    """Update a file in the Datalake."""
    client = ctx.obj['client']
    
    # Parse metadata if provided
    metadata_dict = None
    
    # Check if metadata file is provided
    if metadata_file:
        try:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
        except json.JSONDecodeError:
            click.echo(f"Error: Invalid JSON in metadata file: {metadata_file}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error reading metadata file: {str(e)}", err=True)
            sys.exit(1)
    # Otherwise check if metadata string is provided
    elif metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            click.echo(f"Error: Invalid JSON metadata: {metadata}", err=True)
            sys.exit(1)
    
    try:
        result = client.update_file(
            object_name=object_name,
            file_path=file_path,
            metadata=metadata_dict,
            bucket=bucket,
            prefix=prefix,
            metadata_id=metadata_id
        )
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command('update-metadata')
@click.argument('metadata_id')
@click.option('--metadata', help='JSON metadata string to update')
@click.option('--metadata-file', type=click.Path(exists=True), help='Path to a JSON file containing metadata')
@click.pass_context
def update_metadata(ctx, metadata_id, metadata, metadata_file):
    """Update metadata for a file in the Datalake."""
    client = ctx.obj['client']
    
    # Check if at least one metadata source is provided
    if not metadata and not metadata_file:
        click.echo("Error: Either --metadata or --metadata-file must be provided", err=True)
        sys.exit(1)
    
    # Parse metadata
    metadata_dict = None
    
    # Check if metadata file is provided
    if metadata_file:
        try:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
        except json.JSONDecodeError:
            click.echo(f"Error: Invalid JSON in metadata file: {metadata_file}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error reading metadata file: {str(e)}", err=True)
            sys.exit(1)
    # Otherwise use metadata string
    else:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            click.echo(f"Error: Invalid JSON metadata: {metadata}", err=True)
            sys.exit(1)
    
    try:
        result = client.update_metadata(
            metadata_id=metadata_id,
            metadata=metadata_dict
        )
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command('search')
@click.option('--query', help='JSON query string')
@click.option('--query-file', type=click.Path(exists=True), help='Path to a JSON file containing the query')
@click.option('--bucket', help='Bucket to search in')
@click.pass_context
def search_files(ctx, query, query_file, bucket):
    """Search for files in the Datalake."""
    client = ctx.obj['client']
    
    # Check if at least one query source is provided
    if not query and not query_file:
        click.echo("Error: Either --query or --query-file must be provided", err=True)
        sys.exit(1)
    
    # Parse query
    query_dict = None
    
    # Check if query file is provided
    if query_file:
        try:
            with open(query_file, 'r') as f:
                query_dict = json.load(f)
        except json.JSONDecodeError:
            click.echo(f"Error: Invalid JSON in query file: {query_file}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error reading query file: {str(e)}", err=True)
            sys.exit(1)
    # Otherwise use query string
    else:
        try:
            query_dict = json.loads(query)
        except json.JSONDecodeError:
            click.echo(f"Error: Invalid JSON query: {query}", err=True)
            sys.exit(1)
    
    try:
        result = client.search_files(
            query=query_dict,
            bucket=bucket
        )
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command('query')
@click.option('--query', help='JSON query string')
@click.option('--query-file', type=click.Path(exists=True), help='Path to a JSON file containing the query')
@click.option('--output', help='Path to save the ZIP archive')
@click.option('--bucket', help='Bucket to search in')
@click.pass_context
def query_files(ctx, query, query_file, output, bucket):
    """Query files and get a ZIP archive of the results."""
    client = ctx.obj['client']
    
    # Check if at least one query source is provided
    if not query and not query_file:
        click.echo("Error: Either --query or --query-file must be provided", err=True)
        sys.exit(1)
    
    # Parse query
    query_dict = None
    
    # Check if query file is provided
    if query_file:
        try:
            with open(query_file, 'r') as f:
                query_dict = json.load(f)
        except json.JSONDecodeError:
            click.echo(f"Error: Invalid JSON in query file: {query_file}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error reading query file: {str(e)}", err=True)
            sys.exit(1)
    # Otherwise use query string
    else:
        try:
            query_dict = json.loads(query)
        except json.JSONDecodeError:
            click.echo(f"Error: Invalid JSON query: {query}", err=True)
            sys.exit(1)
    
    try:
        if output:
            result = client.query_files(
                query=query_dict,
                bucket=bucket,
                output_path=output
            )
            click.echo(f"ZIP archive downloaded to: {result}")
        else:
            # If no output path is provided, use a default name
            result = client.query_files(
                query=query_dict,
                bucket=bucket,
                output_path="query_results.zip"
            )
            click.echo(f"ZIP archive downloaded to: {result}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main(obj={})
