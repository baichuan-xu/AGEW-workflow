import requests
import xml.etree.ElementTree as ET
import os
from urllib.parse import urlparse
import time

def download_image(url, folder_path, filename):
    """Download an image from URL and save it with given filename in the specified folder"""
    try:
        headers = {
            'X-ELS-APIKey': os.getenv('ELSEVIER_API_KEY'),  # Replace with your actual API key
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded: {file_path}")
            return True
        else:
            print(f"Failed to download {filename}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def extract_and_download_images(xml_file, folder_path):
    """Extract and download images from Elsevier XML file and save them in the same folder"""
    # Parse XML file
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {str(e)}")
        return

    # Find all object elements with category="high" (high-resolution images)
    namespaces = {
        'default': 'http://www.elsevier.com/xml/svapi/article/dtd'
    }
    
    objects = root.findall('.//default:object[@category="high"]', namespaces)
    
    if not objects:
        print(f"No high-resolution images found in the XML file: {xml_file}")
        return

    # Download each image
    for i, obj in enumerate(objects, 1):
        url = obj.text.strip()
        if not url:
            continue

        # Extract ref attribute for naming
        ref = obj.get('ref', f'image_{i}')
        
        # Create filename from ref attribute
        filename = f"{ref}.jpg"

        # Download image into the same folder as the XML file
        success = download_image(url, folder_path, filename)
        
        # Add delay between downloads to avoid overwhelming the server
        if success:
            time.sleep(1)

def process_xml_directory(base_path):
    """Process all XML files in subfolders L001 to L104"""
    for i in range(1, 3317):  # From 1 to 104 inclusive
        folder_name = f"L{i:04d}"  # Format folder name with leading zeros (e.g., L001, L002)
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Find the XML file in the folder
            xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
            
            if xml_files:
                xml_file = os.path.join(folder_path, xml_files[0])  # Assume only one XML file per folder
                
                # Process the XML file and download images into the same folder
                print(f"\nProcessing folder: {folder_name}")
                extract_and_download_images(xml_file, folder_path)
            else:
                print(f"No XML file found in folder: {folder_path}")
        else:
            print(f"Folder does not exist: {folder_path}")

# Example usage
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    process_xml_directory(base_path)

print("All operations completed.")