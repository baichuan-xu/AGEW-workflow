import requests
import os
import time
from bs4 import BeautifulSoup
import re
import json
from xml.etree import ElementTree as ET

# API Configuration
API_KEY = os.getenv('ELSEVIER_API_KEY')  # Replace with your actual API key
SCOPUS_SEARCH_URL = 'https://api.elsevier.com/content/search/scopus'
SCIDIR_ARTICLE_URL = 'https://api.elsevier.com/content/article/doi/'

# XML namespaces used by Elsevier
NAMESPACES = {
    'svapi': 'http://www.elsevier.com/xml/svapi/article/dtd',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'atom': 'http://www.w3.org/2005/Atom',
    'prism': 'http://prismstandard.org/namespaces/basic/2.0/'
}

XML_HEADERS = {
    'X-ELS-APIKey': API_KEY,
    'Accept': 'text/xml',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

PROGRESS_FILE = 'download_progress.json'  # File to track progress
SAVE_PATH = r'.\HEA_finetune\Yield_Strength'  # Base path for subfolders

def verify_xml_content(xml_content):
    """
    Verify XML content with Elsevier-specific structure
    Returns: (is_valid, message)
    """
    try:
        root = ET.fromstring(xml_content)
        # Check for originalText element
        original_text = root.find('.//svapi:originalText', NAMESPACES)
        if original_text is not None and original_text.text:
            return True, "XML验证通过"
        # If no originalText, check for coredata
        coredata = root.find('.//svapi:coredata', NAMESPACES)
        if coredata is not None:
            return True, "XML验证通过 (仅包含核心数据)"
        return False, "找不到文章内容或核心数据"
    except ET.ParseError as e:
        return False, f"XML解析错误: {str(e)}"
    except Exception as e:
        return False, f"验证错误: {str(e)}"

def download_with_session(url, headers, session):
    """Download using session with multiple retries"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, timeout=300)
            if response.status_code == 200:
                return response
            else:
                print(f"HTTP错误 {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"尝试 {attempt + 1} 失败: {str(e)}")
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    return None

def download_article_xml(doi, title, subfolder_path, session):
    """Download article in XML format with verification and save to subfolder"""
    os.makedirs(subfolder_path, exist_ok=True)  # Ensure the subfolder exists
    xml_path = os.path.join(subfolder_path, "article.xml")
    json_path = os.path.join(subfolder_path, "metadata.json")
    url = f'{SCIDIR_ARTICLE_URL}{doi}'
    print(f"\n尝试下载: {url}")
    response = download_with_session(url, XML_HEADERS, session)
    if not response:
        print(f"下载失败: {title}")
        return False
    if response.status_code == 404:
        print(f"资源未找到 (404): {doi}")
        return False
    xml_content = response.text
    print(f"获取到XML响应，大小: {len(xml_content)} 字节")
    is_valid, message = verify_xml_content(xml_content)
    if is_valid:
        # Save XML content
        try:
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
        except Exception as e:
            print(f"保存XML文件时出错: {str(e)}")
            return False
        # Extract and save metadata
        try:
            root = ET.fromstring(xml_content)
            metadata = {
                'title': title,
                'doi': doi,
                'size': len(xml_content),
                'download_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'has_full_text': root.find('.//svapi:originalText', NAMESPACES) is not None
            }
            # Try to extract additional metadata
            coredata = root.find('.//svapi:coredata', NAMESPACES)
            if coredata is not None:
                for child in coredata:
                    tag = child.tag.split('}')[-1]  # Remove namespace
                    metadata[f'coredata_{tag}'] = child.text if child.text else ''
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"成功下载: {title}")
            print(f"XML大小: {len(xml_content)/1024:.2f}KB")
            return True
        except Exception as e:
            print(f"保存元数据时出错: {str(e)}")
            return False
    else:
        print(f"XML验证失败: {title}")
        print(f"原因: {message}")
        return False

def clean_filename(title):
    """
    Clean filename by removing illegal characters and truncating length
    """
    illegal_chars = r'[<>:"/\\|?*]'
    clean_title = re.sub(illegal_chars, '_', title)
    clean_title = re.sub(r'_+', '_', clean_title)
    clean_title = clean_title.strip()[:240]
    return clean_title

def load_progress():
    """Load progress from file"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'successful_downloads': [], 'failed_downloads': []}

def save_progress(successful_downloads, failed_downloads):
    """Save progress to file"""
    progress = {
        'successful_downloads': successful_downloads,
        'failed_downloads': failed_downloads
    }
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def main():
    os.makedirs(SAVE_PATH, exist_ok=True)
    session = requests.Session()

    # Load progress
    progress = load_progress()
    successful_downloads = progress['successful_downloads']
    failed_downloads = progress['failed_downloads']

    # Search parameters
    query = 'high entropy alloy AND yield strength'  # Search for both topics
    count_per_request = 100  # Maximum allowed by the API
    total_results = 5000   # Total results to retrieve
    start = len(successful_downloads)  # Resume from last successful download

    while start < total_results:
        print(f"\n正在检索文献 (从第 {start + 1} 条开始)...")
        search_params = {
            'query': query,
            'count': count_per_request,
            'start': start,
            'field': 'doi,title,description',
            'sort': 'relevancy'
        }

        response = session.get(
            SCOPUS_SEARCH_URL,
            headers={'X-ELS-APIKey': API_KEY, 'Accept': 'application/json'},
            params=search_params
        )

        if response.status_code != 200:
            print(f"检索失败: {response.status_code}")
            break

        results = response.json()
        entries = results.get('search-results', {}).get('entry', [])
        total_results = int(results.get('search-results', {}).get('opensearch:totalResults', 0))

        if not entries:
            print("没有更多结果")
            break

        print(f"找到 {len(entries)} 篇文献 (总共 {total_results} 篇)")

        for i, entry in enumerate(entries, 1):
            doi = entry.get('prism:doi')
            title = entry.get('dc:title', '未知标题')

            if isinstance(title, str):
                title = BeautifulSoup(title, 'html.parser').get_text().strip()

            print(f"\n处理第 {start + i}/{total_results} 篇文献:")
            print(f"标题: {title}")
            print(f"DOI: {doi}")

            if doi and doi.startswith('10.1016/'):  # Ensure DOI belongs to Elsevier
                if doi not in [item['doi'] for item in successful_downloads]:
                    try:
                        # Create subfolder for this article
                        subfolder_name = f"L{len(successful_downloads) + 1:04d}"
                        subfolder_path = os.path.join(SAVE_PATH, subfolder_name)
                        if download_article_xml(doi, title, subfolder_path, session):
                            successful_downloads.append({'title': title, 'doi': doi, 'folder': subfolder_name})
                            save_progress(successful_downloads, failed_downloads)
                            time.sleep(2)  # Avoid rate limiting
                        else:
                            failed_downloads.append({'title': title, 'doi': doi})
                            save_progress(successful_downloads, failed_downloads)
                    except Exception as e:
                        print(f"处理文章时发生错误: {str(e)}")
                        failed_downloads.append({'title': title, 'doi': doi})
                        save_progress(successful_downloads, failed_downloads)
                else:
                    print("已下载过，跳过")
            else:
                print("非Elsevier DOI或无效DOI，跳过")

        # Move to the next page
        start += count_per_request

    # Save final progress
    save_progress(successful_downloads, failed_downloads)

    print(f"\n下载完成！成功下载 {len(successful_downloads)}/{total_results} 篇文献")
    if failed_downloads:
        print(f"失败记录已保存至: {PROGRESS_FILE}")
    else:
        print("没有下载失败的文献")

if __name__ == "__main__":
    main()