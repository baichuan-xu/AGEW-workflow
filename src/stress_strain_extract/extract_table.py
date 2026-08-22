from lxml import etree
from collections import OrderedDict
import os
import re

# Define helper functions (unchanged from the original code)

NS = {
    'ce': 'http://www.elsevier.com/xml/common/dtd',
    'cals': 'http://www.elsevier.com/xml/common/cals/dtd'
}

def clean_text(text):
    """增强版文本清洗"""
    text = re.sub(r'\s+', ' ', text)
    replacements = OrderedDict([('–', '-'), ('“', '"'), ('”', '"')])
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()

def extract_text(element):
    """递归提取文本"""
    text = element.text.strip() + " " if element.text else ""
    for child in element:
        text += extract_text(child)
        if child.tail and child.tail.strip():
            text += child.tail.strip() + " "
    return clean_text(text)

def extract_table(table_element):
    """修复表格提取逻辑"""
    table_data = []
    
    # 提取表头
    thead = table_element.find(".//cals:thead", NS)
    if thead:
        headers = []
        # 关键修复：改用ce命名空间
        for entry in thead.xpath(".//ce:entry", namespaces=NS):
            headers.append(extract_text(entry))
        if headers:
            table_data.append(headers)
    
    # 提取表体
    tbody = table_element.find(".//cals:tbody", NS)
    if tbody:
        for row in tbody.xpath(".//cals:row", namespaces=NS):
            row_data = []
            # 关键修复：改用ce命名空间
            for entry in row.xpath(".//ce:entry", namespaces=NS):
                row_data.append(extract_text(entry))
            if row_data:
                table_data.append(row_data)
    return table_data

def xml_to_txt(xml_file, output_file):
    try:
        tree = etree.parse(xml_file)
    except Exception as e:
        print(f"XML解析错误: {str(e)}")
        return

    output_content = []
    
    # 提取表格
    tables = []
    for table in tree.xpath('//ce:table', namespaces=NS):
        if table_data := extract_table(table):
            tables.append(table_data)
    
    if tables:
        output_content.append("\n" + "="*40 + " TABLES " + "="*40 + "\n")
        for idx, table in enumerate(tables, 1):
            output_content.append(f"Table {idx}:\n")
            for row in table:
                output_content.append(" | ".join(row) + "\n")
            output_content.append("\n")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output_content)
        print(f"转换成功！输出文件: {output_file}")
    except Exception as e:
        print(f"文件写入错误: {str(e)}")

# Main logic to process all subfolders
if __name__ == "__main__":
    base_path = r".\stress-strain2"
    
    # Loop through subfolders L001 to L104
    for i in range(1, 3356):  # From 1 to 104 inclusive
        folder_name = f"L{i:04d}"  # Format folder name with leading zeros (e.g., L001, L002)
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Find the XML file in the folder
            xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
            
            if xml_files:
                xml_file = os.path.join(folder_path, xml_files[0])  # Assume only one XML file per folder
                output_file = os.path.join(folder_path, "table.txt")  # Output text file
                
                # Process the XML file and generate table.txt
                xml_to_txt(xml_file, output_file)
            else:
                print(f"No XML file found in folder: {folder_path}")
        else:
            print(f"Folder does not exist: {folder_path}")

print("All operations completed.")