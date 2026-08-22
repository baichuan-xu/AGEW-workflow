import os
from lxml import etree
from collections import OrderedDict

# Define helper functions (unchanged from the original code)

def clean_text(text):
    """文本清洗：合并空格、去除首尾空、标准化符号"""
    return text.replace("  ", " ").replace("–", "-").strip()

def extract_text(element):
    """递归提取文本（精确处理嵌套标签）"""
    text = element.text.strip() + " " if element.text else ""
    for child in element:
        text += extract_text(child)
        if child.tail and child.tail.strip():
            text += child.tail.strip() + " "
    return clean_text(text)

def extract_table(table_element):
    """高精度提取表格（兼容复杂命名空间和嵌套结构）""" 
    namespaces = {
        'cals': 'http://www.elsevier.com/xml/common/cals/dtd',
        'ce': 'http://www.elsevier.com/xml/common/dtd'
    }
    table_data = []
    
    # 提取表头
    thead = table_element.find(".//cals:thead", namespaces)
    if thead:
        headers = []
        for entry in thead.xpath(".//cals:entry", namespaces=namespaces):
            header_text = extract_text(entry)
            headers.append(header_text)
        if headers:
            table_data.append(headers)
    
    # 提取表体
    tbody = table_element.find(".//cals:tbody", namespaces)
    if tbody:
        for row in tbody.xpath(".//cals:row", namespaces=namespaces):
            row_data = []
            for entry in row.xpath(".//cals:entry", namespaces=namespaces):
                cell_text = extract_text(entry)
                row_data.append(cell_text)
            if row_data:
                table_data.append(row_data)
    return table_data

def extract_figures(tree):
    """提取图片编号与描述（处理嵌套和缺失标签）"""
    namespaces = {'ce': 'http://www.elsevier.com/xml/common/dtd'}
    figures = []
     
    for fig in tree.xpath('//ce:figure', namespaces=namespaces):
        fig_id = fig.findtext('ce:label', default="Unknown_Fig", namespaces=namespaces)
        caption_elem = fig.find('.//ce:caption//ce:simple-para', namespaces=namespaces)
        caption = extract_text(caption_elem) if caption_elem is not None else "[Caption Missing]"
        figures.append({"id": fig_id, "caption": caption})
    
    return figures

def xml_to_txt(xml_file, output_file):
    # 注册所有命名空间
    namespaces = {
        'ce': 'http://www.elsevier.com/xml/common/dtd',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'cals': 'http://www.elsevier.com/xml/common/cals/dtd',
        'ja': 'http://www.elsevier.com/xml/ja/dtd' 
    }
    
    try:
        tree = etree.parse(xml_file)
    except Exception as e:
        print(f"XML解析错误: {str(e)}")
        return

    output_content = []
    
    # === 1. 文本去重（基于清洗后的内容） ===
    seen_texts = OrderedDict()
    for elem in tree.xpath('//dc:description | //ce:simple-para | //ce:caption | //ce:para', namespaces=namespaces):
        raw_text = extract_text(elem)
        cleaned_text = clean_text(raw_text)
        if cleaned_text and cleaned_text not in seen_texts:
            seen_texts[cleaned_text] = True
    
    output_content.append("="*40 + " TEXT CONTENT (去重后) " + "="*40 + "\n")
    output_content.extend([f"{text}\n\n" for text in seen_texts.keys()])
    
    # === 2. 表格提取（修复缺失问题） ===
    tables = []
    for table in tree.xpath('//ce:table', namespaces=namespaces):
        table_data = extract_table(table)
        if table_data:
            tables.append(table_data)
    
    if tables:
        output_content.append("\n" + "="*40 + " TABLES " + "="*40 + "\n")
        for idx, table in enumerate(tables, 1):
            output_content.append(f"Table {idx}:\n")
            for row in table:
                output_content.append(" | ".join(row) + "\n")
            output_content.append("\n")
    
    # === 3. 图片信息（修复编号和描述缺失） ===
    figures = extract_figures(tree)
    if figures:
        output_content.append("\n" + "="*40 + " FIGURES " + "="*40 + "\n")
        for fig in figures:
            output_content.append(f"{fig['id']}:\n{fig['caption']}\n\n")

    # 写入文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output_content)
        print(f"转换成功！输出文件: {output_file}")
    except Exception as e:
        print(f"文件写入错误: {str(e)}")

# Main logic to process all subfolders
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    
    # Loop through subfolders L001 to L104
    for i in range(1, 3317):  # From 1 to 104 inclusive
        folder_name = f"L{i:04d}"  # Format folder name with leading zeros (e.g., L001, L002)
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Find the XML file in the folder
            xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
            
            if xml_files:
                xml_file = os.path.join(folder_path, xml_files[0])  # Assume only one XML file per folder
                output_file = os.path.join(folder_path, "text.txt")  # Output text file
                
                # Process the XML file and generate text.txt
                xml_to_txt(xml_file, output_file)
            else:
                print(f"No XML file found in folder: {folder_path}")
        else:
            print(f"Folder does not exist: {folder_path}")

print("All operations completed.")