import os
from lxml import etree
from collections import OrderedDict

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

def xml_to_txt(xml_file):
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
        print(f"XML解析错误 ({xml_file}): {str(e)}")
        return ""

    output_content = []

    # === 1. 文本去重（基于清洗后的内容） ===
    seen_texts = OrderedDict()
    for elem in tree.xpath('//dc:description | //ce:simple-para | //ce:caption | //ce:para', namespaces=namespaces):
        raw_text = extract_text(elem)
        cleaned_text = clean_text(raw_text)
        if cleaned_text and cleaned_text not in seen_texts:
            seen_texts[cleaned_text] = True

    output_content.extend([f"{text}\n\n" for text in seen_texts.keys()])

    return "".join(output_content)

def batch_process_xml_to_txt(input_dir, output_file):
    # 获取所有XML文件
    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    
    # 打开输出文件以追加模式写入
    with open(output_file, 'w', encoding='utf-8') as corpus_file:
        for xml_file in xml_files:
            xml_path = os.path.join(input_dir, xml_file)
            print(f"Processing file: {xml_path}")
            
            # 提取文本内容并写入到train_corpus.txt
            text_content = xml_to_txt(xml_path)
            corpus_file.write(text_content)
            corpus_file.write("\n" + "="*80 + "\n")  # 分隔符，区分不同文件的内容

    print(f"Batch processing completed! Corpus saved to: {output_file}")

if __name__ == "__main__":
    input_dir = r".\HEA_corpus\corpus\val"
    output_file = r".\HEA_corpus\corpus\val_corpus.txt"
    
    # 批量处理XML文件并将结果写入train_corpus.txt
    batch_process_xml_to_txt(input_dir, output_file)