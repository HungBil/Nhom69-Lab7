import re
import os
from pathlib import Path
import html

def clean_html_to_md(text):
    """
    Script dọn dẹp rác HTML và Footer quảng cáo cho dữ liệu bài Lab.
    Nhóm 69.
    """
    # 1. Bỏ breadcrumb và link ở đầu bài (Xoá mọi thứ trước thẻ <h1>)
    lines = text.split('\n')
    start = 0
    for i, line in enumerate(lines):
        if '<h1>' in line:
            start = i
            break
    text = '\n'.join(lines[start:])

    # 2. Xoá cục Footer quảng cáo và địa chỉ Bệnh viện Tâm Anh ở cuối
    address_patterns = [
        r'<h4[^>]*>.*?HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH.*?tamanhhospital\.vn[\n\s]*',
        r'HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH[\s\S]*?tamanhhospital\.vn[\n\s]*'
    ]
    for p in address_patterns:
        text = re.sub(p, '', text, flags=re.IGNORECASE)

    # 3. Translate chuẩn syntax từ HTML sang Markdown
    text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1', text, flags=re.DOTALL)
    text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1', text, flags=re.DOTALL)
    text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1', text, flags=re.DOTALL)
    text = re.sub(r'<h4[^>]*>(.*?)</h4>', r'#### \1', text, flags=re.DOTALL)

    text = re.sub(r'<strong>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<b>(.*?)</b>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<em>(.*?)</em>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<i>(.*?)</i>', r'*\1*', text, flags=re.DOTALL)

    # 4. Strip TOÀN BỘ thẻ HTML dư thừa (br, div, span...)
    text = re.sub(r'<[^>]+>', '', text)

    # 5. Decode các ký tự bị lỗi (Vd: &#8211; thành dấu phẩy)
    text = html.unescape(text)

    # 6. Dọn dẹp khoảng trắng, xuống dòng vô nghĩa
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    
    return text.strip()

def main():
    data_dir = Path("data")
    files = [f for f in data_dir.glob('*.md') if f.name != '.gitkeep']
    print(f"Bắt đầu dọn dẹp {len(files)} file Y tế...")
    
    for f in sorted(files):
        original = f.read_text(encoding='utf-8')
        cleaned = clean_html_to_md(original)
        f.write_text(cleaned, encoding='utf-8')
        
        orig_lines = original.count('\n') + 1
        new_lines = cleaned.count('\n') + 1
        print(f"  ✓ {f.name}: Đã ép từ {orig_lines} dòng -> xuống còn {new_lines} dòng cốt lõi.")
        
    print("Dọn dẹp hoàn tất! Dữ liệu đã sẵn sàng để cấy Embeddings.")

if __name__ == "__main__":
    main()
