import os

BAD_CHARS = [
    b'\xe2\x80\x8b',  # U+200B 零宽空格 (最常见的罪魁祸首)
    b'\xe2\x80\x8c',  # U+200C 零宽非连接符
    b'\xe2\x80\x8d',  # U+200D 零宽连接符
    b'\xef\xbb\xbf',  # U+FEFF BOM 头 (有时也会混入)
    b'\xc2\xa0',      # U+00A0 不间断空格 (看起来像普通空格但不是)
]

# 需要扫描的目录
SCAN_DIRS = ["core", "utils", "ui", "."]

# 只处理 .py 文件
TARGET_EXT = ".py"

def clean_file(filepath):
    with open(filepath, 'rb') as f:
        content = f.read()
    
    original = content
    for bad in BAD_CHARS:
        content = content.replace(bad, b'')
    
    if content != original:
        with open(filepath, 'wb') as f:
            f.write(content)
        return True
    return False

def main():
    cleaned = []
    skipped = []
    
    for scan_dir in SCAN_DIRS:
        if not os.path.exists(scan_dir):
            continue
        # 只扫描当前层（不递归），避免误处理系统文件
        if scan_dir == ".":
            targets = [f for f in os.listdir(".") if f.endswith(TARGET_EXT)]
        else:
            targets = [
                os.path.join(scan_dir, f)
                for f in os.listdir(scan_dir)
                if f.endswith(TARGET_EXT)
            ]
        
        for filepath in targets:
            if clean_file(filepath):
                cleaned.append(filepath)
            else:
                skipped.append(filepath)
    
    print("=" * 40)
    print("清理完成！")
    print(f"已清理 ({len(cleaned)} 个文件):")
    for f in cleaned:
        print(f"  ✓ {f}")
    if skipped:
        print(f"无污染，跳过 ({len(skipped)} 个文件):")
        for f in skipped:
            print(f"  - {f}")
    print("=" * 40)
    print("现在可以安全运行 python3 main.py 了！")

if __name__ == "__main__":
    main()
