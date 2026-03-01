import os
import zipfile
import shutil

def extract_zip_files(src_dir, dst_dir):
    # 检查源目录是否存在
    if not os.path.exists(src_dir):
        print(f"Source directory not found: {src_dir}")
        return

    # 创建目标目录（如果不存在）
    if not os.path.exists(dst_dir):
        try:
            os.makedirs(dst_dir)
            print(f"Created directory: {dst_dir}")
        except Exception as e:
            print(f"Error creating directory {dst_dir}: {e}")
            return

    # 获取源目录下所有的 .zip 文件
    zip_files = [f for f in os.listdir(src_dir) if f.lower().endswith('.zip')]
    
    if not zip_files:
        print("No .zip files found in the source directory.")
        return

    print(f"Found {len(zip_files)} zip files. Starting extraction...")

    count = 0
    for filename in zip_files:
        file_path = os.path.join(src_dir, filename)
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # 提取所有文件到目标目录
                zip_ref.extractall(dst_dir)
                print(f"Extracted: {filename}")
                count += 1
        except zipfile.BadZipFile:
            print(f"Error: {filename} is a bad zip file.")
        except Exception as e:
            print(f"Failed to extract {filename}: {e}")

    print(f"\nProcessing complete. Successfully extracted {count} archives to '{dst_dir}'.")

if __name__ == "__main__":
    # 定义源目录和目标目录
    # 注意：根据环境信息，工作区在 d:\huxi
    src_folder = r"d:\huxi\AD+BC"
    dst_folder = r"d:\huxi\data"
    
    extract_zip_files(src_folder, dst_folder)
