import re
import os
import json
import requests
import urllib3
import argparse
from typing import List, Tuple

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CONFIG_FILE = "config/API_key.json"

def get_deepseek_api_key():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        api_key = config.get("deepseek", {}).get("api_key")
        if not api_key:
            print(f"[错误] 错误：在 {CONFIG_FILE} 中未找到 deepseek.api_key 字段")
            return None
        return api_key
    except Exception as e:
        print(f"[错误] 读取配置文件失败：{e}，将使用默认分组")
        return None

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def extract_json_from_text(text: str) -> list:
    """从文本中提取第一个JSON数组（支持被markdown包裹或嵌入其他文字）"""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            return obj
    except:
        pass

    match = re.search(r'\[\s*\{.*?\}\s*\]', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    last_brace_start = text.rfind('[')
    if last_brace_start != -1:
        bracket_count = 0
        end_idx = -1
        for i, ch in enumerate(text[last_brace_start:], start=last_brace_start):
            if ch == '[':
                bracket_count += 1
            elif ch == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break
        if end_idx != -1:
            possible_json = text[last_brace_start:end_idx]
            try:
                return json.loads(possible_json)
            except:
                pass
    return None

def semantic_grouping_batch(sentences: List[str], start_index: int, api_key: str) -> List[Tuple[int, int]]:
    """对一批句子进行语义分组，返回相对于全局的起止编号（从1开始）"""
    numbered = [f"{i+1}. {text}" for i, text in enumerate(sentences)]
    user_content = f"""根据语义连贯性，将以下句子分成若干组（每组1-5句）。只输出JSON数组，格式：[{{"start":x,"end":y}}, ...]。
句子列表：
{chr(10).join(numbered)}"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "你是分组助手。直接输出JSON数组，不要任何解释。"},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.2,
        "max_tokens": 4000
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, verify=False)
        if response.status_code != 200:
            print(f"   [警告] API返回{response.status_code}，文本: {response.text[:200]}")
            return None

        result = response.json()
        choices = result.get('choices', [])
        if not choices:
            return None

        message = choices[0].get('message', {})
        content = message.get('content', '').strip()
        reasoning = message.get('reasoning_content', '')
        finish_reason = choices[0].get('finish_reason', '')

        json_str = None
        if content:
            json_str = content
        elif reasoning:
            json_str = reasoning

        if json_str:
            groups = extract_json_from_text(json_str)
            if groups and isinstance(groups, list):
                parsed = []
                for g in groups:
                    start = g.get('start')
                    end = g.get('end')
                    if start and end and isinstance(start, int) and isinstance(end, int):
                        if 1 <= start <= end <= len(sentences):
                            parsed.append((start + start_index - 1, end + start_index - 1))
                        else:
                            print(f"   [警告] 忽略超出范围的分组: {g}")
                    else:
                        print(f"   [警告] 无效分组格式: {g}")
                if parsed:
                    covered = set()
                    for s, e in parsed:
                        for i in range(s, e+1):
                            covered.add(i)
                    batch_covered = sum(1 for i in range(start_index, start_index+len(sentences)) if i in covered)
                    if batch_covered == len(sentences):
                        return parsed
                    else:
                        print(f"   [警告] 本批次分组未完全覆盖（{batch_covered}/{len(sentences)}），放弃")
                else:
                    print("   [警告] 解析出的分组列表为空")
            else:
                print("   [警告] 未能从响应中提取有效的 JSON 数组")
        else:
            print("   [警告] 响应中既无 content 也无 reasoning_content")
    except Exception as e:
        print(f"   [警告] API调用异常: {e}")
    return None

def semantic_grouping_with_original_indices(sentences_with_idx: List[Tuple[int, str]]) -> List[Tuple[int, int, str]]:
    """
    输入：(原始序号, 文本) 列表
    输出：[(原始起始序号, 原始结束序号, 合并后的文本), ...]
    """
    # 提取纯文本列表和原始序号列表
    indices = [idx for idx, _ in sentences_with_idx]
    texts = [text for _, text in sentences_with_idx]
    total = len(texts)

    api_key = get_deepseek_api_key()
    if not api_key:
        # 默认每句一组，使用原始序号
        return [(indices[i], indices[i], texts[i]) for i in range(total)]

    BATCH_SIZE = 50
    all_groups = []  # 存储相对索引 (1-based) 的分组
    success = True

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_texts = texts[batch_start:batch_end]
        global_start = batch_start + 1
        print(f"处理批次 {global_start}-{batch_end}...")
        groups = semantic_grouping_batch(batch_texts, global_start, api_key)
        if groups is None:
            success = False
            break
        all_groups.extend(groups)

    if success and all_groups:
        # 验证覆盖
        covered = set()
        for s, e in all_groups:
            for i in range(s, e+1):
                covered.add(i)
        if len(covered) == total:
            print(f"[成功] 语义分组成功，共 {len(all_groups)} 组")
            # 将相对索引转换为原始序号
            result = []
            for s_rel, e_rel in all_groups:
                start_orig = indices[s_rel - 1]
                end_orig = indices[e_rel - 1]
                combined = ' '.join(texts[s_rel-1:e_rel])
                result.append((start_orig, end_orig, combined))
            return result
        else:
            print(f"[警告] 全局覆盖不完全（{len(covered)}/{total}），使用默认分组")
    else:
        print("[警告] 部分批次分组失败，使用默认分组")

    # 默认分组：每1句一组，使用原始序号
    return [(indices[i], indices[i], texts[i]) for i in range(total)]

def parse_srt(file_path: str) -> List[Tuple[int, str]]:
    """解析 SRT 文件，返回 (原始序号, 纯文本) 列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n\s*\n', content.strip())
    result = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 2:
            # 第一行是序号
            try:
                idx = int(lines[0].strip())
            except ValueError:
                continue
            # 第二行是时间轴，忽略
            text_lines = lines[2:]
            text = ' '.join(text_lines).strip()
            text = re.sub(r'<[^>]+>', '', text)   # 移除HTML标签
            text = re.sub(r'\s+', ' ', text)      # 合并空白
            if text:
                result.append((idx, text))
    return result

def main():
    parser = argparse.ArgumentParser(description='将 SRT 字幕自动语义分组并生成提示词预览文件')
    parser.add_argument('srt_file', help='输入的 SRT 文件路径')
    parser.add_argument('--output_dir', default=None, help='输出目录（默认为 tmp/字幕文件名）')
    args = parser.parse_args()

    if not os.path.exists(args.srt_file):
        print(f"错误：文件 {args.srt_file} 不存在")
        return

    # 确定输出目录（替换空格为下划线）
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_name = os.path.splitext(os.path.basename(args.srt_file))[0]
        base_name = base_name.replace(' ', '_')
        output_dir = os.path.join("tmp", base_name)
    os.makedirs(output_dir, exist_ok=True)

    # 解析 SRT，得到带原始序号的句子列表
    sentences_with_idx = parse_srt(args.srt_file)
    if not sentences_with_idx:
        print("警告：未提取到任何句子，请检查 SRT 文件格式。")
        return
    print(f"共提取 {len(sentences_with_idx)} 个句子")

    # 语义分组（保留原始序号）
    groups = semantic_grouping_with_original_indices(sentences_with_idx)
    print(f"自动分组完成，共 {len(groups)} 组")

    # 写入 prompts_preview.txt
    output_path = os.path.join(output_dir, "prompts_preview.txt")
    with open(output_path, 'w', encoding='utf-8') as out:
        for start_orig, end_orig, combined in groups:
            out.write(f"图片对应第 {start_orig}-{end_orig} 句：\n{combined}\n\n")
    print(f"提示词预览已写入：{output_path}")

if __name__ == '__main__':
    main()