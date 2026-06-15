import re
import os
import json
import requests
import urllib3
import argparse
from typing import List, Tuple

import sys
import io

from logger import get_logger

logger = get_logger("srt_pre_prompt")

# 重新设置 stdout/stderr 编码（仍保留，不影响日志）
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
            logger.error("在 %s 中未找到 deepseek.api_key 字段", CONFIG_FILE)
            return None
        return api_key
    except Exception as e:
        logger.error("读取配置文件失败：%s，将使用默认分组", e)
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
            logger.warning("API返回 %s，文本 %s", response.status_code, response.text[:200])
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
                            logger.warning("忽略超出范围的分组 %s", g)
                    else:
                        logger.warning("无效分组数据: %s", g)
                        continue
                if parsed:
                    return parsed
                else:
                    logger.warning("未能解析出有效分组，原始响应：%s", json_str[:200])
                    return None
            else:
                logger.warning("响应中没有有效的JSON数组")
                return None
        else:
            logger.warning("API返回内容为空")
            return None
    except Exception as e:
        logger.error("API请求异常：%s", e)
        return None

def semantic_grouping_with_original_indices(sentences_with_idx: List[Tuple[int, str]]) -> List[Tuple[int, int, str]]:
    """
    对带原始序号的句子列表进行语义分组，返回 [(start_orig, end_orig, combined_text), ...]
    """
    indices = [idx for idx, _ in sentences_with_idx]
    texts = [text for _, text in sentences_with_idx]
    total = len(texts)

    api_key = get_deepseek_api_key()
    if not api_key:
        return [(indices[i], indices[i], texts[i]) for i in range(total)]

    BATCH_SIZE = 50
    all_groups = []
    success = True

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_texts = texts[batch_start:batch_end]
        global_start = batch_start + 1
        logger.info("处理批次 %d-%d...", global_start, batch_end)
        groups = semantic_grouping_batch(batch_texts, global_start, api_key)
        if groups is None:
            success = False
            break
        all_groups.extend(groups)

    if success and all_groups:
        covered = set()
        for s, e in all_groups:
            for i in range(s, e+1):
                covered.add(i)
        if len(covered) == total:
            logger.info("语义分组成功，共 %d 组", len(all_groups))
            result = []
            for s_rel, e_rel in all_groups:
                start_orig = indices[s_rel - 1]
                end_orig = indices[e_rel - 1]
                combined = ' '.join(texts[s_rel-1:e_rel])
                result.append((start_orig, end_orig, combined))
            return result
        else:
            logger.warning("全局覆盖不完全（%d/%d），使用默认分组", len(covered), total)
    else:
        logger.warning("部分批次分组失败，使用默认分组")

    return [(indices[i], indices[i], texts[i]) for i in range(total)]

def parse_srt(file_path: str) -> List[Tuple[int, str]]:
    """解析 SRT 文件，返回(原始序号, 纯文本) 列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n\s*\n', content.strip())
    result = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 2:
            try:
                idx = int(lines[0].strip())
            except ValueError:
                continue
            text_lines = lines[2:]
            text = ' '.join(text_lines).strip()
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'\s+', ' ', text)
            if text:
                result.append((idx, text))
    return result

def main():
    parser = argparse.ArgumentParser(description='将 SRT 字幕自动语义分组并生成提示词预览文件')
    parser.add_argument('srt_file', help='输入的 SRT 文件路径')
    parser.add_argument('--output_dir', default=None, help='输出目录（默认为 tmp/字幕文件名）')
    args = parser.parse_args()

    if not os.path.exists(args.srt_file):
        logger.error("文件 %s 不存在", args.srt_file)
        return

    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_name = os.path.splitext(os.path.basename(args.srt_file))[0]
        base_name = base_name.replace(' ', '_')
        output_dir = os.path.join("tmp", base_name)
    os.makedirs(output_dir, exist_ok=True)

    sentences_with_idx = parse_srt(args.srt_file)
    if not sentences_with_idx:
        logger.warning("未提取到任何句子，请检查 SRT 文件格式")
        return
    logger.info("共提取 %d 个句子", len(sentences_with_idx))

    groups = semantic_grouping_with_original_indices(sentences_with_idx)
    logger.info("自动分组完成，共 %d 组", len(groups))

    output_path = os.path.join(output_dir, "prompts_preview.txt")
    with open(output_path, 'w', encoding='utf-8') as out:
        for start_orig, end_orig, combined in groups:
            out.write(f"图片对应第{start_orig}-{end_orig}句：\n{combined}\n\n")
    logger.info("提示词预览已写入：%s", output_path)

if __name__ == '__main__':
    main()
