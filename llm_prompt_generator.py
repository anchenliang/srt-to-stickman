import os
import re
import requests
import json
import time
import urllib3
import argparse
from typing import List, Tuple

import sys
import io

from logger import get_logger

logger = get_logger("llm_prompt_generator")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------- 配置 ----------
CONFIG_FILE = "config/API_key.json"

def get_deepseek_api_key():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        api_key = config.get("deepseek", {}).get("api_key")
        if not api_key:
            logger.error("在 %s 中未找到 deepseek.api_key 字段", CONFIG_FILE)
            exit(1)
        return api_key
    except FileNotFoundError:
        logger.error("找不到配置文件 %s，请确保 config 目录和文件存在", CONFIG_FILE)
        exit(1)
    except json.JSONDecodeError:
        logger.error("%s 格式不是有效的 JSON", CONFIG_FILE)
        exit(1)
    except Exception as e:
        logger.error("读取配置文件时出错：%s", e)
        exit(1)

DEEPSEEK_API_KEY = get_deepseek_api_key()
logger.info("成功从 %s 读取到 DeepSeek API Key", CONFIG_FILE)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

NEGATIVE_PROMPT = "复杂细节，渐变色，阴影，纹理，文字，字幕，logo，多余人物，杂乱背景，3D 渲染，写实风格，水彩，油画，模糊，噪点"

SYSTEM_PROMPT = """你是一个专业的文生图提示词生成助手。用户会提供一段对话或叙述文本（可能是一句话或多句话），你需要根据这段文本，生成一段用于【火柴人插画】的**正向提示词**。

**基本视觉风格**（不可改变的部分）：
- 整体风格：极简黑白线条火柴人插画，纯白色背景，干净利落的黑色轮廓线，其余为线条勾勒（仅可能有小面积黑色填充如领带或衣物），2D 平面动画风格，极简主义设计，居中构图，画面整洁无任何杂物，低对比度，柔和均匀的平光照明
- 画面构图与人物大小：图片总尺寸为横屏16:9（1920x1080）。要求火柴人主体占据画面中央区域，高度约为画面高度的1/2到2/3，宽度比例适中，全身或半身居中。所有图片中，火柴人主体大小和位置应保持高度一致。不要特写，不要远景。人物距离相机视角固定，避免人物忽大忽小

**你需要根据语义动态生成的内容**（务必灵活，不要照抄固定短语）：
1. **人物数量**：根据文本判断是1个还是2个火柴人。如果出现"两个人"、"双方"、"互相"、"我们/你们/他们"同时出现等，一般为2人。
2. **人物穿着**：
   - 商务/正式场景（如面试、会议、汇报）：左侧（若有）穿白衬衫?领带，右侧可穿简约衬衫或无领带。
   - 日常/休闲场景：可穿普通T恤或简约上衣，不强加领带。
   - 如果文本没有明确场景，默认使用"白衬衫"但不必每次加领带，或只用"简约上衣"。
3. **动作**：
   - 根据动词生成具体姿势（如"右手抬起做交谈手势"、"双手交叉抱胸"、"指向某处"、"站立"、"坐下"、"握手"、"举杯"等）。
   - 若为两人，可分别描述左侧和右侧动作（如左侧在说话，右侧在倾听）。
4. **道具与环境**：
   - 常见道具：椅子、桌子、电脑、文件、咖啡杯、手机、白板等。
   - 根据文本中出现的名词合理添加。如果提到"座位"，则"坐在简约线条椅子上"；如果提到"办公"，可增加"面前有一张简约桌子"。
   - 环境描述：根据文本氛围给出场景名，如"商务对话场景"、"面试场景"、"课堂场景"、"休闲聊天场景"等。

**注意事项**：
- 必须围绕"火柴人极简线条"风格输出，不能偏离此风格描述。
- 不得出现写实渲染、3D、水彩、油画等风格描述。
- 仅输出提示词本身，不要序号和前缀，不要多余解释。
- 英语（推荐）或中文皆可。"""

def call_deepseek(prompt_text: str, retry: bool = True, max_tokens: int = 2000):
    """调用 DeepSeek API 生成提示词"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    response = None
    try:
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, verify=False)
        response.raise_for_status()
        result = response.json()
        finish_reason = result['choices'][0].get('finish_reason', '')
        content = result['choices'][0]['message']['content'].strip()

        if finish_reason == 'length' and retry:
            logger.warning("提示词被截断（max_tokens=%d），使用 max_tokens=4000 重试...", max_tokens)
            return call_deepseek(prompt_text, retry=False, max_tokens=4000)

        if not content and retry:
            logger.warning("提示词为空，使用 max_tokens=4000 重试...")
            return call_deepseek(prompt_text, retry=False, max_tokens=4000)

        if not content:
            logger.error("提示词为空且重试后仍为空")
            return None

        return content
    except Exception as e:
        logger.error("调用 DeepSeek API 失败: %s", e)
        if response is not None and response.status_code == 401:
            logger.error("提示：API Key 无效或已过期，请检查 config/API_key.json 中的 deepseek.api_key")
        elif response is not None:
            logger.error("状态码: %s", response.status_code)
            logger.error("响应内容: %s", response.text)
        return None

def parse_prompts_preview(file_path: str) -> List[Tuple[str, str]]:
    """解析 prompts_preview.txt，返回[(header, original_text), ...]"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n\s*\n', content.strip())
    results = []
    for block in blocks:
        if not block:
            continue
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        header = lines[0]
        original_text = ' '.join(lines[1:]).strip()
        results.append((header, original_text))
    return results

def main():
    parser = argparse.ArgumentParser(description='使用 DeepSeek 生成最终的火柴人风格提示词')
    parser.add_argument('--srt_file', required=True, help='输入的 SRT 文件路径（用于确定工作目录）')
    parser.add_argument('--work_dir', default=None, help='工作目录（默认为 tmp/字幕文件名）')
    args = parser.parse_args()

    if args.work_dir:
        work_dir = args.work_dir
    else:
        base_name = os.path.splitext(os.path.basename(args.srt_file))[0]
        base_name = base_name.replace(' ', '_')
        work_dir = os.path.join("tmp", base_name)

    input_file = os.path.join(work_dir, "prompts_preview.txt")
    output_file = os.path.join(work_dir, "llm_prompts.txt")

    if not os.path.exists(input_file):
        logger.error("找不到 %s，请先运行 srt_to_prompts.py 生成该文件。", input_file)
        return

    groups = parse_prompts_preview(input_file)
    logger.info("共发现 %d 个分组，开始调用 DeepSeek API 生成提示词（模型：deepseek-v4-flash）...", len(groups))

    os.makedirs(work_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out:
        for idx, (header, original_text) in enumerate(groups, 1):
            logger.info("正在处理第 %d/%d 组：%s", idx, len(groups), header)
            user_message = f"请根据以下内容生成对应的火柴人风格正向提示词：\n{original_text}"
            positive_prompt = call_deepseek(user_message)
            if positive_prompt is None:
                positive_prompt = "【生成失败，请检查API Key或网络】"
            out.write(f"{header}\n")
            out.write(f"原始文本：{original_text}\n")
            out.write(f"正向提示词：{positive_prompt}\n")
            out.write(f"负向提示词：{NEGATIVE_PROMPT}\n")
            out.write("-" * 80 + "\n\n")
            time.sleep(1)

    logger.info("所有提示词已生成，保存至 %s", output_file)

if __name__ == '__main__':
    main()
