import requests
import time
import json
from PIL import Image
from io import BytesIO
import urllib3
import os
import re
import argparse
from datetime import datetime
import sys
import io

from logger import get_logger

logger = get_logger("generate_image")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CONFIG_FILE = "config/API_key.json"
MODEL_CONFIG_FILE = "config/PicModel.json"

def get_modelscope_api_key():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        api_key = config.get("modelscope", {}).get("api_key")
        if not api_key:
            logger.error("在 %s 中未找到 modelscope.api_key 字段", CONFIG_FILE)
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

def load_model_list():
    try:
        with open(MODEL_CONFIG_FILE, "r", encoding="utf-8") as f:
            models = json.load(f)
        if not isinstance(models, list) or len(models) == 0:
            logger.error("%s 内容不是非空数组", MODEL_CONFIG_FILE)
            exit(1)
        model_list = [(item.get("model_id"), item.get("display_name")) for item in models if item.get("model_id")]
        if not model_list:
            logger.error("%s 中未找到有效的 model_id", MODEL_CONFIG_FILE)
            exit(1)
        return model_list
    except FileNotFoundError:
        logger.error("找不到配置文件 %s，请确保 config 目录和文件存在", MODEL_CONFIG_FILE)
        exit(1)
    except json.JSONDecodeError:
        logger.error("%s 格式不是有效的 JSON", MODEL_CONFIG_FILE)
        exit(1)
    except Exception as e:
        logger.error("读取模型列表时出错：%s", e)
        exit(1)

API_KEY = get_modelscope_api_key()
logger.info("成功从 %s 读取到 ModelScope API Key", CONFIG_FILE)

MODEL_CANDIDATES = load_model_list()
logger.info("已加载 %d 个备选模型（按顺序尝试）：", len(MODEL_CANDIDATES))
for model_id, display_name in MODEL_CANDIDATES:
    logger.info("  - %s (%s)", display_name, model_id)

parser = argparse.ArgumentParser(description="根据 llm_prompts.txt 批量生成图片")
parser.add_argument("--srt_file", required=True, help="输入的 SRT 文件路径（用于确定工作目录）")
parser.add_argument("--work_dir", default=None, help="工作目录（默认为 tmp/字幕文件名）")
parser.add_argument("--size", type=str, default="1920x1080", help="图片尺寸，如 1024x1024，默认 1920x1080")
args = parser.parse_args()

if args.work_dir:
    work_dir = args.work_dir
else:
    base_name = os.path.splitext(os.path.basename(args.srt_file))[0]
    base_name = base_name.replace(' ', '_')
    work_dir = os.path.join("tmp", base_name)

llm_prompts_file = os.path.join(work_dir, "llm_prompts.txt")
if not os.path.exists(llm_prompts_file):
    logger.error("找不到 %s，请先运行 llm_prompt_generator.py 生成该文件。", llm_prompts_file)
    exit(1)

output_dir = os.path.join("output", base_name)
os.makedirs(output_dir, exist_ok=True)
logger.info("图片将保存至：%s", output_dir)

def parse_llm_prompts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'图片对应第(\d+(?:-\d+)?)句.*?正向提示词[：:](.*?)(?=负向提示词|$)'
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        pattern = r'\[(\d+(?:-\d+)?)\]\s*正向提示词[：:](.*?)(?=\[|$)'
        matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        pattern = r'正向提示词[：:](.*?)(?=负向提示词|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            for i, prompt in enumerate(matches, 1):
                pass

    results = []
    for m in matches:
        sentence_range = m[0].strip()
        positive_prompt = re.sub(r'\s+', ' ', m[1].strip())
        if positive_prompt:
            results.append((sentence_range, positive_prompt))
    return results

parse_llm_prompts(llm_prompts_file)

groups = parse_llm_prompts(llm_prompts_file)
if not groups:
    logger.error("未能从 %s 中解析出任何提示词，请检查文件格式", llm_prompts_file)
    exit(1)

logger.info("共读取 %d 组提示词，开始批量生成", len(groups))

def parse_and_print_rate_limits(response):
    """解析并输出速率限制信息"""
    remaining = None
    limit = None
    reset = None
    user_remaining = None
    user_limit = None

    remaining_str = response.headers.get('x-ratelimit-remaining')
    limit_str = response.headers.get('x-ratelimit-limit')
    reset_str = response.headers.get('x-ratelimit-reset')
    user_remaining_str = response.headers.get('x-ratelimit-user-remaining')
    user_limit_str = response.headers.get('x-ratelimit-user-limit')

    if remaining_str:
        try:
            remaining = int(remaining_str)
        except ValueError:
            pass
    if limit_str:
        try:
            limit = int(limit_str)
        except ValueError:
            pass
    if reset_str:
        try:
            reset = int(reset_str)
        except ValueError:
            pass
    if user_remaining_str:
        try:
            user_remaining = int(user_remaining_str)
        except ValueError:
            pass
    if user_limit_str:
        try:
            user_limit = int(user_limit_str)
        except ValueError:
            pass

    if limit is not None:
        logger.debug("速率限制：剩余 %s/%s", remaining if remaining is not None else "?", limit)
    if user_limit is not None:
        logger.debug("用户额度：%s/%s (已用 %s)",
                     user_remaining if user_remaining is not None else "?",
                     user_limit,
                     user_limit - user_remaining if user_remaining is not None else "?")
    if reset is not None:
        reset_time = datetime.fromtimestamp(reset).strftime('%H:%M:%S')
        logger.debug("额度重置时间：%s", reset_time)

    return user_remaining, reset

TASK_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"

def generate_with_model(model_id, display_name, positive_prompt, size, current_user_remaining):
    url = f"https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable"
    }
    width, height = size.split('x')
    payload = {
        "model": model_id,
        "input": {
            "prompt": positive_prompt,
            "negative_prompt": "",
            "size": size
        },
        "parameters": {
            "steps": 20,
            "n": 1,
            "size": size
        }
    }
    response = None
    logger.info("向模型 %s 提交任务...", display_name)

    try:
        response = requests.post(TASK_URL, json=payload, headers=headers, verify=False)
        response.raise_for_status()
        result = response.json()
        task_id = result.get('output', {}).get('task_id')
        if not task_id:
            error_msg = result.get('output', {}).get('task_status', '未知错误')
            logger.error("提交任务失败，响应: %s", result)
            return False, None, current_user_remaining, f"API 返回异常: {error_msg}"
        logger.info("任务已提交，任务ID: %s", task_id)

    except requests.exceptions.RequestException as e:
        logger.error("提交任务失败: %s", e)
        if response is not None:
            user_rem, _ = parse_and_print_rate_limits(response)
            if user_rem is not None:
                current_user_remaining = user_rem
            if response.status_code == 429:
                logger.warning("触发限流，可能是模型额度已用完。")
                return False, None, current_user_remaining, "HTTP 429 限流"
        return False, None, current_user_remaining, str(e)

    logger.info("图片正在生成中...（预计 20-60 秒）")
    max_retries = 45
    retry_interval = 5
    start_time = time.time()
    task_status = "PENDING"
    result = None

    for i in range(max_retries):
        elapsed = time.time() - start_time
        try:
            poll_response = requests.get(f"{TASK_URL}/{task_id}", headers=headers, verify=False)
            poll_response.raise_for_status()
            result = poll_response.json()
            task_status = result.get('output', {}).get('task_status')
            if task_status == "SUCCEEDED":
                logger.info("图片生成成功！")
                break
            elif task_status in ["FAILED", "CANCELED"]:
                logger.error("任务失败: %s", task_status)
                return False, None, current_user_remaining, result.get('output', {}).get('message', '任务失败')
            else:
                logger.info("生成中...（已等待 %d 秒）", int(elapsed))
                time.sleep(retry_interval)
        except requests.exceptions.RequestException as e:
            logger.warning("查询状态出错: %s", e)
            time.sleep(retry_interval)

    if task_status != "SUCCEEDED":
        logger.error("生成超时")
        return False, None, current_user_remaining, "生成超时"

    output_images = result.get('output', {}).get('results', [])
    if not output_images:
        logger.error("未找到生成的图片")
        return False, None, current_user_remaining, "未找到生成的图片"

    image_url = output_images[0].get('url')
    if not image_url:
        logger.error("未找到图片 URL")
        return False, None, current_user_remaining, "未找到图片 URL"

    logger.info("图片地址: %s", image_url)

    try:
        img_response = requests.get(image_url, verify=False)
        img_response.raise_for_status()
        img = Image.open(BytesIO(img_response.content))
        return True, img, current_user_remaining, None
    except Exception as e:
        logger.error("下载图片失败: %s", e)
        return False, None, current_user_remaining, str(e)

success_count = 0
fail_count = 0

for idx, (sentence_range, positive_prompt) in enumerate(groups, 1):
    logger.info("=" * 60)
    logger.info("正在处理第 %d/%d 组：句子 %s", idx, len(groups), sentence_range)
    logger.info("=" * 60)

    global_user_remaining = None
    success = False
    final_img = None
    last_error = None

    for model_idx, (model_id, display_name) in enumerate(MODEL_CANDIDATES, 1):
        logger.info("尝试模型 %d/%d: %s", model_idx, len(MODEL_CANDIDATES), display_name)
        success, img, global_user_remaining, error = generate_with_model(
            model_id, display_name, positive_prompt, args.size, global_user_remaining
        )
        if success:
            final_img = img
            break
        else:
            logger.warning("模型 %s 失败: %s", display_name, error)
            last_error = error
            if global_user_remaining is not None and global_user_remaining <= 0:
                logger.warning("用户总额度已用完，停止后续模型尝试")
                break
            time.sleep(2)

    if success and final_img:
        img_filename = f"img_{idx:03d}_{sentence_range}.png"
        img_path = os.path.join(output_dir, img_filename)
        final_img.save(img_path)
        logger.info("图片已保存: %s", img_path)
        success_count += 1
    else:
        logger.error("无法为句子 %s 生成图片，最后错误: %s", sentence_range, last_error)
        fail_count += 1

    time.sleep(2)

logger.info("=" * 60)
logger.info("批量生成完成！成功: %d, 失败: %d", success_count, fail_count)
logger.info("图片保存在 %s", output_dir)
