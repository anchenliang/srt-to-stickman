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
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 禁用 SSL 警告（临时绕过证书验证）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 1. 读取统一配置文件 ---
CONFIG_FILE = "config/API_key.json"
MODEL_CONFIG_FILE = "config/PicModel.json"

def get_modelscope_api_key():
    """从 config/API_key.json 中读取 ModelScope API Key"""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        api_key = config.get("modelscope", {}).get("api_key")
        if not api_key:
            print(f"[错误] 错误：在 {CONFIG_FILE} 中未找到 modelscope.api_key 字段")
            exit(1)
        return api_key
    except FileNotFoundError:
        print(f"[错误] 错误：找不到配置文件 {CONFIG_FILE}，请确保 config 目录和文件存在")
        exit(1)
    except json.JSONDecodeError:
        print(f"[错误] 错误：{CONFIG_FILE} 格式不是有效的 JSON")
        exit(1)
    except Exception as e:
        print(f"[错误] 读取配置文件时出错：{e}")
        exit(1)

def load_model_list():
    """从 config/PicModel.json 加载模型列表（按顺序）"""
    try:
        with open(MODEL_CONFIG_FILE, "r", encoding="utf-8") as f:
            models = json.load(f)
        if not isinstance(models, list) or len(models) == 0:
            print(f"[错误] 错误：{MODEL_CONFIG_FILE} 内容不是非空数组")
            exit(1)
        # 提取 model_id 列表，同时保留 display_name 用于日志
        model_list = [(item.get("model_id"), item.get("display_name")) for item in models if item.get("model_id")]
        if not model_list:
            print(f"[错误] 错误：{MODEL_CONFIG_FILE} 中未找到有效的 model_id")
            exit(1)
        return model_list
    except FileNotFoundError:
        print(f"[错误] 错误：找不到配置文件 {MODEL_CONFIG_FILE}，请确保 config 目录和文件存在")
        exit(1)
    except json.JSONDecodeError:
        print(f"[错误] 错误：{MODEL_CONFIG_FILE} 格式不是有效的 JSON")
        exit(1)
    except Exception as e:
        print(f"[错误] 读取模型列表时出错：{e}")
        exit(1)

# 获取 API Key 和模型列表
API_KEY = get_modelscope_api_key()
print(f"[成功] 成功从 {CONFIG_FILE} 读取到 ModelScope API Key")

MODEL_CANDIDATES = load_model_list()
print(f"📋 已加载 {len(MODEL_CANDIDATES)} 个备选模型（按顺序尝试）：")
for model_id, display_name in MODEL_CANDIDATES:
    print(f"   - {display_name} ({model_id})")

# --- 2. 解析命令行参数 ---
parser = argparse.ArgumentParser(description="根据 llm_prompts.txt 批量生成图片")
parser.add_argument("--srt_file", required=True, help="输入的 SRT 文件路径（用于确定工作目录）")
parser.add_argument("--work_dir", default=None, help="工作目录（默认为 tmp/字幕文件名）")
parser.add_argument("--size", type=str, default="1920x1080", help="图片尺寸，如 1024x1024，默认 1920x1080")
args = parser.parse_args()

# 确定工作目录（与 srt_to_prompts.py 和 llm_prompt_generator.py 保持一致）
if args.work_dir:
    work_dir = args.work_dir
else:
    base_name = os.path.splitext(os.path.basename(args.srt_file))[0]
    base_name = base_name.replace(' ', '_')  # 统一替换空格为下划线
    work_dir = os.path.join("tmp", base_name)

llm_prompts_file = os.path.join(work_dir, "llm_prompts.txt")
if not os.path.exists(llm_prompts_file):
    print(f"[错误] 找不到 {llm_prompts_file}，请先运行 llm_prompt_generator.py 生成该文件。")
    exit(1)

# 输出目录：output/<字幕文件名>/
output_dir = os.path.join("output", base_name)
os.makedirs(output_dir, exist_ok=True)
print(f"[目录] 图片将保存至：{output_dir}")

# --- 3. 解析 llm_prompts.txt，提取每组信息 ---
def parse_llm_prompts(file_path):
    """
    解析 llm_prompts.txt，返回列表，每个元素为 (sentence_range, positive_prompt)
    sentence_range 形如 "1-4"
    positive_prompt 为提取的正向提示词文本（可能跨行）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按 "----------------" 分隔每个组
    blocks = re.split(r'-{80,}', content)
    groups = []
    
    for block in blocks:
        if not block.strip():
            continue
        # 提取图片对应第 X-Y 句
        range_match = re.search(r'图片对应第\s*(\d+)-(\d+)\s*句', block)
        if not range_match:
            continue
        start, end = int(range_match.group(1)), int(range_match.group(2))
        sentence_range = f"{start}-{end}"
        
        # 提取正向提示词：从 "正向提示词：" 到下一个 "负向提示词：" 或文件末尾
        positive_match = re.search(r'正向提示词[：:]\s*(.*?)(?=\n负向提示词[：:]|\Z)', block, re.DOTALL)
        if not positive_match:
            print(f"[警告] 跳过 {sentence_range}：未找到正向提示词")
            continue
        positive_prompt = positive_match.group(1).strip()
        # 移除可能的尾部换行和多余空白
        positive_prompt = re.sub(r'\n+', ' ', positive_prompt).strip()
        if not positive_prompt:
            print(f"[警告] 跳过 {sentence_range}：正向提示词为空")
            continue
        
        groups.append((sentence_range, positive_prompt))
    
    return groups

groups = parse_llm_prompts(llm_prompts_file)
if not groups:
    print("[错误] 未从 llm_prompts.txt 中提取到任何有效分组")
    exit(1)
print(f"📄 共发现 {len(groups)} 个图片分组，准备生成...")

# --- 4. API 配置 ---
BASE_URL = "https://api-inference.modelscope.cn"
GENERATE_URL = f"{BASE_URL}/v1/images/generations"
TASK_URL = f"{BASE_URL}/v1/tasks"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "X-ModelScope-Async-Mode": "true"  # 开启异步模式
}

POLL_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "X-ModelScope-Task-Type": "image_generation"
}

def parse_and_print_rate_limits(response):
    """从响应头中解析并打印额度信息，返回 (用户剩余额度, 模型剩余额度)"""
    headers = response.headers
    user_limit = headers.get("modelscope-ratelimit-requests-limit")
    user_remaining = headers.get("modelscope-ratelimit-requests-remaining")
    model_limit = headers.get("modelscope-ratelimit-model-requests-limit")
    model_remaining = headers.get("modelscope-ratelimit-model-requests-remaining")

    print("\n[额度] ========== 额度信息 ==========")
    if user_limit and user_remaining:
        print(f"   用户总额度: {user_limit} 次")
        print(f"   今日剩余:   {user_remaining} 次")
    else:
        print("   用户额度信息: 未获取到（可能未绑定阿里云账号或未实名）")

    if model_limit and model_remaining:
        print(f"   当前模型限额: {model_limit} 次")
        print(f"   模型剩余:     {model_remaining} 次")
    else:
        print("   模型额度信息: 未获取到（可能该模型不返回额度）")
    print("================================\n")

    user_rem = int(user_remaining) if user_remaining is not None else None
    model_rem = int(model_remaining) if model_remaining is not None else None
    return user_rem, model_rem

def generate_with_model(model_id, display_name, prompt, size, current_user_remaining):
    """
    使用指定模型生成图片
    返回 (是否成功, 图片路径, 更新后的用户剩余额度, 错误信息)
    """
    payload = {
        "model": model_id,
        "prompt": prompt,
        "size": size
    }

    print(f"\n[开始] 尝试使用模型: {display_name} ({model_id})")
    print(f"[信息] 提示词: {prompt[:150]}...")

    response = None
    try:
        response = requests.post(GENERATE_URL, json=payload, headers=HEADERS, verify=False)
        response.raise_for_status()

        user_remaining, model_remaining = parse_and_print_rate_limits(response)
        if user_remaining is not None:
            current_user_remaining = user_remaining

        if model_remaining is not None and model_remaining <= 0:
            print(f"[警告] 模型 {display_name} 今日额度已用完，将切换到下一个模型。")
            return False, None, current_user_remaining, "模型额度已用完"

        task_data = response.json()
        task_id = task_data.get('task_id') or task_data.get('id')
        if not task_id:
            print("[错误] API 返回异常，未包含任务ID。")
            return False, None, current_user_remaining, "API 返回异常"
        print(f"[信息] 任务已提交，任务ID: {task_id}")

    except requests.exceptions.RequestException as e:
        print(f"[错误] 提交任务失败: {e}")
        if response is not None:
            user_rem, _ = parse_and_print_rate_limits(response)
            if user_rem is not None:
                current_user_remaining = user_rem
            if response.status_code == 429:
                print("[警告] 触发限流，可能是模型额度已用完。")
                return False, None, current_user_remaining, "HTTP 429 限流"
        return False, None, current_user_remaining, str(e)

    # 轮询任务状态
    print("[等待] 图片正在生成中... (预计 20-60 秒)")
    max_retries = 45          # 最多尝试 45 次
    retry_interval = 5        # 每 5 秒查询一次
    start_time = time.time()   # 记录开始时间
    task_status = "PENDING"
    result = None

    for i in range(max_retries):
        elapsed = time.time() - start_time
        try:
            poll_response = requests.get(f"{TASK_URL}/{task_id}", headers=POLL_HEADERS, verify=False)
            poll_response.raise_for_status()
            result = poll_response.json()
            task_status = result.get('task_status')
            if task_status == "SUCCEED":
                print("[完成] 图片生成成功！")
                break
            elif task_status in ["FAILED", "CANCELED"]:
                print(f"[错误] 任务失败: {task_status}")
                return False, None, current_user_remaining, result.get('message', '任务失败')
            else:
                # 显示已等待秒数
                print(f"[处理] 生成中... (已等待 {elapsed:.0f} 秒)")
                time.sleep(retry_interval)
        except requests.exceptions.RequestException as e:
            print(f"[警告] 查询状态出错: {e}")
            time.sleep(retry_interval)

    if task_status != "SUCCEED":
        print("[错误] 生成超时")
        return False, None, current_user_remaining, "生成超时"

    output_images = result.get('output_images')
    if not output_images:
        return False, None, current_user_remaining, "未找到生成的图片"

    image_url = output_images[0]
    print(f"[图片] 图片地址: {image_url}")

    try:
        img_response = requests.get(image_url, verify=False)
        img_response.raise_for_status()
        img = Image.open(BytesIO(img_response.content))
        return True, img, current_user_remaining, None
    except Exception as e:
        print(f"[错误] 下载图片失败: {e}")
        return False, None, current_user_remaining, str(e)

# --- 5. 批量生成图片 ---
success_count = 0
fail_count = 0

for idx, (sentence_range, positive_prompt) in enumerate(groups, 1):
    print(f"\n{'='*60}")
    print(f"正在处理第 {idx}/{len(groups)} 组：句子 {sentence_range}")
    print(f"{'='*60}")

    # 对当前图片尝试所有备选模型
    global_user_remaining = None
    success = False
    final_img = None
    last_error = None

    for model_idx, (model_id, display_name) in enumerate(MODEL_CANDIDATES, 1):
        print(f"\n尝试模型 {model_idx}/{len(MODEL_CANDIDATES)}: {display_name}")
        success, img, global_user_remaining, error = generate_with_model(
            model_id, display_name, positive_prompt, args.size, global_user_remaining
        )
        if success:
            final_img = img
            break
        else:
            print(f"[警告] 模型 {display_name} 失败: {error}")
            last_error = error
            if global_user_remaining is not None and global_user_remaining <= 0:
                print("用户总额度已用完，停止后续模型尝试。")
                break
            time.sleep(2)

    if success and final_img:
        # 生成文件名，例如 img_001_1-4.png
        img_filename = f"img_{idx:03d}_{sentence_range}.png"
        img_path = os.path.join(output_dir, img_filename)
        final_img.save(img_path)
        print(f"[成功] 图片已保存: {img_path}")
        success_count += 1
    else:
        print(f"[错误] 无法为句子 {sentence_range} 生成图片，最后错误: {last_error}")
        fail_count += 1

    # 每组之间稍作停顿，避免频繁请求
    time.sleep(2)

print(f"\n{'='*60}")
print(f"批量生成完成！成功: {success_count}, 失败: {fail_count}")
print(f"图片保存在: {output_dir}")