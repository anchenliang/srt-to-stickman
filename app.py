import os
import sys
import uuid
import json
import threading
import subprocess
import shutil
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('tmp', exist_ok=True)
os.makedirs('output', exist_ok=True)

tasks = {}
tasks_lock = threading.Lock()

def update_task(task_id, **kwargs):
    with tasks_lock:
        tasks[task_id].update(kwargs)

def run_pipeline(task_id, srt_path, base_name):
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    try:
        # 阶段1: 语义分组
        update_task(task_id, status='processing', phase='srt', message='正在进行语义分组...')
        cmd_srt = [sys.executable, 'srt_pre_prompt.py', srt_path, '--output_dir', os.path.join('tmp', base_name)]
        result = subprocess.run(cmd_srt, capture_output=True, text=True, timeout=300, env=env, encoding='utf-8')
        if result.returncode != 0:
            raise Exception(f"语义分组失败: {result.stderr}")
        update_task(task_id, progress=30, message='语义分组完成，正在生成提示词...')

        # 阶段2: 生成最终提示词
        update_task(task_id, phase='llm', message='调用 DeepSeek 生成提示词...')
        cmd_llm = [sys.executable, 'llm_prompt_generator.py', '--srt_file', srt_path]
        result = subprocess.run(cmd_llm, capture_output=True, text=True, timeout=600, env=env, encoding='utf-8')
        if result.returncode != 0:
            raise Exception(f"生成提示词失败: {result.stderr}")
        update_task(task_id, progress=60, message='提示词生成完成，正在生成图片...')

        # 阶段3: 生成图片
        update_task(task_id, phase='image', message='调用文生图模型生成图片...')
        cmd_img = [sys.executable, 'generate_image.py', '--srt_file', srt_path]
        result = subprocess.run(cmd_img, capture_output=True, text=True, timeout=1800, env=env, encoding='utf-8')
        if result.returncode != 0:
            raise Exception(f"生成图片失败: {result.stderr}")

        update_task(task_id, status='completed', phase='done', progress=100, message='所有图片生成完毕！', output_dir=os.path.join('output', base_name))
    except Exception as e:
        update_task(task_id, status='failed', message=str(e))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400
    if not file.filename.lower().endswith('.srt'):
        return jsonify({'error': '只支持 .srt 文件'}), 400

    original_filename = secure_filename(file.filename)
    if not original_filename:
        original_filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    file.save(filepath)

    base_name = os.path.splitext(original_filename)[0].replace(' ', '_')
    task_id = str(uuid.uuid4())

    with tasks_lock:
        tasks[task_id] = {
            'status': 'pending',
            'phase': 'init',
            'progress': 0,
            'message': '任务已提交，正在启动...',
            'base_name': base_name,
            'srt_path': filepath
        }

    thread = threading.Thread(target=run_pipeline, args=(task_id, filepath, base_name))
    thread.daemon = True
    thread.start()

    return jsonify({'task_id': task_id})

@app.route('/status/<task_id>')
def status(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
        if not task:
            return jsonify({'error': '任务不存在'}), 404
        status_info = {
            'status': task['status'],
            'phase': task.get('phase', ''),
            'progress': task.get('progress', 0),
            'message': task.get('message', '')
        }
        if task['status'] == 'completed':
            output_dir = task.get('output_dir')
            if output_dir and os.path.exists(output_dir):
                images = [f for f in sorted(os.listdir(output_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                status_info['images'] = images
        return jsonify(status_info)

@app.route('/download/<task_id>/<filename>')
def download(task_id, filename):
    with tasks_lock:
        task = tasks.get(task_id)
        if not task or task['status'] != 'completed':
            return jsonify({'error': '任务未完成'}), 404
        output_dir = task.get('output_dir')
        if not output_dir:
            return jsonify({'error': '输出目录不存在'}), 404
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': '文件不存在'}), 404
        return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)