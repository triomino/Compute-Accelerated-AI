# 昇腾平台 DeepSeek 模型 Token 输出数量对比

## 目的

对比华为昇腾平台上，不同推理框架（mindie、vllm-ascend）下 DeepSeek 模型的 token 输出数量，是否和 DeepSeek 官方宣称保持一致，具体而言就是验证 V3.1 Terminus token 数量少于 R1。

## 结果

vllm-ascend 和 mindie 上，DeepSeek V3.1 Terminus 输出 token 数少于 DeepSeek R1。

### 数据总结
所有测试的结果汇总：
| 框架 | 模型 | Q1 avg (tokens) | Q2 avg (tokens) | Q3 avg (tokens) | Q4 avg (tokens) | Q5 avg (tokens) | 总平均 (tokens) |
|------|------|--------|--------|--------|--------|--------|--------|
| vllm-ascend | R1 | 393.6 | 1033.3 | 717.1 | 330.5 | 283.6 | 551.6 |
| vllm-ascend | Terminus | 286.7 | 503.5 | 476.7 | 172.5 | 171.2 | 322.1 |
| MindIE | R1 | 426.4 | 1164.8 | 736.1 | 380.9 | 293.7 | 600.4 |
| MindIE | Terminus | 263.6 | 426.2 | 484.3 | 183.0 | 166.6 | 304.7 |

下面两节是具体数据
### vllm-ascend 框架

**DeepSeek R1**

| Run | Q1 (tokens) | Q2 (tokens) | Q3 (tokens) | Q4 (tokens) | Q5 (tokens) |
|-----|-----|-----|-----|-----|-----|
| 1 | 440 | 564 | 679 | 275 | 311 |
| 2 | 492 | 609 | 743 | 353 | 322 |
| 3 | 439 | 647 | 721 | 381 | 353 |
| 4 | 436 | 627 | 764 | 407 | 327 |
| 5 | 247 | 2421 | 652 | 348 | 273 |
| 6 | 345 | 669 | 873 | 342 | 171 |
| 7 | 342 | 825 | 777 | 317 | 348 |
| 8 | 402 | 978 | 637 | 248 | 305 |
| 9 | 509 | 572 | 637 | 300 | 301 |
| 10 | 284 | 2421 | 688 | 334 | 125 |
| **avg** | **393.6** | **1033.3** | **717.1** | **330.5** | **283.6** |

**DeepSeek V3.1 Terminus**

| Run | Q1 (tokens) | Q2 (tokens) | Q3 (tokens) | Q4 (tokens) | Q5 (tokens) |
|-----|-----|-----|-----|-----|-----|
| 1 | 330 | 403 | 478 | 146 | 215 |
| 2 | 283 | 1357 | 471 | 190 | 167 |
| 3 | 400 | 420 | 451 | 191 | 151 |
| 4 | 275 | 438 | 483 | 164 | 149 |
| 5 | 277 | 383 | 504 | 163 | 139 |
| 6 | 212 | 464 | 462 | 163 | 185 |
| 7 | 284 | 377 | 484 | 167 | 151 |
| 8 | 264 | 361 | 485 | 161 | 234 |
| 9 | 321 | 447 | 480 | 190 | 151 |
| 10 | 221 | 385 | 469 | 190 | 170 |
| **avg** | **286.7** | **503.5** | **476.7** | **172.5** | **171.2** |

### MindIE 框架

**DeepSeek R1**

| Run | Q1 (tokens) | Q2 (tokens) | Q3 (tokens) | Q4 (tokens) | Q5 (tokens) |
|-----|-----|-----|-----|-----|-----|
| 1 | 321 | 502 | 813 | 446 | 303 |
| 2 | 510 | 1053 | 663 | 480 | 300 |
| 3 | 356 | 2441 | 687 | 293 | 272 |
| 4 | 545 | 2107 | 751 | 393 | 310 |
| 5 | 476 | 649 | 745 | 397 | 305 |
| 6 | 371 | 740 | 697 | 290 | 309 |
| 7 | 444 | 2211 | 726 | 363 | 321 |
| 8 | 439 | 604 | 745 | 413 | 318 |
| 9 | 386 | 566 | 714 | 413 | 334 |
| 10 | 416 | 775 | 820 | 321 | 165 |
| **avg** | **426.4** | **1164.8** | **736.1** | **380.9** | **293.7** |

**DeepSeek V3.1 Terminus**

| Run | Q1 (tokens) | Q2 (tokens) | Q3 (tokens) | Q4 (tokens) | Q5 (tokens) |
|-----|-----|-----|-----|-----|-----|
| 1 | 304 | 387 | 499 | 163 | 152 |
| 2 | 281 | 489 | 477 | 239 | 148 |
| 3 | 350 | 395 | 461 | 230 | 158 |
| 4 | 277 | 437 | 468 | 217 | 167 |
| 5 | 247 | 392 | 481 | 163 | 186 |
| 6 | 236 | 371 | 490 | 167 | 163 |
| 7 | 243 | 490 | 465 | 162 | 162 |
| 8 | 208 | 361 | 503 | 163 | 204 |
| 9 | 273 | 442 | 491 | 161 | 163 |
| 10 | 217 | 498 | 508 | 165 | 163 |
| **avg** | **263.6** | **426.2** | **484.3** | **183** | **166.6** |

## 版本
- **mindie 版本**: 2.1.RC2
- **vllm-ascend 版本**: 0.13.0

### 模型信息

- **DeepSeek R1**: https://www.modelscope.cn/models/Eco-Tech/DeepSeek-R1-0528-w8a8-mtp-QuaRot
- **DeepSeek V3.1 Terminus**: https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V3.1-w8a8-mtp-QuaRot

## 脚本（含问题和参数）
用以下脚本提问
```python
# ds_client_v31_t.py
from openai import OpenAI
from transformers import AutoTokenizer

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1025/v1"
model_path = "/home/z00693113/weights/deepseek/ds-terminus3.1-w8a8"

def count_tokens(text, model_path):
    try:
        # 加载分词器，trust_remote_code=True 是因为 DeepSeek 可能包含自定义逻辑
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # 将文本转换为 token id 列表
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # 返回数量
        return len(tokens)
    except Exception as e:
        return f"加载出错: {str(e)}"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

sampling_params = {
    "temperature": 0.1,        # 低温度，减少随机性
    "top_p": 0.9,             # 核采样参数
    "top_k": 50,              # top-k采样
    "max_tokens": 4096,       # 最大生成token数
    "frequency_penalty": 0.1,  # 频率惩罚
    "presence_penalty": 0.1,   # 存在惩罚
    "stop": ["<|endoftext|>", "<|im_end|>"],  # 停止词
    "repetition_penalty": 1.1, # 重复惩罚
    "seed": 1024
}

test_questions = [
    "9.11 and 9.8, which is greater?",
    "How many Rs are there in the word 'strawberry'?",
    "小明家要重新粉刷客厅的墙壁。客厅长6米，宽4米，高3米，门窗总面积是8平方米。\n1. 需要粉刷的墙壁总面积是多少平方米？",
    "Apple Sharing Problem:\nThere are 24 apples to be shared among 3 children.\n1. Tom gets 1/3 of the apples. How many apples does Tom get?",
    "The classroom is rectangular: length = 8 meters, width = 6 meters\n1. What is the area of the whole classroom"
]

for question in test_questions:
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "<think>Hmm</think>I am DeepSeek"},
        {"role": "user", "content": question},
    ]
    extra_body = {"chat_template_kwargs": {"enable_thinking": True, "thinking": True}}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=sampling_params["temperature"],
        top_p=sampling_params["top_p"],
        max_tokens=sampling_params["max_tokens"],
        seed=sampling_params["seed"],
        extra_body=extra_body
    )
    content = response.choices[0].message.content
    print("content:\n", content)
    print(f"usage {response.usage}")
```
把上面的脚本执行十次并保存日志。
```bash
for i in {1..10}; do     python ds_client_v31_t.py | tee "v31_counttoken_$i.log"; done
```

用以下脚本从日志中提取 token 数量
```python
# collect_token_log.py
import re
import glob
import os
import csv

def extract_tokens_to_csv_horizontal():
    # 1. 获取所有日志文件并按数字顺序排序
    log_files = glob.glob("v31_counttoken_*.log")
    # 按照文件名中的数字进行自然排序 (1, 2, 3... 10)
    log_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else x)

    # 2. 正则表达式：提取 completion_tokens= 后面的数字
    token_pattern = re.compile(r"completion_tokens=(\d+)")

    output_file = "token_usage_report.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        # 3. 写入表头：文件名 + 5个数据列
        header = ["文件名", "Question_1", "Question_2", "Question_3", "Question_4", "Question_5"]
        writer.writerow(header)

        # 4. 遍历文件并提取数据
        for file_path in log_files:
            filename = os.path.basename(file_path)
            with open(file_path, 'r', encoding='utf-8') as f_in:
                content = f_in.read()
                tokens = token_pattern.findall(content)
                
                # 确保每行数据长度一致（如果某个文件不足5条，用空字符串填充）
                row = [filename] + tokens[:5]
                while len(row) < 6:
                    row.append("")
                
                writer.writerow(row)

    print(f"解析完成！结果已保存至: {output_file}")

if __name__ == "__main__":
    extract_tokens_to_csv_horizontal()
```
最后结果保存在 token_usage_report.csv 里。
