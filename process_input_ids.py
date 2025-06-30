import json
import numpy as np
from tqdm import tqdm  # 添加进度条库
from transformers import AutoTokenizer
from streaming import StreamingDataset, MDSWriter

def convert_prolong_to_qwen(
    input_mds_dir: str,
    jsonl_output_path: str,
    output_mds_dir: str,
    original_model: str = "princeton-nlp/Llama-3-8B-ProLong-64k-Base",
    qwen_model: str = "Qwen/Qwen2.5-1.5B",
    max_length: int = 1024*16
):
    # Step 0: 初始化进度条
    print("Initializing conversion process...")
    
    # Step 1: 解码 ProLong 数据
    print("\nStep 1/4: Decoding ProLong data...")
    original_tokenizer = AutoTokenizer.from_pretrained(original_model)

    def decode_function(sample):
        tokens = sample["input_ids"].tolist()
        return original_tokenizer.decode(tokens, skip_special_tokens=True)  # 直接返回文本

    dataset = StreamingDataset(
        local=input_mds_dir,
        shuffle=False,
        batch_size=32,
    )
    
    # 添加带进度条的解码过程
    text_samples = []
    for sample in tqdm(dataset, desc="Decoding samples", unit="sample"):
        text_samples.append(decode_function(sample))

    # Step 2: 用 Qwen tokenizer 编码
    print("\nStep 2/4: Tokenizing with Qwen...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model)
    
    # 添加带进度条的编码过程
    tokenized_samples = []
    for text in tqdm(text_samples, desc="Tokenizing", unit="sample"):
        tokenized_samples.append(qwen_tokenizer.encode(text, add_special_tokens=True))

    # Step 2.5: 保存 JSONL 文件
    print("\nStep 3/4: Saving JSONL file...")
    with open(jsonl_output_path, "w", encoding="utf-8") as f:
        total = len(text_samples)
        for i, (text, tokens) in tqdm(enumerate(zip(text_samples, tokenized_samples)), 
            total=total,
            desc="Writing records",
            unit="record"
        ):
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            record = {
                "text": text,
                "input_ids": tokens,
                "num_tokens": len(tokens)
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Step 3: 打包数据块
    print("\nStep 4/4: Packing data blocks...")
    packed_blocks = []
    current_block = []
    
    # 添加带进度条的打包过程
    for tokens in tqdm(tokenized_samples, desc="Packing", unit="sample"):
        while len(tokens) > 0:
            remaining = max_length - len(current_block)
            if remaining <= 0:
                packed_blocks.append(current_block)
                current_block = []
                remaining = max_length
            current_block.extend(tokens[:remaining])
            tokens = tokens[remaining:]
    if current_block:
        packed_blocks.append(current_block)

    # Step 4: 保存为 .mds
    print("\nFinalizing MDS dataset...")
    with MDSWriter(
        out=output_mds_dir,
        columns={"input_ids": "ndarray:int32"},
        compression=None,
    ) as writer:
        total_blocks = len(packed_blocks)
        for block in tqdm(packed_blocks, desc="Writing blocks", unit="block", total=total_blocks):
            writer.write({"input_ids": np.array(block, dtype=np.int32)})

    print("\nConversion completed successfully!")

# 运行
convert_prolong_to_qwen(
    "/hpc2hdd/home/qxiao183/SLM/PLM/plm_long/ProLong/datasets/long-context-65536",
    "/hpc2hdd/home/qxiao183/linweiquan/llm_train/prolong/datasets/long-context-65536.json",
    "/hpc2hdd/home/qxiao183/linweiquan/llm_train/prolong/datasets/long-context-65536"
)