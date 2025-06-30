import json
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from streaming import StreamingDataset, MDSWriter

def convert_single_prolong_dataset(
    input_mds_dir: str,
    output_base_dir: str,
    dataset_name: str,
    original_model: str,
    qwen_model: str,
    max_length: int = 1024*16
):
    """处理单个数据集目录"""
    print(f"\n📂 Processing dataset: {dataset_name}")
    
    # 创建输出目录
    jsonl_output_path = os.path.join(output_base_dir, f"{dataset_name}.jsonl")
    output_mds_dir = os.path.join(output_base_dir, f"{dataset_name}-mds")
    
    # 验证输入目录
    index_path = os.path.join(input_mds_dir, 'index.json')
    if not os.path.exists(index_path):
        print(f"⚠️ Warning: Skipping {dataset_name} - index.json not found at {index_path}")
        return
    
    # Step 1: 解码 ProLong 数据
    print("  Step 1/4: Decoding ProLong data...")
    original_tokenizer = AutoTokenizer.from_pretrained(original_model)

    def decode_function(sample):
        tokens = sample["input_ids"].tolist()
        return original_tokenizer.decode(tokens, skip_special_tokens=True)

    # 创建数据集
    try:
        dataset = StreamingDataset(
            local=input_mds_dir,
            remote=input_mds_dir,
            shuffle=False,
            batch_size=32,
        )
    except Exception as e:
        print(f"❌ Failed to create StreamingDataset for {dataset_name}: {str(e)}")
        return
    
    # 解码所有样本
    text_samples = []
    try:
        for sample in tqdm(dataset, desc="Decoding samples", unit="sample", leave=False):
            text_samples.append(decode_function(sample))
    except Exception as e:
        print(f"❌ Decoding failed for {dataset_name}: {str(e)}")
        return

    # Step 2: 用 Qwen tokenizer 编码
    print("  Step 2/4: Tokenizing with Qwen...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model)
    
    tokenized_samples = []
    try:
        for text in tqdm(text_samples, desc="Tokenizing", unit="sample", leave=False):
            tokenized_samples.append(qwen_tokenizer.encode(text, add_special_tokens=True))
    except Exception as e:
        print(f"❌ Tokenization failed for {dataset_name}: {str(e)}")
        return

    # Step 3: 保存 JSONL 文件
    print("  Step 3/4: Saving JSONL file...")
    try:
        os.makedirs(os.path.dirname(jsonl_output_path), exist_ok=True)
        with open(jsonl_output_path, "w", encoding="utf-8") as f:
            for text, tokens in tqdm(zip(text_samples, tokenized_samples), 
                total=len(text_samples),
                desc="Writing JSONL",
                unit="record",
                leave=False
            ):
                record = {
                    "text": text,
                    "input_ids": tokens,
                    "num_tokens": len(tokens)
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"❌ JSONL save failed for {dataset_name}: {str(e)}")
        return

    # Step 4: 打包数据块
    print("  Step 4/4: Packing data blocks...")
    packed_blocks = []
    current_block = []
    
    try:
        for tokens in tqdm(tokenized_samples, desc="Packing", unit="sample", leave=False):
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
    except Exception as e:
        print(f"❌ Packing failed for {dataset_name}: {str(e)}")
        return

    # 保存为 .mds
    print("  Finalizing MDS dataset...")
    try:
        os.makedirs(output_mds_dir, exist_ok=True)
        with MDSWriter(
            out=output_mds_dir,
            columns={"input_ids": "ndarray:int32"},
            compression=None,
        ) as writer:
            for block in tqdm(packed_blocks, desc="Writing blocks", unit="block", leave=False):
                writer.write({"input_ids": np.array(block, dtype=np.int32)})
    except Exception as e:
        print(f"❌ MDS save failed for {dataset_name}: {str(e)}")
        return
    
    print(f"✅ Successfully processed {dataset_name}")
    print(f"   - JSONL: {jsonl_output_path}")
    print(f"   - MDS: {output_mds_dir}")

def convert_prolong_data(
    input_path: str,
    output_base_dir: str,
    original_model: str = "princeton-nlp/Llama-3-8B-ProLong-64k-Base",
    qwen_model: str = "Qwen/Qwen2.5-1.5B",
    max_length: int = 1024*16
):
    """智能处理 ProLong 数据，自动检测单数据集或多数据集模式"""
    print(f"🚀 Starting conversion for: {input_path}")
    
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    # 判断处理模式：单数据集还是多数据集
    index_path = os.path.join(input_path, 'index.json')
    
    if os.path.exists(index_path):
        # 单数据集模式：输入目录本身就是数据集
        print("🔍 Detected single dataset mode")
        dataset_name = os.path.basename(input_path.rstrip('/'))
        convert_single_prolong_dataset(
            input_path,
            output_base_dir,
            dataset_name,
            original_model,
            qwen_model,
            max_length
        )
    else:
        # 多数据集模式：输入目录包含多个子数据集
        print("🔍 Detected multi-dataset mode")
        
        # 获取所有子目录（每个都是一个数据集）
        all_datasets = [d for d in os.listdir(input_path) 
                      if os.path.isdir(os.path.join(input_path, d))]
        
        print(f"Found {len(all_datasets)} datasets to process:")
        for i, dataset in enumerate(all_datasets, 1):
            print(f"  {i}. {dataset}")
        
        # 处理每个数据集
        for dataset_name in all_datasets:
            input_dir = os.path.join(input_path, dataset_name)
            convert_single_prolong_dataset(
                input_dir,
                output_base_dir,
                dataset_name,
                original_model,
                qwen_model,
                max_length
            )
    
    print("\n🏁 All datasets processed successfully!")
    print(f"Output saved to: {output_base_dir}")

# 示例用法：
if __name__ == "__main__":
    # 情况1：处理单个数据集（目录包含index.json）
    # convert_prolong_data(
    #     input_path="/path/to/single-dataset",
    #     output_base_dir="/output/dir"
    # )
    
    # 情况2：处理包含多个数据集的目录（子目录各含index.json）
    convert_prolong_data(
        input_path="/hpc2hdd/home/qxiao183/SLM/PLM/plm_long/ProLong/datasets/long-context-65536",
        output_base_dir="/hpc2hdd/home/qxiao183/linweiquan/llm_train/prolong/datasets"
    )