import torch
import torch.nn.functional as F
import os
import json
import re
import time
import sys
import platform


# ===============================
# 1. 纯手工迷你分词器
# ===============================
class MiniTokenizer:
    def __init__(self, json_path):
        self.vocab = {}
        self.id2token = {}
        with open(json_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        self.vocab = tokenizer_data['model']['vocab']
        for token, idx in self.vocab.items():
            self.id2token[idx] = token
        
        self.cls_id = self.vocab.get("[CLS]", 101)
        self.sep_id = self.vocab.get("[SEP]", 102)
        self.mask_id = self.vocab.get("[MASK]", 103)
        self.unk_id = self.vocab.get("[UNK]", 100)
        self.pad_id = self.vocab.get("[PAD]", 0)

    def encode(self, text, max_length=512):
        tokens = [self.cls_id]
        for char in text:
            tokens.append(self.vocab.get(char, self.unk_id))
        tokens.append(self.sep_id)
        
        actual_len = len(tokens)
        attention_mask = [1] * actual_len
        token_type_ids = [0] * actual_len
        
        padding_len = max_length - actual_len
        if padding_len > 0:
            tokens += [self.pad_id] * padding_len
            attention_mask += [0] * padding_len
            token_type_ids += [0] * padding_len
        else:
            tokens = tokens[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]
        return tokens, attention_mask, token_type_ids

    def decode_id(self, idx):
        return self.id2token.get(idx, "?")

# ===============================
# 2. 模型初始化
# ===============================
print("正在加载RoBERTa-wwm-ext-Large纠错模型...")
tokenizer_path = "./tokenizer.json"
pt_model_path = "./model_traced.pt"

if not os.path.exists(pt_model_path) or not os.path.exists(tokenizer_path):
    print(f"❌ 找不到文件！")
    exit()

if 'aarch' in platform.machine().lower() or 'arm' in platform.machine().lower():
    torch.backends.quantized.engine = 'qnnpack'
    print("已启用移动端 QNNPACK 量化引擎！")

torch.set_num_threads(4)
tokenizer = MiniTokenizer(tokenizer_path)
model = torch.jit.load(pt_model_path)
model.eval()

try:
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print("已使用INT8动态量化！")
except Exception as e:
    print(f"⚠️ 量化失败({e})，使用原始FP32模型。")

DE_ID = tokenizer.vocab.get('的', tokenizer.unk_id)
DI_ID = tokenizer.vocab.get('地', tokenizer.unk_id)
DE2_ID = tokenizer.vocab.get('得', tokenizer.unk_id)
TARGET_IDS_T = torch.tensor([DE_ID, DI_ID, DE2_ID])
TARGET_LABELS = ['的', '地', '得']

print("Large模型加载完毕。")

# ===============================
# 3. 绝对白名单机制 (不变)
# ===============================
IGNORE_WORDS = [
    "地产", "地道",
    "地表", "地步", "地主", "心地", "领地", "驻地",
    "阵地", "境地", "殖民地", "根据地", "发源地", "本地", "外地", "内地", "地下", "地面",
    "得罪", "得意", "得体", "心得", "舍不得", "值得", "懂得", "记得", "获得", "取得", "赢得", "免得", "懒得",
    "的确", "的士", "目的", "标的", "有的放矢", "众矢之的","懂的","倒地","怎的",
]

def is_in_ignore_words(text, pos):
    for word in IGNORE_WORDS:
        start_idx = 0
        while True:
            idx = text.find(word, start_idx)
            if idx == -1: break
            if idx <= pos < idx + len(word):
                return True
            start_idx = idx + 1
    return False

# ===============================
# 4. 核心功能：切分与纠错
# ===============================

def correct_chunk(text):
    chars = list(text)
    changes = []
    has_changed = False
    
    
    TEMPERATURE = 1.2           
    CONFIDENCE_THRESHOLD = 0.7  # 置信度阈值
    
    # 收集目标位置
    target_positions = []
    for i, char in enumerate(chars):
        if char in ('的', '地', '得') and not is_in_ignore_words(text, i):
            target_positions.append(i)
            
    if not target_positions:
        return False, text, text, []

    exact_length = min(len(chars) + 2, 512) 
    
    input_ids, attention_mask, token_type_ids = tokenizer.encode(text, max_length=exact_length)
    
    # 遮蔽目标位置
    # 训练时模型看到的是[MASK]，推理时也必须是[MASK]
    for i in target_positions:
        mask_pos = i + 1  # +1 因为 [CLS] 在位置0
        if mask_pos < exact_length:
            input_ids[mask_pos] = tokenizer.mask_id  # 替换为 [MASK]=103
    
    input_ids_t = torch.tensor([input_ids], dtype=torch.long)
    attn_mask_t = torch.tensor([attention_mask], dtype=torch.long)
    tok_type_t = torch.tensor([token_type_ids], dtype=torch.long)
    
    with torch.inference_mode():
        logits = model(input_ids_t, attn_mask_t, tok_type_t)[0]
        
    logits_slice = logits[0] / TEMPERATURE
    
    for i in target_positions:
        mask_pos = i + 1 
        if mask_pos >= exact_length:
            continue
            
        char = chars[i]
        
        three_logits = logits_slice[mask_pos, TARGET_IDS_T]
        three_probs = F.softmax(three_logits, dim=-1)
        max_val, max_idx = torch.max(three_probs, dim=-1)
        
        pred_char = TARGET_LABELS[max_idx.item()]
        confidence = max_val.item()
        
        if pred_char != char and confidence >= CONFIDENCE_THRESHOLD:
            changes.append(f"【{char}】->【{pred_char}】(置信度:{confidence:.1%})")
            chars[i] = pred_char
            has_changed = True

    corrected_text = "".join(chars)
    return has_changed, text, corrected_text, changes

def correct_article(long_text):
    chunks = re.split(r'(?<=[。！？\n])', long_text)
    final_text = ""
    logs = []
    
    all_sub_chunks = []
    for chunk in chunks:
        if chunk.strip():
            sub_chunks = [chunk[i:i+365] for i in range(0, len(chunk), 365)]
            all_sub_chunks.extend(sub_chunks)
        else:
            all_sub_chunks.append(chunk)
            
    total_chunks = len([sc for sc in all_sub_chunks if sc.strip()])
    processed = 0
    start_time = time.time()
    
    for sc in all_sub_chunks:
        if not sc.strip():
            final_text += sc
            continue
            
        has_changed, old_sc, new_sc, changes = correct_chunk(sc)
        final_text += new_sc
        
        if has_changed:
            sys.stdout.write('\r' + ' ' * 70 + '\r')
            sys.stdout.flush()
            print(f"错误：{' | '.join(changes)}")
            print(f"❌ 原句：{old_sc.strip()}")
            print(f"✅ 改后：{new_sc.strip()}")
            print("-" * 40)
            logs.append({'old': old_sc, 'new': new_sc, 'details': changes})
            
        processed += 1
        elapsed = time.time() - start_time  
        avg_time = elapsed / processed  
        eta = avg_time * (total_chunks - processed)  
        
        # 新增百分比计算
        percent = (processed / total_chunks) * 100
          
        # 修改输出格式，加入百分比
        progress_str = f"进度: {percent:.1f}% ({processed}/{total_chunks}) | 用时: {elapsed:.1f}秒 | 预计剩: {eta:.1f}秒    "  
        sys.stdout.write('\r' + progress_str)  
        sys.stdout.flush()
        
    print()
    return final_text, logs

# ===============================
# 5. 交互与输出 (不变)
# ===============================
def interactive_loop():
    END_BLANK_LINES = 2
    print("\n" + "="*40)
    print("RoBERTa-wwm-ext-Large纠错 (遮蔽预测版)")
    print("粘贴文本后，连续按 3 次回车开始。")
    print("输入 'q' 退出。")
    print("="*40)

    while True:
        print("\n粘贴/输入内容：")
        lines = []
        blank_count = 0

        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip().lower() == 'q' and not lines:
                print("👋 再见！")
                return
            if line == "":
                blank_count += 1
                if blank_count >= END_BLANK_LINES: break
                lines.append("")
            else:
                blank_count = 0
                lines.append(line)

        raw_text = "\n".join(lines).strip()
        if not raw_text: continue
            
        corrected_text, logs = correct_article(raw_text)

        if len(logs) == 0:
            print("未检测到错误。")
        else:
            print(f"审核完毕，共修正了 {len(logs)} 处。\n")
            
            print("修正后全文（可直接复制）：\n")
          
            print(corrected_text)
            print("——" * 20)
        

if __name__ == "__main__":
    if os.path.exists("文章.txt"):
        print("\n检测到 [文章.txt]，正在执行全文批量纠错...\n")
        with open("文章.txt", "r", encoding="utf-8") as f:
            novel_text = f.read()
            
        if novel_text.strip():
            corrected_novel, logs = correct_article(novel_text)
            
            if len(logs) == 0:
                print("文件内未检测到任何错误。")
            else:
                print(f"全文审查完毕，共修正了 {len(logs)} 处错误。")
                
            with open("改——文章.txt", "w", encoding="utf-8") as f:
                f.write(corrected_novel)
            print(f"结果已保存为 [改——文章.txt]。\n")
            
    interactive_loop()
