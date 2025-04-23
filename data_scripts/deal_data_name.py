import json
import jsonlines
import os



def deal_lora_identity():
    with open("./dataset/lora_identity.jsonl", 'r') as f:
        for line in f:
            if("MiniMind" in line):
                line = line.replace("MiniMind", "MiniLLM")
            if("Jingyao Gong" in line):
                line = line.replace("Jingyao Gong", "YanPing Zhou")
            data = json.loads(line)
            with jsonlines.open("./dataset/lora_identity1.jsonl", 'a') as f1:
                f1.write(data)

def deal_r1_mix():
    with open("./dataset/r1_mix_1024.jsonl", 'r') as f:
        for line in f:
            if("MiniMind" in line):
                line = line.replace("MiniMind", "MiniLLM")
            if("Jingyao Gong" in line):
                line = line.replace("Jingyao Gong", "YanPing Zhou")
            data = json.loads(line)
            with jsonlines.open("./dataset/r1_mix_10241.jsonl", 'a') as f1:
                f1.write(data)

def test_data():
    with open("./dataset/sft_mini_512.jsonl", 'r') as f:
        for line in f:
            if("MiniMind" in line):
                print(line)
            if("Jingyao Gong" in line):
                print(line)

if __name__ == "__main__":
    # deal_lora_identity()
    # deal_r1_mix()
    
    # test_data()
    ...