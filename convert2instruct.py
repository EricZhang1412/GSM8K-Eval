import json

# 逐行读取原始数据集（jsonl 格式）
def load_data(input_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))  # 每行单独解析
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSONDecodeError: {e}")
    return data

# 转换数据格式
def convert_to_instruct_format(data):
    instruct_data = []
    
    for item in data:
        question = item.get('question', '')
        solution = item.get('solution', '')
        answer = item.get('answer', '')
        
        # 创建指令-响应对
        instruction = f"Please solve the following math problem: {question}"
        response = f"To solve this, {solution}"

        instruct_data.append({
            "instruction": instruction,
            "response": response,
            "answer": answer
        })
    
    return instruct_data

# 保存转换后的数据集
def save_to_file(data, output_file):
    with open(output_file, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')  # 每个 JSON 对象后添加换行符，保持 jsonl 格式

# 主函数
def main(input_file, output_file):
    # 加载原始数据集
    data = load_data(input_file)
    
    # 转换为Instruct格式
    instruct_data = convert_to_instruct_format(data)
    
    # 保存为新的文件
    save_to_file(instruct_data, output_file)

# 文件路径
input_file = '/9950backfile/zjy_2/rwkv_loop/dataset/grade-school-math-master/grade_school_math/data/test.jsonl'  # 原始数据集文件路径
output_file = '/9950backfile/zjy_2/rwkv_loop/dataset/grade-school-math-master/grade_school_math_instruct/data/test.jsonl'  # 输出的Instruct数据集文件路径

# 执行脚本
main(input_file, output_file)
