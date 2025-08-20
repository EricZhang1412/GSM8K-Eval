model_name = "/9950backfile/zjy_2/rwkv_loop/pretrained_models/Llama-3.2-1B-Instruct"

import transformers
import torch
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name, 
)

generator = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
)

# # https://huggingface.co/docs/transformers/main/chat_templating
# messages = [
#     {
#         "role": "system",
#         "content": "You are a friendly chatbot who always responds in the style of a pirate",
#     },
#     {
#         "role": "user",
#         "content": "Write me a mail.",
#     },
#     {
#         "role": "assistant",
#         "content": "To your friend Andrew?",
#     },
#     {
#         "role": "user",
#         "content": "Yes.",
#     },
# ]
# # Print the assistant's response
# output = pipe(messages, max_new_tokens=32, num_return_sequences=3)

import json

def load_jsonlines(file_name: str):
    with open(file_name, 'r') as f:
        return [json.loads(line) for line in f]

import random

def nshot_chats(nshot_data: list, n: int, question: str) -> dict:

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f'Answer: {s}'

    chats = []

    random.seed(42)
    for qna in random.sample(nshot_data, n):
        chats.append(
            {"role": "user", "content": question_prompt(qna["question"])})
        chats.append(
            {"role": "assistant", "content": answer_prompt(qna["answer"])})

    chats.append({"role": "user", "content": question_prompt(question)+" Let's think step by step. At the end, you MUST write the answer as an integer after '####'."})

    return chats


def extract_ans_from_response(answer: str, eos=None):
    if eos:
        answer = answer.split(eos)[0].strip()

    answer = answer.split('####')[-1].strip()

    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')

    try:
        return int(answer)
    except ValueError:
        return answer

def get_response(chats): 
    gen_text = generator(
        chats,
        max_new_tokens=256,
    )[0]  # First return sequence
    return gen_text['generated_text'][-1]['content']


train_data = load_jsonlines('/9950backfile/zjy_2/rwkv_loop/dataset/grade-school-math-master/grade_school_math/data/train.jsonl')
test_data = load_jsonlines('/9950backfile/zjy_2/rwkv_loop/dataset/grade-school-math-master/grade_school_math/data/test.jsonl')

N_SHOT = 8

# messages = nshot_chats(nshot_data=train_data, n=N_SHOT, question=test_data[0]['question'])  # 8-shot prompt

# response = get_response(messages)
# print(response)

# pred_ans = extract_ans_from_response(response)
# print(pred_ans)

import os
if not os.path.exists('log'):
    os.makedirs('log')

log_file_path = 'log/errors.txt'
with open(log_file_path, 'w') as log_file:
    log_file.write('')

from tqdm import tqdm
total = correct = 0
for qna in tqdm(test_data):

    messages = nshot_chats(nshot_data=train_data, n=N_SHOT, question=qna['question'])
    response = get_response(messages)
    
    pred_ans = extract_ans_from_response(response)
    true_ans = extract_ans_from_response(qna['answer'])

    total += 1
    if pred_ans != true_ans:
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{messages}\n\n")
            log_file.write(f"Response: {response}\n\n")
            log_file.write(f"Ground Truth: {qna['answer']}\n\n")
            log_file.write(f"Current Accuracy: {correct/total:.3f}\n\n")
            log_file.write('\n\n')
    else:
        correct += 1

print(f"Total Accuracy: {correct/total:.3f}")
