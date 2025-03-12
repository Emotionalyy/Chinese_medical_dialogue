# 创建一个空的集合来存储停用词
stopwords = set()

# 打开文件
with open(r'停用词.txt', 'r', encoding='GB18030') as f:
    # 遍历文件的每一行
    for line in f:
        # 移除行尾的换行符并添加到set中
        stopwords.add(line.rstrip('\n'))
# 划词函数
def segment(text):
    import jieba
    words = jieba.lcut(text)
    temp = []
    for word in words:
        if word not in stopwords:
            temp.append(word)
    # 过滤
    # word = filter(lambda x: len(x) > 1, word)
    return ' '.join(temp)

# 将句子转换为整数列表
tokenizer = lambda x: x.split(' ')
import pickle as pkl
vocab = pkl.load(open(r"..\data\Processed_data\vocab.pkl", 'rb'))
print(f"Vocab size: {len(vocab)}")

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
def load_dataset(sentence, pad_size=32):
    contents = []
    content, label = sentence, 0
    words_line = []
    token = tokenizer(content)
    seq_len = len(token)
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size
    # word to id
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    contents.append((words_line, int(label), seq_len))
    return contents  # [([...], 0)]

import torch

sentence = "室友想勃起困难"
sentence = segment(sentence)

# 使用join()函数以空格隔开列表中的元素
sentence = ' '.join(list(set(sentence.split(" "))))

sentence_ids = load_dataset(sentence)

sentence_tensor = torch.tensor([sentence_ids[0][0]])
device = torch.device("cuda:0")
sentence_tensor = sentence_tensor.to(device)

sentence_tensor = tuple([sentence_tensor, torch.tensor([0])])

model_path = r"..\Chinese_medical_dialogue\saved_dict\TextRCNN.pkl"
model = torch.load(model_path)
model.eval()

# Make a prediction using the model
label_list = ['儿科', '内科', '外科', '妇产科', '男科', '肿瘤科']
output = 0
with torch.no_grad():
    output = model(sentence_tensor)
    prediction = label_list[output.argmax()]

print(f"The predicted label for '{sentence}' is '{prediction}'.")
print(output.tolist())