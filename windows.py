import torch
import jieba
jieba.initialize()

# 创建一个空的集合来存储停用词
stopwords = set()

# 打开文件
with open(r'jupyter\停用词.txt', 'r', encoding='GB18030') as f:
    # 遍历文件的每一行
    for line in f:
        # 移除行尾的换行符并添加到set中
        stopwords.add(line.rstrip('\n'))

# 划词函数
def segment(text):
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

vocab = pkl.load(open(r"data\Processed_data\vocab.pkl", 'rb'))

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

# Make a prediction using the model
label_list = ['儿科', '内科', '外科', '妇产科', '男科', '肿瘤科']
output = 0

import tkinter as tk

class MyTinker:
    def __init__(self, root):
        self.model = self.load_model()
        self.root = root
        self.root.title("My Tinker")
        self.root.geometry("400x247")

        # input box
        self.input_label = tk.Label(root, text="请问您的问题是：")
        self.input_label.pack(pady=20)
        self.input_var = tk.StringVar()
        self.input_box = tk.Text(root, width=50, height=4)
        self.input_box.pack()

        # output box
        self.output_label = tk.Label(self.root, text="建议您挂号：")
        self.output_label.pack(pady=10)
        self.output_box = tk.Text(self.root, height=1, width=10)
        self.output_box.tag_configure("center", justify='center')
        # self.output_box.insert(1.0, "")
        self.output_box.tag_add("center", "1.0", "end")
        self.output_box.pack()

        # submit button
        self.submit_button = tk.Button(self.root, text="  提 交  ", command=self.submit)
        self.submit_button.pack(pady=10)

    def load_model(self):
        model_path = r"Chinese_medical_dialogue/saved_dict/TextRCNN.pkl"
        model = torch.load(model_path)
        model.eval()
        return model

    def predict(self, sentence):
        sentence = segment(sentence)

        # 使用join()函数以空格隔开列表中的元素
        sentence = ' '.join(list(set(sentence.split(" "))))
        print(sentence)

        sentence_ids = load_dataset(sentence)

        sentence_tensor = torch.tensor([sentence_ids[0][0]])
        device = torch.device("cuda:0")
        sentence_tensor = sentence_tensor.to(device)

        sentence_tensor = tuple([sentence_tensor, torch.tensor([0])])
        with torch.no_grad():
            output = self.model(sentence_tensor)
            prediction = label_list[output.argmax()]
        return prediction

    def submit(self):
        # call Special case testing.py file's predict function
        result = self.predict(self.input_box.get("1.0", "end-1c"))
        # update output box
        self.output_box.delete(1.0, "end")
        self.output_box.insert(1.0, result)
        self.output_box.tag_add("center", "1.0", "end")

root = tk.Tk()
my_tinker = MyTinker(root)
root.mainloop()

