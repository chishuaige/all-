from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
# print("Tensorflow Version:", tf.__version__)   # 打印tensorflow版本
import numpy as np
import os
import time

# 使用tf.keras.utils.get_file方法从指定地址下载数据，得到原始数据本地路径
path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# 读取数据:
# 打开原始数据文件并读取文本内容
text = open(path_to_file, 'r', encoding='utf-8').read()
# print(type(text))   # <class 'str'>
# # 统计字符个数并查看前250个字符
print('Length of text: {} characters'.format(len(text)))
print(text[:250])
# 统计文本中非重复字符数量
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# 65 unique characters

# 对文本进行数值映射:
# 对字符进行数值映射，将创建两个映射表：字符映射成数字，数字映射成字符
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
# 使用字符到数字的映射表示所有文本
text_as_int = np.array([char2idx[c] for c in text])
# 查看映射表
print('{')
for char, _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

# 查看原始语料前13个字符映射后的结果
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# 生成训练数据
# 设定输入序列长度
seq_length = 100
# 获得样本总数
examples_per_epoch = len(text)//seq_length
#  加载数据，将数值映射后的文本转换成dataset对象方便后续处理
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# 通过dataset的take方法以及映射表查看前5个字符
for i in char_dataset.take(5):
    print(idx2char[i.numpy()])
# F
# i
# r
# s
# t
# 使用dataset的batch方法按照字符长度+1划分（要留出一个向后顺移的位置）
# drop_remainder=True表示删除掉最后一批可能小于批次数量的数据
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# 查看划分后的5条数据对应的文本内容
for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))
    
# 'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou '
# 'are all resolved rather to die than to famish?\n\nAll:\nResolved. resolved.\n\nFirst Citizen:\nFirst, you k'
# "now Caius Marcius is chief enemy to the people.\n\nAll:\nWe know't, we know't.\n\nFirst Citizen:\nLet us ki"
# "ll him, and we'll have corn at our own price.\nIs't a verdict?\n\nAll:\nNo more talking on't; let it be d"
# 'one: away, away!\n\nSecond Citizen:\nOne word, good citizens.\n\nFirst Citizen:\nWe are accounted poor citi'

def split_input_target(chunk):
    """划分输入序列和目标序列函数"""
    # 前100个字符为输入序列，第二个字符开始到最后为目标序列
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# 使用map方法调用该函数对每条序列进行划分
dataset = sequences.map(split_input_target)

# # 查看划分后的第一批次结果
# for input_example, target_example in dataset.take(1):
#     print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
#     print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

# 输出效果:
# Input data:  'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou'
# Target data: 'irst Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou '

# 查看将要输入模型中的每个时间步的输入和输出(以前五步为例)
# # 循环每个字符，并打印每个时间步对应的输入和输出
# for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
#     print("Step {:4d}".format(i))
#     print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
#     print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

# 输出效果：
# Step    0
#   input: 18 ('F')
#   expected output: 47 ('i')
# Step    1
#   input: 47 ('i')
#   expected output: 56 ('r')
# Step    2
#   input: 56 ('r')
#   expected output: 57 ('s')
# Step    3
#   input: 57 ('s')
#   expected output: 58 ('t')
# Step    4
#   input: 58 ('t')
#   expected output: 1 (' ')

# 创建批次数据
# 定义批次大小为64
BATCH_SIZE = 64
# 设定缓冲区大小，以重新排列数据集
# 缓冲区越大数据混乱程度越高，所需内存也越大
BUFFER_SIZE = 10000
# 打乱数据并分批次
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# 打印数据集对象查看数据张量形状
print(dataset)
#输出效果：
# <BatchDataset shapes: ((64, 100), (64, 100)), types: (tf.int64, tf.int64)>

# 第二步: 构建模型并训练模型
# 获得词汇集大小
vocab_size = len(vocab)

# 定义词嵌入维度
embedding_dim = 256

# 定义GRU的隐层节点数量
rnn_units = 1024


# 模型包括三个层：输入层即embedding层，中间层即GRU层（详情查看）输出层即全连接层
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """模型构建函数"""
    # 使用tf.keras.Sequential定义模型
    # GRU层的参数return_sequences为True说明返回结果为每个时间步的输出，而不是最后时间步的输出
    # stateful参数为True，说明将保留每个batch数据的结果状态作为下一个batch的初始化数据
    # recurrent_initializer='glorot_uniform'，说明GRU的循环核采用均匀分布的初始化方法
    # 模型最终通过全连接层返回一个所有可能字符的概率分布.
    model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
      tf.keras.layers.GRU(rnn_units,
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='glorot_uniform'),
      tf.keras.layers.Dense(vocab_size)
    ])
    return model

# 传入超参数构建模型
model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

# 训练前试用模型:
# 使用一个批次的数据作为输入
# 查看通过模型后的结果形状是否满足预期
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
# 输出效果：
# (64, 100, 65) # (batch_size, sequence_length, vocab_size)
# 查看模型参数情况
# model.summary()

# random categorical:
# 从理论上来讲，如果模型足够准确，我们只需要从概率分布中选择概率最大的值的索引即可，这就是贪心算法。
# 但在实际中，模型的预测效果很难确定，一直按照最大概率选取很容易陷入重复的循环中，
# 因此会将分布的概率值作为其被选中的概率值，这样每个分布中的值都有可能被选中，
# tensorflow中使用tf.random.categorical方法来实现.
# 使用random categorical
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
# squeeze表示消减一个维度
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
print(sampled_indices)
# 输出效果:
# array([63, 38,  7,  1, 59, 11, 25, 46, 36, 46, 59, 43, 52, 21, 48, 41, 39,
#        53, 59,  6, 63, 13,  1, 39,  3, 18, 30, 23, 29, 29, 38, 37, 42, 10,
#        37,  7, 63, 25, 37, 55, 62, 54,  8, 13, 25, 56, 50, 64, 48, 62, 34,
#        33, 25, 48, 39, 38,  3, 16, 25, 37, 31, 19,  1, 21, 30, 18,  2,  6,
#         0, 55, 56, 13,  5, 63, 44, 27, 12, 34, 54, 30, 38, 36, 24, 43, 62,
#        61, 23, 14, 43, 19, 30, 58,  6, 21, 56,  6, 54, 48,  2, 54])

# 也将输入映射成文本内容
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
# 映射这些索引查看对应的文本
# 在没有训练之前，生成的文本没有任何规律
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))

# 输出效果:
# Input:
#  "ght me craft\nTo counterfeit oppression of such grief\nThat words seem'd buried in my sorrow's grave.\n"
# Next Char Predictions:
#  "yZ- u;MhXhuenIjcaou,yA a$FRKQQZYd:Y-yMYqxp.AMrlzjxVUMjaZ$DMYSG IRF!,\nqrA'yfO?VpRZXLexwKBeGRt,Ir,pj!p"

# 添加损失函数
# 此时可以将生成问题看作是标准的分类问题，即给定RNN的状态和该时间步的输入，
# 预测下一个字符的类别（从分布中只选择一个），类别总数即不重复的字符总数，因此这是一个稀疏类别矩阵.
# 使用keras预置的稀疏类别交叉熵损失（当类别矩阵为稀疏类别矩阵时使用该损失）
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
# 使用损失函数
example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())
# Prediction shape:  (64, 100, 65)  # (batch_size, sequence_length, vocab_size)
# scalar_loss:       4.175118
# 添加优化器
# 配置优化器为'adam'
model.compile(optimizer='adam', loss=loss)
# 配置检测点
# 检查点保存至的目录
checkpoint_dir = './training_checkpoints'
# 检查点的文件名
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
# 创建检测点保存的回调对象
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# 模型训练并打印日志
EPOCHS=10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
# 输出效果：
# Epoch 1/10
# 172/172 [==============================] - 8s 49ms/step - loss: 2.6956
# Epoch 2/10
# 172/172 [==============================] - 7s 39ms/step - loss: 1.9793
# Epoch 3/10
# 172/172 [==============================] - 7s 38ms/step - loss: 1.7066
# Epoch 4/10
# 172/172 [==============================] - 6s 37ms/step - loss: 1.5544
# Epoch 5/10
# 172/172 [==============================] - 7s 38ms/step - loss: 1.4630
# Epoch 6/10
# 172/172 [==============================] - 6s 38ms/step - loss: 1.4019
# Epoch 7/10
# 172/172 [==============================] - 6s 37ms/step - loss: 1.3562
# Epoch 8/10
# 172/172 [==============================] - 7s 38ms/step - loss: 1.3177
# Epoch 9/10
# 172/172 [==============================] - 6s 37ms/step - loss: 1.2836
# Epoch 10/10
# 172/172 [==============================] - 6s 37ms/step - loss: 1.2513

# 第三步: 使用模型生成文本内容
# 恢复模型结构
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
# 从检测点中获得训练后的模型参数
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

# 构建生成函数：
def generate_text(model, start_string):
    """
    :param model: 训练后的模型
    :param start_string: 任意起始字符串
    """
    # 要生成的字符个数
    num_generate = 1000

    # 将起始字符串转换为数字（向量化）
    input_eval = [char2idx[s] for s in start_string]

    # 扩展维度满足模型输入要求
    input_eval = tf.expand_dims(input_eval, 0)

    # 空列表用于存储结果
    text_generated = []

    # 设定“温度参数”，根据tf.random_categorical方法特点，
    # 温度参数能够调节该方法的输入分布中概率的差距，以便控制随机被选中的概率大小
    temperature = 1.0

    # 初始化模型参数
    model.reset_states()

    # 开始循环生成
    for i in range(num_generate):
        # 使用模型获得输出
        predictions = model(input_eval)
        # 删除批次的维度
        predictions = tf.squeeze(predictions, 0)

        # 使用“温度参数”和tf.random.categorical方法生成最终的预测字符索引
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # 将预测的输出再扩展维度作为下一次的模型输入
        input_eval = tf.expand_dims([predicted_id], 0)

        # 将该次输出映射成字符存到列表中
        text_generated.append(idx2char[predicted_id])

    # 最后将初始字符串和生成的字符进行连接
    return (start_string + ''.join(text_generated))

# 调用：
print(generate_text(model, start_string=u"ROMEO: "))
# 输出效果：
# ROMEO: it may be see, I say.
# Elong where I have sea loved for such heart
# As of all desperate in your colls?
# On how much to purwed esumptrues as we,
# But taker appearing our great Isabel,;
# Of your brother's needs.
# I cannot but one hour, by nimwo and ribs
# After 't? O Pedur, break our manory,
# The shadot bestering eyes write; onfility;
# Indeed I am possips And feated with others and throw it?
#
# CAPULET:
# O, not the ut with mine own sort.
# But, with your souls, sir, well we would he,
# And videwith the sungesoy begins, revell;
# Much it in secart.
#
# PROSPERO:
# Villain, I darry record;
# In sea--lodies, nor that I do I were stir,
# You appointed with that sed their o tailor and hope left fear'd,
# I so; that your looks stand up,
# Comes I truly see this last weok not the
# sul us.
#
# CAMILLO:
# You did and ever sea,
# Into these hours: awake! Ro with mine enemies,
# Were werx'd in everlawacted man been to alter
# As Lewis could smile to his.
#
# Farthus:
# Marry! I'll do lose a man see me
# To no drinking often hat back on an illing mo

# 更高级的方式！！！
# 构建训练模型与函数
# 构建模型
model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

# 选择优化器
optimizer = tf.keras.optimizers.Adam()


# 编写带有装饰器@tf.function的函数进行训练
@tf.function
def train_step(inp, target):
    """
    :param inp: 模型输入
    :param tatget: 输入对应的标签
    """
    # 打开梯度记录管理器
    with tf.GradientTape() as tape:
        # 使用模型进行预测
        predictions = model(inp)
        # 使用sparse_categorical_crossentropy计算平均损失
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True))
    # 使用梯度记录管理器求解全部参数的梯度
    grads = tape.gradient(loss, model.trainable_variables)
    # 使用梯度和优化器更新参数
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # 返回平均损失
    return loss

# 进行训练
# 训练轮数
EPOCHS = 10

#进行轮数循环
for epoch in range(EPOCHS):
    # 获得开始时间
    start = time.time()
    # 初始化隐层状态
    hidden = model.reset_states()
    # 进行批次循环
    for (batch_n, (inp, target)) in enumerate(dataset):
        # 调用train_step进行训练, 获得批次循环的损失
        loss = train_step(inp, target)
        # 每100个批次打印轮数，批次和对应的损失
        if batch_n % 100 == 0:
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch+1, batch_n, loss))

    # 每5轮保存一次检测点
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    # 打印轮数，当前损失，和训练耗时
    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# 保存最后的检测点
model.save_weights(checkpoint_prefix.format(epoch=epoch))



