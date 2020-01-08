# -*- coding:UTF-8 -*-
# =========================================================================
import tensorflow as tf

# =========================================================================
# 网络结构定义
# 输入参数：images，image batch、4D tensor、tf.float32、[batch_size, width, height, channels]
# 返回参数：logits, float、 [batch_size, n_classes]
def inference(images, batch_size, n_classes):
    # 一个简单的卷积神经网络，卷积+池化层x2，全连接层x2，最后一个softmax层做分类。
    # 卷积层1
    # 64个3x3的卷积核（3通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    parameters = []
    # 第1个卷积层
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    # 添加LRN层和最大池化层
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')
    print_activations(pool1)

    # 设计第2个卷积层
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

    # 对第2个卷积层的输出进行处理，同样也是先做LRN处理再做最大化池处理。
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
    print_activations(pool2)

    # 设计第3个卷积层
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    # 设计第4个卷积层
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    # 设计第5个卷积层
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

    # 最大池化层
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
    print_activations(pool5)

    #     return pool5, parameters
    # 全连接层3
    # 128个神经元，将之前pool层的输出reshape成一行，激活函数relu()
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(tf.truncated_normal(shape=[dim, 4096], stddev=0.005, dtype=tf.float32),
                              name='weights', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[4096]),
                             name='biases', dtype=tf.float32)

        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # 全连接层4
    # 128个神经元，激活函数relu()
    with tf.variable_scope('local4') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[4096, 4096], stddev=0.005, dtype=tf.float32),
                              name='weights', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[4096]),
                             name='biases', dtype=tf.float32)

        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # dropout层
    #    with tf.variable_scope('dropout') as scope:
    #        drop_out = tf.nn.dropout(local4, 0.8)

    # Softmax回归层
    # 将前面的FC层输出，做一个线性回归，计算出每一类的得分，在这里是2类，所以这个层输出的是两个得分。
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[4096, n_classes], stddev=0.005, dtype=tf.float32),
                              name='softmax_linear', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[n_classes]),
                             name='biases', dtype=tf.float32)

        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear


# -----------------------------------------------------------------------------
# loss计算
# 传入参数：logits，网络计算输出值。labels，真实值，在这里是0或者1
# 返回参数：loss，损失值
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


# --------------------------------------------------------------------------
# loss损失值优化
# 输入参数：loss。learning_rate，学习速率。
# 返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# -----------------------------------------------------------------------
# 评价/准确率计算
# 输入参数：logits，网络计算值。labels，标签，也就是真实值，在这里是0或者1。
# 返回参数：accuracy，当前step的平均准确率，也就是在这些batch中多少张图片被正确分类了。
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

# ========================================================================

