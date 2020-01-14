import tensorflow as tf
from sys_config.sysconfig import tf_v


# 1.14.0版本的tensorflow
if tf_v == "1.14.0":
    LSTM_model = tf.nn.rnn_cell.LSTMCell()
    """
    num_units. lstm 模块中隐含层节点数大小
    use_peepholes =False  bool型
    cell_clip = None   可选项， float  
    initializer =None  可选项， 初始化权重
    num_proj = None  int型
    num_unit_shards =None   
    num_proj_shards =None
    forget_bias=1.0    浮点型数据  遗忘门权重，默认设为1，  重载的时候，必须设置为0
    state_is_tuple =True 如果为真，则接受和返回的状态是C_state和m_state的2元组。如果为false，则它们沿着列轴连接。后一种行为将很快被否决
    activation =None  激活函数， 默认为tanh函数
    reuse =None  是否变量服用
    name =None   层名
    dtype =None  使用 输入数据类型 默认为None
    **kwargs
    """

    LSTM_cell = tf.nn.rnn_cell.BasicLSTMCell() # lstm cell
    """
    num_units.  lstm  lstm 中隐含层节点数大小
    forget_bias =1.0
    state_is_tupe = True
    activation =None
    reuse = None
    name =None
    dtype = None
    **kwargs
    
    """

else:
    LSTM_model = tf.compat.v2.nn.rnn_cell.LSTMCelll()

    LSTM_cell = tf.compat.v2.nn.rnn_cell.BasicLSTMCell()