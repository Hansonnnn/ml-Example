### apply函数

apply函数是pandas里面所有函数中自由度最高的函数。该函数如下：

DataFrame.apply(func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds)

该函数最有用的是第一个参数，这个参数是函数，相当于C/C++的函数指针。

这个函数需要自己实现，函数的传入参数根据axis来定，比如axis = 1，就会把一行数据作为Series的数据
结构传入给自己实现的函数中，我们在函数中实现对Series不同属性之间的计算，返回一个结果，则apply函数
会自动遍历每一行DataFrame的数据，最后将所有结果组合成一个Series数据结构并返回。

比如以下使用方式：
```
df['num_punctuations'] = df['text'].apply(lambda x: len([word for word in str(x) if word in string.punctuation]))
```
以上的使用方式是对DataFrame的数据格式中的一列做操作，在第一个参数中传入lamda函数作为参数。

### K折交叉验证

```
sklearn.model_selection.KFold(n_splits=3, shuffle=False, random_state=None)
```
思路：将训练/测试数据集划分n_splits个互斥子集，每次用其中一个子集当作验证集，剩下的n_splits-1个作为训练集，进行n_splits次训练和测试，得到n_splits个结果

注意点：对于不能均等份的数据集，其前n_samples % n_splits子集拥有n_samples // n_splits + 1个样本，其余子集都只有n_samples // n_splits样本

参数说明：
n_splits：表示划分几等份
shuffle：在每次划分时，是否进行洗牌

①若为Falses时，其效果等同于random_state等于整数，每次划分的结果相同

②若为True时，每次划分的结果都不一样，表示经过洗牌，随机取样的

random_state：随机种子数

属性：
```
①get_n_splits(X=None, y=None, groups=None)：获取参数n_splits的值
```
```
②split(X, y=None, groups=None)：将数据集划分成训练集和测试集，返回索引生成器
```
### MultinomialNB(多项式模型)

![image](https://note.youdao.com/yws/public/resource/e583908e8c48d653389a24ccb5ddb58b/xmlnote/706CCF03943348A183CB04595065CB88/4817)

### 伯努利模型
基于上述例子：

![image](https://note.youdao.com/yws/public/resource/e583908e8c48d653389a24ccb5ddb58b/xmlnote/0BA966ACBC77487A9DD868383237B104/4822)
