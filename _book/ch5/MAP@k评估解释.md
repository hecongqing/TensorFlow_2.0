# AP

如果要求我们推荐N个项目，项目的全空间中相关项目的数量为 `m`，

则:
$$
\textrm{AP@N} = \frac{1}{m}\sum_{k=1}^N \textrm{($P(k)$ if $k^{th}$ item was relevant)} = \frac{1}{m}\sum_{k=1}^N P(k)\cdot rel(k)
$$
其中$rel(k)$只是一个指标，表示$k^{th}$是否相关，如果相关，则$rel(k)=1$；否则$rel(k)=0$。



# 什么是MAP

上面所述的就是**AP**（Average Precision），它适用于单个数据点（例如单个用户）。 那么什么是 $MAP@N$ 呢？剩下的只是对所有$| U |$中的$AP@N$指标求平均值。
$$
\textrm{MAP@N} = \frac{1}{|U|}\sum_{u=1}^{|U|}(\textrm{AP@N})_u = \frac{1}{|U|} \sum_{u=1}^{|U|} \frac{1}{m}\sum_{k=1}^N P_u(k)\cdot rel_u(k).
$$


# AP公式的常见变化

通常，当可能有更多的正确建议而不是要求您提供的建议数量时，你会看到AP指标略有修改。假设银行的超级活跃用户下个月添加了$m = 10$个帐户，而算法仅应报告为$N = 5$。在这种情况下，使用的归一化因子为$\frac{1}{min(m，N)}$，当推荐数量无法捕获所有正确的建议时，可以防止AP分数受到不公平的压制。
$$
\textrm{AP@N} = \frac{1}{\textrm{min}(m,N)}\sum_{k=1}^N P(k)\cdot rel(k).
$$


也可能会遇到一些更草率的用法，其中$AP @ N$之和中没有指示符函数$rel(k)$。在这种情况下，当第$k^{th}$个推荐不正确时，将截止$k$处的精度隐式定义为零，这样它仍然不会对总和做出贡献：
$$

\textrm{AP@N} = \frac{1}{m}\sum_{k=1}^N P(k),\\
P(k) = 0 \textrm{ if $k^{th}$ element is irrelevant / incorrect.}
$$


最后，如果可能没有相关或正确的建议（m = 0），那么对于这些点，通常将AP定义为零。请注意，这将使算法的MAP编号向下拖动，使实际上没有添加任何产品的用户越多。比较相同数据集上两种算法的性能并不重要，但这确实意味着您不应在最终数字上加上任何绝对含义。
$$
\textrm{AP@N} = \frac{1}{\textrm{min}(m,N)}\sum_{k=1}^N P(k)\cdot rel(k) \qquad \textrm{ if $m\neq 0$,}\\
AP = 0 \qquad \textrm{if $m=0$}.
$$


```python
import numpy as np

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])



actual = [1]

predicted = [1,2,3,4,5]

print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',apk(actual,predicted,5) )

predicted = [2,1,3,4,5]
print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',apk(actual,predicted,5) )

predicted = [3,2,1,4,5]
print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',apk(actual,predicted,5) )

predicted = [4,2,3,1,5]
print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',apk(actual,predicted,5) )

predicted = [4,2,3,5,1]
print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',apk(actual,predicted,5) )
```



```reStructuredText
Answer= [1] predicted= [1, 2, 3, 4, 5]
AP@5 = 1.0
Answer= [1] predicted= [2, 1, 3, 4, 5]
AP@5 = 0.5
Answer= [1] predicted= [3, 2, 1, 4, 5]
AP@5 = 0.3333333333333333
Answer= [1] predicted= [4, 2, 3, 1, 5]
AP@5 = 0.25
Answer= [1] predicted= [4, 2, 3, 5, 1]
AP@5 = 0.2
```

