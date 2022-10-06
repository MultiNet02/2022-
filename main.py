# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data = pd.read_csv("C:/Users/王剑/Desktop/附件三.csv")
data.head()

# 查看数据信息
data.info()

data.describe(include='all')

from sklearn.preprocessing import LabelEncoder
# 数字编码
le = LabelEncoder()

# 数字编码
data["表面风化"] = le.fit_transform(data["表面风化"][:])

data

from sklearn.impute import SimpleImputer

# 采用均值填补法
imp = SimpleImputer(strategy="mean")
data["氧化铁"] = imp.fit_transform(data["氧化铁"].to_frame())

data.describe()

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义weathers列
weathers = ['风化', '无风化']
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 5))


# 画出风化柱状图
sns.countplot(x='表面风化', data=data, palette="Set2", ax=ax2)


ax2.set(xlabel='是否风化', ylabel='数量')
ax2.set_xticklabels(weathers)

from sklearn.feature_selection import f_classif


fdata = pd.DataFrame(data.drop(['文物编号', '表面风化'], axis=1))# 删除gender列与label_cal列
label_cal = pd.DataFrame(data['表面风化'])

# 使用f_classif计算label_cal与其他连续型变量间的关系
F, p_val = f_classif(fdata, label_cal)
# f分布的0.05分位数
print('各连续型变量的名称：')
print(fdata.columns.tolist())
print('各连续型变量与是否得病之间的F值为：')
print(F)
print('各连续型变量与是否得病之间的pvalue为：')
print(p_val)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
x = data.drop(['文物编号','表面风化'], axis=1) # x为删除label_cal的数据集
y = data['表面风化'] # y为label_cal列
#归一化
from sklearn.preprocessing import StandardScaler
stdScaler=StandardScaler()
x1=stdScaler.fit_transform(x)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=2, stratify=data['表面风化'])

clf=LogisticRegression(random_state=0,solver='lbfgs',multi_class='multinomial').fit(x_train,y_train)
clf.score(x_test,y_test)



                                        # 为权重赋值
                                        weights = {0: 1, 1: 1.3}

                                        # 进行logistic回归
                                        lr = LogisticRegression(penalty='l2', random_state=8, class_weight=weights)
                                        lr.fit(x_train, y_train)

# 对y进行预测
y_predprb = clf.predict_proba(x_test)[:, 1]
y_pred = clf.predict(x_test)

from sklearn import metrics
from sklearn.metrics import auc

# 计算fpr，tpr及thresholds的值
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predprb)
# 计算gmean的值
gmean = np.sqrt(tpr*(1-fpr))

# 计算最大的gmean值对应的thresholds值
dictionary = dict(zip(thresholds,gmean))
max_thresholds = max(dictionary, key=dictionary.get)

print("最大的GMean值为：%.4f"%(max(gmean)))
print("最大的GMean对应的thresholds为：%.4f"%(max_thresholds))

from sklearn.metrics import roc_auc_score
# 计算AUC值
test_roc_auc = roc_auc_score(y_test, y_predprb)
print(test_roc_auc)

# 打印模型分类预测报告
print(classification_report(y_test, y_pred))

# 画出混淆矩阵热力图
cm1 = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm1, annot=True, linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'ROC AUC Score: {0}'.format(round(test_roc_auc,2))
plt.title(all_sample_title, size=15)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#归一化
from sklearn.preprocessing import StandardScaler
stdScaler=StandardScaler()
x1=stdScaler.fit_transform(x)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=2, stratify=data['表面风化'])

clf1=KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)
clf.score(x_test,y_test)

y_predprb = clf1.predict_proba(x_test)[:, 1]
y_predict = clf1.predict(x_test)

# 计算AUC值
test_roc_auc = roc_auc_score(y_test, y_predprb)
print(test_roc_auc)

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

#归一化
from sklearn.preprocessing import StandardScaler
stdScaler=StandardScaler()
x1=stdScaler.fit_transform(x)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=2, stratify=data['表面风化'])

clf2=GaussianNB().fit(x_train,y_train)
clf2.score(x_test,y_test)

y_predprb = clf2.predict_proba(x_test)[:, 1]
y_predict = clf2.predict(x_test)

# 计算AUC值
test_roc_auc = roc_auc_score(y_test, y_predprb)
print(test_roc_auc)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
weights = {0: 1, 1: 1.3}

# 建立决策树模型
model = DecisionTreeClassifier(random_state=5, class_weight=weights)
model = model.fit(x_train, y_train)

# 对y进行预测
y_predict = model.predict(x_test)
y_predprb = model.predict_proba(x_test)[:, 1]

# 计算AUC值
test_roc_auc = roc_auc_score(y_test, y_predprb)
print(test_roc_auc)

# 打印模型分类预测报告
print(classification_report(y_test, y_predict))

# 绘制混淆矩阵热力图
cm2 = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(9, 9))
sns.heatmap(cm2, annot=True, linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'ROC AUC score: {0}'.format(round(test_roc_auc,2))
plt.title(all_sample_title, size=15)

#使用RMSE和MSE作为模型性能的评价指标

from sklearn.metrics import make_scorer
def rmse(y_true,y_pred):
    diff=y_pred-y_true
    sum_sq=sum(diff**2)
    n=len(y_pred)

    return np.sqrt(sum_sq/n)
def mse(y_true,y_pred):
    return mean_squared_error(y_true,y_pred)

rmse_scorer=make_scorer(rmse,greater_is_better=False)
mse_scorer=make_scorer(mse,greater_is_better=False)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 建立随机森林模型
ran_for = RandomForestClassifier(n_estimators=80, random_state=0, class_weight=weights)
ran_for.fit(x_train, y_train)

# 对y进行预测
y_pred_ran = ran_for.predict(x_test)
y_predprb = ran_for.predict_proba(x_test)[:, 1]

# 计算AUC值
test_roc_auc = roc_auc_score(y_test, y_predprb)
print(test_roc_auc)

# 打印模型分类预测报告
print(classification_report(y_test, y_pred_ran, digits=2))

# 绘制混淆矩阵热力图
cm3 = confusion_matrix(y_test, y_pred_ran)
plt.figure(figsize=(9, 9))
sns.heatmap(cm3, annot=True, linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'ROC AUC score: {0}'.format(round(test_roc_auc,2))
plt.title(all_sample_title, size=15)

np.random.seed(123)
perm = np.random.permutation(len(X)) # 将数组随机生成一个新的序列
X = X.loc[perm]
y = y[perm]
X = preprocessing.scale(X)# 进行标准化处理

from sklearn.decomposition import PCA
# 使用PCA进行降维
pca = PCA(copy=True, n_components=6, whiten=False, random_state=1)
X_new = pca.fit_transform(X)

print(u'所保留的6个主成分的方差贡献率为：')
print(pca.explained_variance_ratio_)
print(u'排名前2的主成分特征向量为：')
print(pca.components_[0:1])
print(u'累计方差贡献率为：')
print(sum(pca.explained_variance_ratio_))

# 对数据集进行划分
x_train, x_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=2, stratify=data['表面风化'])

# 建立随机森林模型
ran_for = RandomForestClassifier(n_estimators=80, random_state=0, class_weight=weights)
# 训练模型
ran_for.fit(x_train, y_train)

# 对y进行预测
y_pred_ran = ran_for.predict(x_test)
y_predprb = ran_for.predict_proba(x_test)[:, 1]

# 计算AUC值
test_roc_auc = roc_auc_score(y_test, y_predprb)
print(test_roc_auc)

# 打印模型分类预测报告
print(classification_report(y_test, y_pred_ran, digits=2))

# 绘制混淆矩阵热力图
cm4 = confusion_matrix(y_test, y_pred_ran)
plt.figure(figsize=(9, 9))
sns.heatmap(cm4, annot=True, linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'ROC AUC score: {0}'.format(round(test_roc_auc,2))
plt.title(all_sample_title, size=15)

from sklearn.ensemble import BaggingClassifier
from sklearn neighbors import KNeighborsClassifier
clf4=BaggingClassifier(KNeighborsClassifier(),max_samples=0.5,max_features=0.5)
clf4=clf4.fit(x_train,y_train)
clf4.score(x_test,y_test)

model='GradientBoosting'

from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import learning_curve
plt.figure(figsize=(18,10),dpi=150)
def plot_learning_curve(estimator,title,x,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(.1,1.0,5):
    plt.figure()
    plt.title(title)
    if ylimis not None:
       plt.ylim(*ylim)
    plt.xlabel("Training Samples")
    plt.ylabel("Score")
    train_sizes,train_scores,test_scores=leaning_curve(estimator,x,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)
    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)

