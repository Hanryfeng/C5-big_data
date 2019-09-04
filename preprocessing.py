# 步骤1  数据读取，导入数据，查看数据头信息
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno
loan_data = pd.read_csv('E:\Big_data\loan.csv')
print(loan_data.head())

# 步骤2  查看缺失比例大于20%的属性
check_null = loan_data.isnull().sum().sort_values(ascending=False)/float(len(loan_data))
print(check_null[check_null > 0.2])

# 步骤3  删除缺失量超过阈值的缺失项
thresh_count = len(loan_data)*0.9  # 设定阈值
loan_data = loan_data.dropna(thresh=thresh_count, axis=1)  # 若某一列数据缺失的数量超过阈值就会被删除
print(loan_data.shape)

# 步骤4  将处理后的数据储存到loan2007_2015.csv文件中
loan_data.to_csv('E:\Big_data\loan2007_2015.csv', index=False)

# 步骤5  读取处理后的数据，打印分类统计数据类型
loans = pd.read_csv('E:\Big_data\loan2007_2015.csv')
print(loans.dtypes.value_counts())
print(loans.shape)

# 步骤6  同值化处理（将1列全是同一个值的列删除）
loans = loans.loc[:, loans.apply(pd.Series.nunique) != 1]  # loc表示多列选择
# 变量大部分的观测都是相同的特征，那么这个特征或者输入变量就是无法用来区分目标属性
print(loans.shape)

# 步骤7  缺失值处理——分类变量（看看object类型列的空值情况）
# 获取所有的类别特征，如url属性等
objectColumns = loans.select_dtypes(include=["object"]).columns
print(loans[objectColumns].isnull().sum().sort_values(ascending=False))
print(loans[objectColumns])

# 步骤8  调用missingno库来快速评估数据缺失的情况，由于部分数据读取为字符串类型，因此需要进行一些数据转换
loans['int_rate'] = loans['int_rate'].astype(str).astype('float')  # 类型转换
loans['revol_util'] = loans['revol_util'].astype(str).astype('float')  # 类型转换
objectColumns = loans.select_dtypes(include=["object"]).columns  # 选取对象类型数据
msno.matrix(loans[objectColumns])  # 缺失值可视化
plt.show()
# 从图中可以直观的看出变量 "last_pymnt_d" "emp_title" "emp_length" 缺失值较多。
# 这里先用'unknown'来填充。
objectColumns = loans.select_dtypes(include=["object"]).columns
loans[objectColumns] = loans[objectColumns].fillna("Unknown")

# 步骤9  缺失值处理——数值变量
numColumns = loans.select_dtypes(include=[np.number]).columns  # 选取数据变量的列
pd.set_option('display.max_columns', len(numColumns))
print(loans[numColumns].tail())
# 对于数值型的属性，可采用sklearn的Preprocessing模块，参数strategy选用mean，采用均值插补的方法填充缺失值。
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

# verbose = 0
imr = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
# imr = SimpleImputer(missing_values='NaN', strategy='mean', verbose=0) 
imr = imr.fit(loans[numColumns])
loans[numColumns] = imr.transform(loans[numColumns])
msno.matrix(loans[numColumns])  # 缺失值可视化
plt.show()
# 结果显示没有缺失值了

# 步骤10  分类数据过滤，过滤掉重复或者对预测模型没有实际意义的属性
# 设置过滤属性列
drop_list = ['term', 'sub_grade', 'title', 'zip_code', 'addr_state', 'earliest_cr_line',
             'url', 'last_pymnt_d', 'last_credit_pull_d', 'issue_d', 'emp_title']
loans.drop(drop_list, axis=1, inplace=True)  # 通过过滤属性列删除属性
print(loans.select_dtypes(include=["object"]).shape)  # 输出过滤完成后的数据集维度

# 步骤11 特征衍生
# 新特征'installment_feat'代表客户每月还款支出占月收入的比，'installment_feat'的值越大，意味着贷款人的偿债压力越大，违约的可能性越大
loans['installment_feat'] = loans['installment']/((loans['annual_inc']+1)/12)

# 步骤12  特征抽象，主要是对数据进行编码处理，将Y的标记值转化成实际值0或1标签值


def coding(col, codeDict):
    colCoded = pd.Series(col, copy=True)
    for key, value in codeDict.items():
        colCoded.replace(key, value, inplace=True)
    return colCoded

#!!!def可能有问题.....return忘退格了>_<


# 把贷款状态LoanStatus 编码为违约=1 ， 正常=0：
#loans["loan_status"] = coding(loans["loan_status"], {'Current': 0, 'Issued': 0, 'Fully Paid': 0, 'In Grace Period': 1, 'Late (31-120 days)': 1, 'Late (16-30 days)': 1,
#                                                     'Charged Off': 1, 'Does not meet the credit policy. Status:Charged Off': 1, 'Does not meet the credit policy. Status:Fully Paid': 0, 'Default': 0})  # 打印状态信息
loans["loan_status"] = coding(loans["loan_status"], {'Current': 0, 'Issued': 0, 'Fully Paid': 0, 'In Grace Period': 1, 'Late (31-120 days)': 1, 'Late (16-30 days)': 1, 'Charged Off': 1,
                                                     'Does not meet the credit policy. Status:Charged Off': 1, 'Does not meet the credit policy. Status:Fully Paid': 0, 'Default': 0})

print('\nAfter Coding:')
# 打印状态信息
print(pd.value_counts(loans["loan_status"]))

print(loans.select_dtypes(include=["object"]).head())

#步骤13  部分数值特征抽象化
#有序特征的映射，对字符串属性数据进行编码处理，转换成数值型数据参与后面的计算
mapping_dict={
    "emp_length":{
        "10+ years":10,
        "9 years":9,
        "8 years":8,
        "7 years":7,
        "6 years":6,
        "5 years":5,
        "4 years":4,
        "3 years":3,
        "2 years":2,
        "1 year":1,
        "< 1 year":0,
        "Unknown":0,
        
    },
    "grade":{
        "A":1,
        "B":2,
        "C":3,
        "D":4,
        "E":5,
        "F":6,
        "G":7,
            
    }
}
loans = loans.replace(mapping_dict)
print(loans[['emp_length','grade']].head())

#步骤14 剩余特征值one-hot编码
n_columns=["home_ownership","verification_status","purpose","application_type","initial_list_status","pymnt_plan"]
dummy_df=pd.get_dummies(loans[n_columns]) #用get_dummies进行one hot编码
loans=pd.concat([loans,dummy_df],axis=1)
#当axis=1的时候，concat就是行对齐，然后将不同列名称的两张表合并

#步骤15 清除原有属性
loans=loans.drop(n_columns,axis=1)
col=loans.select_dtypes(include=['int64','float64']).columns
col=col.drop('loan_status')
col=col.drop('id')
loans_ml_df=loans

#步骤16 特征缩放
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#loans_ml_df[col]=sc.fit_transform(loans_ml_df[col])

# 步骤17 特征选择 (Feature Selecting)
# 目的：首先，优先选择与目标相关性较高的特征；其次，去除不相关特征可以降低学习的难度构建X特征变量和Y目标变量，这里主要是对数据进行封装
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import pca
print("***********", loans_ml_df.columns)
x_feature = list(loans_ml_df.columns)
x_feature.remove('loan_status')
x_val = loans_ml_df[x_feature]
y_val = loans_ml_df['loan_status']
#!!!文件输出地址改一下
loans_ml_df.to_csv('E:\Big_data\loan2007_2015_test.csv', index=False)
print(x_val.head())
print(len(x_feature))  # 查看初始特征集合的数量

# 首先，选出与目标变量相关性较高的特征。这里采用的是Wrapper方法，通过暴力的递归特征消除
# (Recursive
# Feature Elimination)方法筛选30个与目标变量相关性强的特征，逐步剔除特征从而达到首次降维，自变量从63个降到30个。
from sklearn.linear_model.logistic import LogisticRegression
model = LogisticRegression()
from sklearn.feature_selection import RFE  # 导入特征选择库
rfe = RFE(model, 30) #通过递归选择特征，选择30个特征
print(x_val.info())
rfe =rfe.fit(x_val,y_val)
#打印筛选结果
print("rfe.n_features_")
print(rfe.n_features_)
print("rfe.estimator_")
print(rfe.estimator_)
print("rfe.support_")
print(rfe.support_)
print("rfe.ranking_")
print(rfe.ranking_) #ranking 为1代表被选中，其他则未被代表未被选中

#通过布尔值筛选首次降维后的变量
col_filter = x_val.columns[rfe.support_]
print(col_filter)

#在第一次降维的基础上，通过皮尔森相关性图谱找出冗余特征并将其剔除；
#同时，可以通过相关性图谱进一步引导特征选择的方向
colormap = plt.cm.viridis
import seaborn as sns
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features',y=1.05,size=15)
sns.heatmap(loans_ml_df[col_filter].corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,linecolor='white',annot=True)
plt.show()

# 剔除冗余特征
#drop_col=['funded_amnt','funded_amnt_inv','out_prncp_inv','total_pymnt_inv','total_rec_prncp','total_rec_int',
#           'verification_status_Not Verified','verification_status_Source Verified','collection_recovery_fee','verification_status_Verified']
drop_col=['funded_amnt','funded_amnt_inv','out_prncp_inv','total_pymnt_inv','total_rec_prncp','total_rec_int',
           'verification_status_Not Verified','collection_recovery_fee']
col_new= col_filter.drop(drop_col)
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features',y=1.05,size=15)
sns.heatmap(loans_ml_df[col_new].corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,linecolor='white',annot=True)
print(len(col_new))
plt.show()

# 下面需要对特征的权重有一个正确的判断和排序，可以通过特征重要性排序来挖掘那些变量是比较重要的，
# 降低学习难度，最终达到优化模型计算的目的。这里，采用随机森林算法判定特征的重要性，工程实现方式
# 采用scikit-learn的featureimportances的方法
from sklearn.ensemble import RandomForestClassifier
names =loans_ml_df[col_new].columns
clf = RandomForestClassifier(n_estimators=10,random_state=123)#构建分类随机森林分类器
clf.fit(x_val[col_new],y_val)#对自变量进行拟合
for feature in zip(names, clf.feature_importances_):
    print(feature)
plt.style.use('ggplot')
##feature importance 可视化##
importances = clf.feature_importances_
feat_names = names 
indices = np.argsort(importances)[::-1]

fig = plt.figure(figsize=(20,6))
plt.title("Feature importances by RandomForestClassfier")
plt.bar(range(len(indices)),importances[indices],color='lightblue',align='center')
plt.step(range(len(indices)),np.cumsum(importances[indices]),where = 'mid',label = 'Cumulative')
plt.xticks(range(len(indices)),feat_names[indices],rotation='vertical',fontsize=14)
plt.xlim([-1,len(indices)])
plt.show()


#步骤18  不平衡数据的处理（过抽样，样本生成，随机森林可以设置样本权重）
# 原数据拆分，70%用于训练，30%用于测试
from sklearn.model_selection import train_test_split
def data_prepration(x):
    x_features=x.ix[:,x.columns != "loan_status"]#采用熟悉特征作为记录属性特征
    x_labels=x.ix[:,x.columns == "loan_status"]
    #调用train_test_split拆分数据
    x_features_train,x_features_test,x_labels_train,x_labels_test=train_test_split(x_features,x_labels,test_size=0.3)
    return (x_features_train,x_features_test,x_labels_train,x_labels_test)

X=loans_ml_df[col_new]
y=loans_ml_df['loan_status']

df=loans_ml_df
data_train_X,data_test_X,data_train_y,data_test_y = data_prepration(df)
print(pd.value_counts(data_test_y['loan_status']))
print(pd.value_counts(data_train_y['loan_status']))

#调用SMOTE算法进行倾斜数据的平衡化处理
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
os_data_X,os_data_y = os.fit_sample(data_train_X.values, data_train_y.values.ravel())
columns = data_train_X.columns
os_data_X = pd.DataFrame(data=os_data_X,columns=columns)
os_data_y = pd.DataFrame(data=os_data_y,columns=["loan_status"])

newtraindata=pd.concat([os_data_X,os_data_y],axis=1)
print(newtraindata.head())

os_test_X,os_test_y=os.fit_sample(data_test_X.values,data_test_y.values.ravel())
columns = data_test_X.columns
print(columns)
os_test_X = pd.DataFrame(data=os_test_X,columns=columns)
os_test_y = pd.DataFrame(data=os_test_y,columns=["loan_status"])

newtraindata.set_index('id',inplace=True)
newtestdata=pd.concat([os_test_X,os_test_y],axis=1)
newtestdata.set_index('id',inplace=True)

#将处理后的数据保存到指定目录下
newtraindata.to_csv('E:\Big_data\\train_unnorm.csv',sep=',')
newtestdata.to_csv('E:\Big_data\\test_unnorm.csv',sep=',')
print("结束")
