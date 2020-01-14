# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:24:38 2019

@author: Administrator
"""

import featuretools as ft
import pandas as pd
pd.set_option('display.max_columns',None)

"""
本demo 中有三张数据表， 表即为实体entity
1.customers: 由不同客户记录组陈， 一个客户可以有多个session
2.sessions: 由不同的session记录组成，一个session 记录包括多个属性
3.transaction: 由不同交易记录组成，一个session包含多个交易时间
"""

data =ft.demo.load_mock_customer()  # 导入mock 数据
es =ft.demo.load_mock_customer(return_entityset=True)
print(es)  # 显示实体集
"""
Entityset: transactions
  Entities:
    transactions [Rows: 500, Columns: 5]
    products [Rows: 5, Columns: 2]
    sessions [Rows: 35, Columns: 4]
    customers [Rows: 5, Columns: 4]
  Relationships:
    transactions.product_id -> products.product_id
    transactions.session_id -> sessions.session_id
    sessions.customer_id -> customers.customer_id
"""
customers_df=data["customers"]
print(type(customers_df))  #df 格式数据
#print(data)
print(customers_df.head(2))
"""
   customer_id zip_code           join_date date_of_birth
0            1    60091 2011-04-17 10:48:33    1994-07-18
1            2    13244 2012-04-15 23:31:04    1986-08-18
"""

  
es.plot() # 绘制实体名单

sessions_df=data["sessions"]
print(sessions_df.head(2))
"""
   session_id  customer_id   device       session_start
0           1            2  desktop 2014-01-01 00:00:00
1           2            5   mobile 2014-01-01 00:17:20
"""


transaction_df =data["transactions"]
print(transaction_df.head(2))
"""
   transaction_id  session_id    transaction_time product_id  amount
0             298           1 2014-01-01 00:00:00          5  127.64
1               2           1 2014-01-01 00:01:05          2  109.48
"""
products_df =data["products"]
print(products_df.head(2))
"""
  product_id brand
0          1     B
1          2     B
"""

# 描述出来的关系
feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="customers")
print(feature_matrix.head(2))



