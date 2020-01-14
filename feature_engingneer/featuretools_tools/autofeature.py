import featuretools as ft
# import numpy as np
# import pandas as pd


class AutoFeaturetools(object):
    def __init__(self, entityname):
        self.entityname = entityname
        self.es = self._creat_empty_entity()

    def _creat_empty_entity(self):  # 创建空实体集
        return ft.EntitySet(id=self.entityname)

    @staticmethod  # 直接创建
    def _creat_entity(entityname, entity, relationship):
        """
        :param entityname:  #  实体名
        :param entity: # 实体
        :param relationship:  # 关系
        :return:
        """
        ft.EntitySet(id=entityname, entities=entity, relationships=relationship)

    # 往实体集中加实体
    def add_entity(self, entity_id, df, index=None, variable_types=None, make_index=False,
                   time_index=None, secondary_time_index=None, already_sorted=False):  # 添加一个带索引的df数据到实体中
        self.es.entity_from_dataframe(entity_id=entity_id,
                                      dataframe=df,
                                      index=index,
                                      variable_types=variable_types,
                                      make_index=make_index,
                                      time_index=time_index,
                                      secondary_time_index=secondary_time_index,
                                      already_sorted=already_sorted)

    # 创建关系
    def create_relationship(self, entity_id_a, indexa, entity_id_b, indexb):
        new_relationship = ft.Relationship(self.es[entity_id_a][indexa], self.es[entity_id_b][indexb])
        return new_relationship

    # 添加关系到实体集中
    def add_relationship_entity(self, relationship):
        self.es.add_relationship(relationship)

    # 查看实体
    def view_entity(self):
        print(self.es)

    # def merge(self):  # 聚合操作
    #
    #
    # def transform(self):  # 转换操作

    # 实体集进行拆分处理
    def split(self, base_entity_id,  new_entity_id,  index,
              additional_variables=None, copy_variables=None,
              make_time_index=None, make_secondary_time_index=None,
              new_entity_time_index=None, new_entity_secondary_time_index=None):
        self.es.normalize_entity(base_entity_id=base_entity_id,   # 父实体
                                 new_entity_id=new_entity_id,    # 新实体
                                 index=index,   # 创建唯一index
                                 additional_variables=additional_variables,  # 添加变量
                                 copy_variables=copy_variables,
                                 make_time_index=make_time_index,
                                 make_secondary_time_index=make_secondary_time_index,
                                 new_entity_time_index=new_entity_time_index,
                                 new_entity_secondary_time_index=new_entity_secondary_time_index,
                                 )

    # 获取特征/深度特征合成
    def get_feature(self, entities=None, relationships=None, target_entity=None,
                    cutoff_time=None, instance_ids=None, agg_primitives=None, trans_primitives=None,
                    groupby_trans_primitives=None, allowed_paths=None, max_depth=2, ignore_entities=None,
                    ignore_variables=None, primitive_options=None, seed_features=None, drop_contains=None,
                    drop_exact=None, where_primitives=None, max_features=-1, cutoff_time_in_index=False,
                    save_progress=None, features_only=False, training_window=None, approximate=None,
                    chunk_size=None, n_jobs=1, dask_kwargs=None, verbose=False,
                    return_variable_types=None, progress_callback=None):
        """
        :param entities:  # 实体字典   dict
        :param relationships:   # 关系  list[(str,str,str)]
        :param target_entity:   # 目标实体
        :param cutoff_time:  截断时间   df 或Datetime
        :param instance_ids: list 要计算特征的实例列表。
        :param agg_primitives:  可选项， list  要扩展的 特征，
        要应用的聚合特征类型列表。
        Default: [“sum”, “std”, “max”, “skew”, “min”, “mean”, “count”,
        “percent_true”, “num_unique”, “mode”]
        :param trans_primitives:
        要应用的转换特征函数列表。
        Default: [“day”, “year”, “month”, “weekday”, “haversine”,
        “num_words”, “num_characters”]
        :param groupby_trans_primitives:
        用于制作groupby Transform特性的转换原始列表
        list
        :param allowed_paths:  允许创建特征的实体路径。
        :param max_depth:  最大允许的深度， 一般为2 int
        :param ignore_entities:  创建特征时要列入黑名单的实体列表。list
        :param ignore_variables:  在创建特征时，列出每个实体中要列入黑名单的特定变量。 list
        :param primitive_options:
        :param seed_features:  要使用的手动定义的特征列表。
        :param drop_contains:  删除名称中包含这些字符串的特征。list 可选项
        :param drop_exact:  删除在名称上与这些字符串完全匹配的特性。可选项
        :param where_primitives: list可选项，Default:[“count”]
        :param max_features: 限制生成特征数量的上线，-1无限制
        :param cutoff_time_in_index: 设为True,  返回带多个索引的df数据， 第一个索引 instance id，
        第二索引为 time, 最后根据 (time ,instance id) 进行排序
        :param save_progress:  可选项，保存中间计算结果的路径
        :param features_only:
        :param training_window:timedelta, or str 可选项。如果使用None，所有数据在截止时间前就已经使用，默认为None
        窗口定义在计算特征时可以使用截止时间数据之前的时间。
        :param approximate: Timedelta or str 频率，以分组实例与相似的截止时间，通过特征与昂贵的计算。
        例如，如果桶是24小时，所有在同一天具有截止时间的实例将使用相同的计算昂贵的特征。
        :param chunk_size:  每次计算的计算的矩阵的行数
        :param n_jobs:  # 并行进程个数
        :param dask_kwargs: dict 可选项  创建DASK客户端和调度程序时要传递的关键字参数字典。即使没有设置n_jobs，
        使用dask_kwargs也将启用多进程。
        :param verbose: 是否开启打印
        :param return_variable_types:  list  or str 可选项， 返回的变量类型， 默认为None,输出为数值的（连续型）
        离散与布尔型   若写成"all", 给出所有类型的数据
        :param progress_callback:
        :return:
        entityset  初始化的实体集
        """
        feature_matrix, feature_names = ft.dfs(
            entities=entities,   # 实体字典
            relationships=relationships,  # 关系
            entityset=self.es,   # 实体集
            target_entity=target_entity,  # 目标实体id
            cutoff_time=cutoff_time,  #
            instance_ids=instance_ids,
            agg_primitives=agg_primitives,
            trans_primitives=trans_primitives,  # 要应用的变换特征函数列表
            groupby_trans_primitives=groupby_trans_primitives,
            allowed_paths=allowed_paths,
            max_depth=max_depth,
            ignore_entities=ignore_entities,
            ignore_variables=ignore_variables,
            primitive_options=primitive_options,
            seed_features=seed_features,
            drop_contains=drop_contains,
            drop_exact=drop_exact,
            where_primitives=where_primitives,
            max_features=max_features,
            cutoff_time_in_index=cutoff_time_in_index,
            save_progress=save_progress,
            features_only=features_only,
            training_window=training_window,
            approximate=approximate,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            dask_kwargs=dask_kwargs,
            verbose=verbose,
            return_variable_types=return_variable_types,
            progress_callback=progress_callback)
        return feature_matrix, feature_names

    # 对特征矩阵进行编码
    @staticmethod
    def feature_encoder(feature_matrix, features, top_n=10, include_unknown=True,
                        to_encode=None, inplace=False, drop_first=False, verbose=False):
        """
        :param feature_matrix:  df 格式的特征
        :param features:   特征矩阵的特征定义
        :param top_n:   包含的tokn的数目， int 或 使用字典， key为特征的名字，value对敌营top的数目，默认为10
        :param include_unknown: #默认为True,添加编码未知类的特性。默认为true
        :param to_encode: #要编码的特征名字列表 ["name1","name2"]
        :param inplace: 默认为False, 是否替代
        :param drop_first: 是否通过去除第一级从k个分类级别获取k-1个。默认为false
        :param verbose: 打印进度信息，
        :return:
        """
        fm_encoded, f_encoded = ft.encode_features(feature_matrix=feature_matrix,
                                                   features=features,
                                                   top_n=top_n,
                                                   include_unknown=include_unknown,
                                                   to_encode=to_encode,
                                                   inplace=inplace,
                                                   drop_first=drop_first,
                                                   verbose=verbose)
        return fm_encoded, f_encoded

    # 计算特征矩阵
    def cal_feature_martix(self, features, cutoff_time=None, instance_ids=None,
                           entities=None, relationships=None, cutoff_time_in_index=False, training_window=None,
                           approximate=None, verbose=False, chunk_size=None, n_jobs=1, dask_kwargs=None,
                           save_progress=None, progress_callback=None):
        """
        :param features:  list, 要被计算的特征定义
        :param cutoff_time:  df/datatime, 指定时间端计算特征
        :param instance_ids: list 计算特征的实例列表。只有当截止时间是单个日期时间时才使用
        :param entities:  dict 实体字典
        :param relationships: #关系， list
        :param cutoff_time_in_index: bool 如果为True, 返回多索引的df，第一个索引instance id，
        第二个索引cutoff time  结果按照（time,instance id）排序
        :param training_window: timedelta, or str 可选项。如果使用None，所有数据在截止时间前就已经使用，默认为None
        窗口定义在计算特征时可以使用截止时间数据之前的时间。
        :param approximate: Timedelta or str 频率，以分组实例与相似的截止时间，通过特征与昂贵的计算。
        例如，如果桶是24小时，所有在同一天具有截止时间的实例将使用相同的计算昂贵的特征。
        :param verbose: bool型， 可选项，打印进度信息
        :param chunk_size:  int or float or None,  每次输出特征矩阵的最大行
        :param n_jobs:  # 并行进程个数 int , 可选项
        :param dask_kwargs: dict, 可选项
        :param save_progress: # str ，可选项，保存中间计算结果
        :param progress_callback:
        :return:
        entityset: 实体对象
        """
        features_matrix = ft.calculate_feature_matrix(features=features,
                                                      entityset=self.es,
                                                      cutoff_time=cutoff_time,
                                                      instance_ids=instance_ids,
                                                      entities=entities,
                                                      relationships=relationships,
                                                      cutoff_time_in_index=cutoff_time_in_index,
                                                      training_window=training_window,
                                                      approximate=approximate,
                                                      verbose=verbose,
                                                      chunk_size=chunk_size,
                                                      n_jobs=n_jobs,
                                                      dask_kwargs=dask_kwargs,
                                                      save_progress=save_progress,
                                                      progress_callback=progress_callback)
        return features_matrix

    @staticmethod
    def _save_features(features, location=None, profile_name=None):
        """
        :param features:  list, 特征列表
        :param location:  str 可选项，或文件对象，默认为None，如果为 None, 返回一个系列化特征的json字符串
        :param profile_name: strbool
        :return:
        """
        ft.save_features(features=features,
                         location=location,
                         profile_name=profile_name)

    @staticmethod  # 与保存特征刚好相反
    def _load_features(features, profile_name=None):
        """
        :param features: str 或文件对象，  特征所在的位置
        :param profile_name: str bool 型
        :return:   list
        """
        feature = ft.load_features(features=features, profile_name=profile_name)
        return feature

    def add_last_time_indexes(self, updated_entities=None):  # 添加最后一次索引
        self.es.add_last_time_indexes(updated_entities=updated_entities)

    def set_interesting_values(self,entity_id, columns,values):  # 设置感兴趣的值
        """
        :param entity_id: 实体名
        :param columns: 列名
        :param values:  设定的值 list
        :return:
        """
        self.es[entity_id][columns] = values

    def get_entity_df(self, entity_id):  # 将实体转成df 数据
        return self.es[entity_id].df
