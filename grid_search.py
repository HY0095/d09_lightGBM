# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics
import time


def main():
    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    parser = argparse.ArgumentParser()
    parser.add_argument('-dpath', default='')
    parser.add_argument('-data', default='')
    parser.add_argument('-target', default='label')
    parser.add_argument('--config', default='./configs/lightgbm_0.json')
    parser.add_argument('-log', default='grid_search_' + current_time + '.log')
    args = parser.parse_args()

    # step 1. Load Data
    log = open(args.log, 'w')
    print >> log, 'Load Data'
    l1_score_train = pd.read_csv('l1_score_train.csv', index_col=0)
    l1_score_test = pd.read_csv('l1_score_test.csv', index_col=0)
    raw_dt = pd.read_csv(args.dpath + args.data, index_col=0)
    raw_dt_0 = raw_dt[raw_dt.label == 0]
    raw_dt_1 = raw_dt[raw_dt.label == 1]
    raw_dt = raw_dt_0.append(raw_dt_1)
    print >> log, "l1_score_train.shape: "
    print >> log, l1_score_train.shape
    print >> log, "l1_score_test.shape: "
    print >> log, l1_score_test.shape

    # Step 2. Data Merge
    print >> log, 'Data Merge'
    l1_score_train = pd.merge(l1_score_train, raw_dt, left_index=True, right_index=True, how='inner')
    l1_score_test = pd.merge(l1_score_test, raw_dt, left_index=True, right_index=True, how='inner')
    trains = pd.concat([l1_score_train, l1_score_test], axis=0)
    print >> log, "trains.shape: "
    print >> log, trains.shape
    log.close()

    # Step 3: Data Split
    features = [item for item in trains.columns if item not in [args.target, 'New_label', 'date']]
    with open('features.txt', 'w') as f:
        f.write(','.join(features))

    # Step 4. Data Trans.

    lgb_train = lgb.Dataset(trains[features].values, trains[args.target].values, free_raw_data=False)
    # lgb_test = lgb.Dataset(l1_score_test[features].values, l1_score_train[args.target].values, reference=lgb_train, free_raw_data=False)

    print('params. Init. ')
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
    }

    print('Cross Validate')
    min_error = float('Inf')
    best_params = {}

    log = open(args.log, 'a')
    print >> log, "Turing 1：num_leaves & max_depth"
    # accuracy
    for num_leaves in range(20, 200, 5):
        for max_depth in range(3, 8, 1):
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth
            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=2018,
                nfold=5,
                metrics=['binary_error'],
                early_stopping_rounds=50,
                verbose_eval=True
            )

            mean_merror = pd.Series(cv_results['binary_error-mean']).min()
            boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()

            if mean_merror < min_error:
                min_error = mean_merror
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth

    params['num_leaves'] = best_params['num_leaves']
    params['max_depth'] = best_params['max_depth']
    print >> log, "Turing 1 Results "
    print >> log, best_params
    log.close()

    log = open(args.log, 'a')
    print >> log, "Turing 2：max_bin & min_data_in_leaf"
    for max_bin in range(1, 255, 5):
        for min_data_in_leaf in range(10, 200, 5):
            params['max_bin'] = max_bin
            params['min_data_in_leaf'] = min_data_in_leaf

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=42,
                nfold=5,
                metrics=['binary_error'],
                early_stopping_rounds=3,
                verbose_eval=True
            )

            mean_merror = pd.Series(cv_results['binary_error-mean']).min()
            boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()

            if mean_merror < min_error:
                min_error = mean_merror
                best_params['max_bin'] = max_bin
                best_params['min_data_in_leaf'] = min_data_in_leaf

    params['min_data_in_leaf'] = best_params['min_data_in_leaf']
    params['max_bin'] = best_params['max_bin']
    print >> log, "Turing 2 Results "
    print >> log, best_params
    log.close()

    log = open(args.log, 'a')
    print >> log, "Turing 3:feature_fraction & bagging_fraction & bagging_freq"
    for feature_fraction in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for bagging_fraction in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for bagging_freq in range(0, 50, 5):
                params['feature_fraction'] = feature_fraction
                params['bagging_fraction'] = bagging_fraction
                params['bagging_freq'] = bagging_freq

                cv_results = lgb.cv(
                    params,
                    lgb_train,
                    seed=42,
                    nfold=5,
                    metrics=['binary_error'],
                    early_stopping_rounds=3,
                    verbose_eval=True
                )

                mean_merror = pd.Series(cv_results['binary_error-mean']).min()
                boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()

                if mean_merror < min_error:
                    min_error = mean_merror
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
                    best_params['bagging_freq'] = bagging_freq

    params['feature_fraction'] = best_params['feature_fraction']
    params['bagging_fraction'] = best_params['bagging_fraction']
    params['bagging_freq'] = best_params['bagging_freq']
    print >> log, "Turing 3 Result:"
    print >> log, best_params
    log.close()

    log = open(argparse.log, 'a')
    print >> log, "turing 4：lambda_l1 & lambda_l2 & min_split_gain"
    for lambda_l1 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for lambda_l2 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                params['lambda_l1'] = lambda_l1
                params['lambda_l2'] = lambda_l2
                params['min_split_gain'] = min_split_gain

                cv_results = lgb.cv(
                    params,
                    lgb_train,
                    seed=42,
                    nfold=5,
                    metrics=['binary_error'],
                    early_stopping_rounds=3,
                    verbose_eval=True
                )

                mean_merror = pd.Series(cv_results['binary_error-mean']).min()
                boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()

                if mean_merror < min_error:
                    min_error = mean_merror
                    best_params['lambda_l1'] = lambda_l1
                    best_params['lambda_l2'] = lambda_l2
                    best_params['min_split_gain'] = min_split_gain

    params['lambda_l1'] = best_params['lambda_l1']
    params['lambda_l2'] = best_params['lambda_l2']
    params['min_split_gain'] = best_params['min_split_gain']

    print >> log, "Final Best Params: "
    print >> log, best_params
    log.close()


if __name__ == "__main__":
    main()

    # ### 训练
    # params['learning_rate'] = 0.01
    # lgb.train(
    #     params,  # 参数字典
    #     lgb_train,  # 训练集
    #     valid_sets=lgb_eval,  # 验证集
    #     num_boost_round=2000,  # 迭代次数
    #     early_stopping_rounds=50  # 早停次数
    # )
    #
    # ### 线下预测
    # print ("线下预测")
    # preds_offline = lgb.predict(offline_test_X, num_iteration=lgb.best_iteration)  # 输出概率
    # offline = offline_test[['instance_id', 'is_trade']]
    # offline['preds'] = preds_offline
    # offline.is_trade = offline['is_trade'].astype(np.float64)
    # print('log_loss', metrics.log_loss(offline.is_trade, offline.preds))
    #
    # ### 线上预测
    # print("线上预测")
    # preds_online = lgb.predict(online_test_X, num_iteration=lgb.best_iteration)  # 输出概率
    # online = online_test[['instance_id']]
    # online['preds'] = preds_online
    # online.rename(columns={'preds': 'predicted_score'}, inplace=True)  # 更改列名
    # online.to_csv("./data/20180405.txt", index=None, sep=' ')  # 保存结果
    #
    # ### 保存模型
    # from sklearn.externals import joblib
    #
    # joblib.dump(lgb, 'lgb.pkl')
    #
    # ### 特征选择
    # df = pd.DataFrame(X_train.columns.tolist(), columns=['feature'])
    # df['importance'] = list(lgb.feature_importance())  # 特征分数
    # df = df.sort_values(by='importance', ascending=False)  # 特征排序
    # df.to_csv("./data/feature_score_20180331.csv", index=None, encoding='gbk')  # 保存分数

