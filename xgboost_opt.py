from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.feature_selection import chi2
import xgboostlib.xgboost as xgboost

__author__ = 'YBeer'

train_result = pd.DataFrame.from_csv("train_result.csv")
col = list(train_result.columns.values)
result_ind = list(train_result[col[0]].value_counts().index)
# train_result = pd.get_dummies(train_result)
train_result = np.array(train_result).ravel()
train_result_xgb = np.zeros(train_result.shape)
for i in range(1, len(result_ind)):
    train_result_xgb += (train_result == result_ind[i]) * i
# print train_result_xgb

train = pd.DataFrame.from_csv("train_dummied_150_sep_dep_fln_b_r_v5.csv")
train.fillna(0)
train_arr = np.array(train)
col_list = list(train.columns.values)

# print train_result.shape[1], ' categorial'
print train.shape[1], ' columns'

print 'absing'
for i in range(train_arr.shape[0]):
    for j in range(train_arr.shape[1]):
        train_arr[i, j] = np.abs(train_arr[i, j])

chi2_params = chi2(train_arr, train_result)

# for i in range(train_arr.shape[1]):
#     print col_list[i], chi2_params[0][i]

del train_arr

best_metric = 10
best_params = []
param_grid = {'silent': [1], 'nthread': [4], 'num_class': [38], 'eval_metric': ['mlogloss'], 'eta': [0.1],
              'objective': ['multi:softprob'], 'max_depth': [5], 'chi2_lim': [0], 'num_round': [500],
              'subsample': [0.5, 0.6]}

for params in ParameterGrid(param_grid):
    print params

    # filtering low chi2 cols
    chi2_lim = params['chi2_lim']
    chi2_cols = []
    for i in range(train.shape[1]):
        if chi2_params[0][i] > chi2_lim:
            chi2_cols.append(col_list[i])

    print len(chi2_cols), ' chi2 columns'
    train_arr = train.copy(deep=True)
    train_arr = train_arr[chi2_cols]

    # Standardizing
    stding = StandardScaler()
    train_arr = stding.fit_transform(train_arr)

    print 'start CV'

    # CV
    X_train, X_test, y_train, y_test = train_test_split(train_arr, train_result_xgb, test_size=0.25, random_state=1)
    metric = []

    # train machine learning
    xg_train = xgboost.DMatrix(X_train, label=y_train)
    xg_test = xgboost.DMatrix(X_test, label=y_test)

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]

    num_round = params['num_round']
    xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

    # predict
    class_pred = xgclassifier.predict(xg_test)
    class_pred = class_pred.reshape(y_test.shape[0], 38)

    # evaluate
    # print log_loss(y_test, class_pred)
    metric = log_loss(y_test, class_pred)

    print 'The log loss is: ', metric
    if metric < best_metric:
        best_metric = metric
        best_params = params
    print 'The best metric is:', best_metric, 'for the params:', best_params

# 1571  columns
# absing
# {'num_class': 38, 'silent': 1, 'eval_metric': 'mlogloss', 'subsample': 0.7, 'nthread': 4, 'objective': 'multi:softprob', 'eta': 0.1, 'num_round': 500, 'max_depth': 5, 'chi2_lim': 0}
# 1571  chi2 columns
# C:\Users\yaia\Anaconda\lib\site-packages\sklearn\utils\validation.py:498: UserWarning: StandardScaler assumes floating point values as input, got int64
#   "got %s" % (estimator, X.dtype))
# start CV
# [0]	train-mlogloss:2.797980	test-mlogloss:2.810046
# [1]	train-mlogloss:2.496364	test-mlogloss:2.515575
# [2]	train-mlogloss:2.281890	test-mlogloss:2.307391
# [3]	train-mlogloss:2.116182	test-mlogloss:2.147533
# [4]	train-mlogloss:1.980520	test-mlogloss:2.016521
# [5]	train-mlogloss:1.866207	test-mlogloss:1.906769
# [6]	train-mlogloss:1.767753	test-mlogloss:1.812126
# [7]	train-mlogloss:1.680725	test-mlogloss:1.729014
# [8]	train-mlogloss:1.605376	test-mlogloss:1.656757
# [9]	train-mlogloss:1.537959	test-mlogloss:1.592986
# [10]	train-mlogloss:1.476077	test-mlogloss:1.534297
# [11]	train-mlogloss:1.420754	test-mlogloss:1.482501
# [12]	train-mlogloss:1.370801	test-mlogloss:1.435462
# [13]	train-mlogloss:1.325140	test-mlogloss:1.392668
# [14]	train-mlogloss:1.283265	test-mlogloss:1.353527
# [15]	train-mlogloss:1.244964	test-mlogloss:1.317754
# [16]	train-mlogloss:1.209752	test-mlogloss:1.285354
# [17]	train-mlogloss:1.177248	test-mlogloss:1.255268
# [18]	train-mlogloss:1.146759	test-mlogloss:1.227133
# [19]	train-mlogloss:1.118795	test-mlogloss:1.201421
# [20]	train-mlogloss:1.092766	test-mlogloss:1.177949
# [21]	train-mlogloss:1.068132	test-mlogloss:1.155536
# [22]	train-mlogloss:1.045604	test-mlogloss:1.135149
# [23]	train-mlogloss:1.024512	test-mlogloss:1.116340
# [24]	train-mlogloss:1.004719	test-mlogloss:1.098502
# [25]	train-mlogloss:0.985955	test-mlogloss:1.081780
# [26]	train-mlogloss:0.968521	test-mlogloss:1.066486
# [27]	train-mlogloss:0.951904	test-mlogloss:1.052039
# [28]	train-mlogloss:0.935833	test-mlogloss:1.038229
# [29]	train-mlogloss:0.921019	test-mlogloss:1.025209
# [30]	train-mlogloss:0.907004	test-mlogloss:1.013162
# [31]	train-mlogloss:0.893331	test-mlogloss:1.001534
# [32]	train-mlogloss:0.880539	test-mlogloss:0.990433
# [33]	train-mlogloss:0.868466	test-mlogloss:0.980208
# [34]	train-mlogloss:0.857085	test-mlogloss:0.970563
# [35]	train-mlogloss:0.846159	test-mlogloss:0.961616
# [36]	train-mlogloss:0.835526	test-mlogloss:0.953032
# [37]	train-mlogloss:0.825195	test-mlogloss:0.944943
# [38]	train-mlogloss:0.815552	test-mlogloss:0.937367
# [39]	train-mlogloss:0.805955	test-mlogloss:0.929613
# [40]	train-mlogloss:0.796908	test-mlogloss:0.922354
# [41]	train-mlogloss:0.788215	test-mlogloss:0.915407
# [42]	train-mlogloss:0.779695	test-mlogloss:0.908774
# [43]	train-mlogloss:0.771961	test-mlogloss:0.902678
# [44]	train-mlogloss:0.764389	test-mlogloss:0.897038
# [45]	train-mlogloss:0.756762	test-mlogloss:0.891573
# [46]	train-mlogloss:0.749654	test-mlogloss:0.886236
# [47]	train-mlogloss:0.742888	test-mlogloss:0.881244
# [48]	train-mlogloss:0.735850	test-mlogloss:0.876089
# [49]	train-mlogloss:0.729077	test-mlogloss:0.871141
# [50]	train-mlogloss:0.722797	test-mlogloss:0.866796
# [51]	train-mlogloss:0.716946	test-mlogloss:0.862700
# [52]	train-mlogloss:0.711118	test-mlogloss:0.858837
# [53]	train-mlogloss:0.705346	test-mlogloss:0.854716
# [54]	train-mlogloss:0.699921	test-mlogloss:0.850982
# [55]	train-mlogloss:0.694531	test-mlogloss:0.847377
# [56]	train-mlogloss:0.689264	test-mlogloss:0.843846
# [57]	train-mlogloss:0.684386	test-mlogloss:0.840602
# [58]	train-mlogloss:0.679166	test-mlogloss:0.837164
# [59]	train-mlogloss:0.674427	test-mlogloss:0.833986
# [60]	train-mlogloss:0.669878	test-mlogloss:0.831079
# [61]	train-mlogloss:0.665697	test-mlogloss:0.828524
# [62]	train-mlogloss:0.661150	test-mlogloss:0.825574
# [63]	train-mlogloss:0.656775	test-mlogloss:0.822901
# [64]	train-mlogloss:0.652630	test-mlogloss:0.820397
# [65]	train-mlogloss:0.648401	test-mlogloss:0.817845
# [66]	train-mlogloss:0.644358	test-mlogloss:0.815354
# [67]	train-mlogloss:0.640633	test-mlogloss:0.813116
# [68]	train-mlogloss:0.636574	test-mlogloss:0.810734
# [69]	train-mlogloss:0.632763	test-mlogloss:0.808660
# [70]	train-mlogloss:0.629150	test-mlogloss:0.806725
# [71]	train-mlogloss:0.625830	test-mlogloss:0.804898
# [72]	train-mlogloss:0.622414	test-mlogloss:0.802874
# [73]	train-mlogloss:0.618937	test-mlogloss:0.800830
# [74]	train-mlogloss:0.615531	test-mlogloss:0.798760
# [75]	train-mlogloss:0.611982	test-mlogloss:0.797105
# [76]	train-mlogloss:0.608698	test-mlogloss:0.795068
# [77]	train-mlogloss:0.605621	test-mlogloss:0.793553
# [78]	train-mlogloss:0.602513	test-mlogloss:0.792151
# [79]	train-mlogloss:0.599317	test-mlogloss:0.790618
# [80]	train-mlogloss:0.596145	test-mlogloss:0.789242
# [81]	train-mlogloss:0.593111	test-mlogloss:0.787793
# [82]	train-mlogloss:0.589980	test-mlogloss:0.786135
# [83]	train-mlogloss:0.587115	test-mlogloss:0.784740
# [84]	train-mlogloss:0.584335	test-mlogloss:0.783338
# [85]	train-mlogloss:0.581447	test-mlogloss:0.781898
# [86]	train-mlogloss:0.578366	test-mlogloss:0.780499
# [87]	train-mlogloss:0.575566	test-mlogloss:0.779263
# [88]	train-mlogloss:0.572830	test-mlogloss:0.777886
# [89]	train-mlogloss:0.570085	test-mlogloss:0.776794
# [90]	train-mlogloss:0.567617	test-mlogloss:0.775669
# [91]	train-mlogloss:0.565175	test-mlogloss:0.774531
# [92]	train-mlogloss:0.562589	test-mlogloss:0.773384
# [93]	train-mlogloss:0.559783	test-mlogloss:0.772266
# [94]	train-mlogloss:0.557115	test-mlogloss:0.771269
# [95]	train-mlogloss:0.554701	test-mlogloss:0.770329
# [96]	train-mlogloss:0.552044	test-mlogloss:0.768992
# [97]	train-mlogloss:0.549490	test-mlogloss:0.767865
# [98]	train-mlogloss:0.547174	test-mlogloss:0.766985
# [99]	train-mlogloss:0.544945	test-mlogloss:0.765885
# [100]	train-mlogloss:0.542639	test-mlogloss:0.764883
# [101]	train-mlogloss:0.540187	test-mlogloss:0.763895
# [102]	train-mlogloss:0.537704	test-mlogloss:0.762935
# [103]	train-mlogloss:0.535322	test-mlogloss:0.762017
# [104]	train-mlogloss:0.533145	test-mlogloss:0.761160
# [105]	train-mlogloss:0.530742	test-mlogloss:0.760030
# [106]	train-mlogloss:0.528085	test-mlogloss:0.758945
# [107]	train-mlogloss:0.525841	test-mlogloss:0.758088
# [108]	train-mlogloss:0.523526	test-mlogloss:0.757011
# [109]	train-mlogloss:0.521440	test-mlogloss:0.756259
# [110]	train-mlogloss:0.519321	test-mlogloss:0.755685
# [111]	train-mlogloss:0.517289	test-mlogloss:0.754802
# [112]	train-mlogloss:0.515155	test-mlogloss:0.754114
# [113]	train-mlogloss:0.512913	test-mlogloss:0.753302
# [114]	train-mlogloss:0.510912	test-mlogloss:0.752540
# [115]	train-mlogloss:0.508825	test-mlogloss:0.751814
# [116]	train-mlogloss:0.506521	test-mlogloss:0.751118
# [117]	train-mlogloss:0.504346	test-mlogloss:0.750453
# [118]	train-mlogloss:0.502358	test-mlogloss:0.749709
# [119]	train-mlogloss:0.500358	test-mlogloss:0.748991
# [120]	train-mlogloss:0.498453	test-mlogloss:0.748278
# [121]	train-mlogloss:0.496539	test-mlogloss:0.747819
# [122]	train-mlogloss:0.494650	test-mlogloss:0.747287
# [123]	train-mlogloss:0.492749	test-mlogloss:0.746558
# [124]	train-mlogloss:0.490965	test-mlogloss:0.745965
# [125]	train-mlogloss:0.488767	test-mlogloss:0.745252
# [126]	train-mlogloss:0.486946	test-mlogloss:0.744731
# [127]	train-mlogloss:0.485094	test-mlogloss:0.744070
# [128]	train-mlogloss:0.483307	test-mlogloss:0.743517
# [129]	train-mlogloss:0.481536	test-mlogloss:0.742945
# [130]	train-mlogloss:0.479748	test-mlogloss:0.742398
# [131]	train-mlogloss:0.477879	test-mlogloss:0.741865
# [132]	train-mlogloss:0.475952	test-mlogloss:0.741108
# [133]	train-mlogloss:0.474170	test-mlogloss:0.740667
# [134]	train-mlogloss:0.472351	test-mlogloss:0.740180
# [135]	train-mlogloss:0.470557	test-mlogloss:0.739533
# [136]	train-mlogloss:0.468859	test-mlogloss:0.739212
# [137]	train-mlogloss:0.467183	test-mlogloss:0.738735
# [138]	train-mlogloss:0.465182	test-mlogloss:0.738171
# [139]	train-mlogloss:0.463501	test-mlogloss:0.737671
# [140]	train-mlogloss:0.461794	test-mlogloss:0.737189
# [141]	train-mlogloss:0.459946	test-mlogloss:0.736609
# [142]	train-mlogloss:0.458008	test-mlogloss:0.735860
# [143]	train-mlogloss:0.456325	test-mlogloss:0.735348
# [144]	train-mlogloss:0.454611	test-mlogloss:0.734535
# [145]	train-mlogloss:0.452864	test-mlogloss:0.734140
# [146]	train-mlogloss:0.451230	test-mlogloss:0.733612
# [147]	train-mlogloss:0.449727	test-mlogloss:0.733159
# [148]	train-mlogloss:0.447976	test-mlogloss:0.732762
# [149]	train-mlogloss:0.446224	test-mlogloss:0.732254
# [150]	train-mlogloss:0.444615	test-mlogloss:0.731826
# [151]	train-mlogloss:0.443011	test-mlogloss:0.731443
# [152]	train-mlogloss:0.441588	test-mlogloss:0.730954
# [153]	train-mlogloss:0.440131	test-mlogloss:0.730624
# [154]	train-mlogloss:0.438592	test-mlogloss:0.730237
# [155]	train-mlogloss:0.437024	test-mlogloss:0.729788
# [156]	train-mlogloss:0.435362	test-mlogloss:0.729342
# [157]	train-mlogloss:0.433934	test-mlogloss:0.729020
# [158]	train-mlogloss:0.432386	test-mlogloss:0.728395
# [159]	train-mlogloss:0.430995	test-mlogloss:0.727992
# [160]	train-mlogloss:0.429692	test-mlogloss:0.727557
# [161]	train-mlogloss:0.428312	test-mlogloss:0.727265
# [162]	train-mlogloss:0.426855	test-mlogloss:0.726966
# [163]	train-mlogloss:0.425350	test-mlogloss:0.726533
# [164]	train-mlogloss:0.423965	test-mlogloss:0.726138
# [165]	train-mlogloss:0.422621	test-mlogloss:0.725747
# [166]	train-mlogloss:0.421070	test-mlogloss:0.725165
# [167]	train-mlogloss:0.419618	test-mlogloss:0.724869
# [168]	train-mlogloss:0.418100	test-mlogloss:0.724543
# [169]	train-mlogloss:0.416622	test-mlogloss:0.724230
# [170]	train-mlogloss:0.415201	test-mlogloss:0.723974
# [171]	train-mlogloss:0.413894	test-mlogloss:0.723705
# [172]	train-mlogloss:0.412619	test-mlogloss:0.723467
# [173]	train-mlogloss:0.411310	test-mlogloss:0.723227
# [174]	train-mlogloss:0.409869	test-mlogloss:0.722890
# [175]	train-mlogloss:0.408428	test-mlogloss:0.722417
# [176]	train-mlogloss:0.407157	test-mlogloss:0.722114
# [177]	train-mlogloss:0.405900	test-mlogloss:0.721725
# [178]	train-mlogloss:0.404679	test-mlogloss:0.721429
# [179]	train-mlogloss:0.403351	test-mlogloss:0.721118
# [180]	train-mlogloss:0.402129	test-mlogloss:0.720810
# [181]	train-mlogloss:0.400717	test-mlogloss:0.720351
# [182]	train-mlogloss:0.399337	test-mlogloss:0.720011
# [183]	train-mlogloss:0.398115	test-mlogloss:0.719755
# [184]	train-mlogloss:0.396727	test-mlogloss:0.719509
# [185]	train-mlogloss:0.395543	test-mlogloss:0.719295
# [186]	train-mlogloss:0.394223	test-mlogloss:0.718991
# [187]	train-mlogloss:0.392841	test-mlogloss:0.718582
# [188]	train-mlogloss:0.391701	test-mlogloss:0.718348
# [189]	train-mlogloss:0.390384	test-mlogloss:0.718092
# [190]	train-mlogloss:0.389141	test-mlogloss:0.717957
# [191]	train-mlogloss:0.387875	test-mlogloss:0.717595
# [192]	train-mlogloss:0.386725	test-mlogloss:0.717224
# [193]	train-mlogloss:0.385451	test-mlogloss:0.716857
# [194]	train-mlogloss:0.384230	test-mlogloss:0.716459
# [195]	train-mlogloss:0.383073	test-mlogloss:0.716040
# [196]	train-mlogloss:0.381829	test-mlogloss:0.715783
# [197]	train-mlogloss:0.380650	test-mlogloss:0.715448
# [198]	train-mlogloss:0.379435	test-mlogloss:0.715270
# [199]	train-mlogloss:0.378236	test-mlogloss:0.714941
# [200]	train-mlogloss:0.377022	test-mlogloss:0.714663
# [201]	train-mlogloss:0.375810	test-mlogloss:0.714414
# [202]	train-mlogloss:0.374682	test-mlogloss:0.714157
# [203]	train-mlogloss:0.373671	test-mlogloss:0.713919
# [204]	train-mlogloss:0.372366	test-mlogloss:0.713663
# [205]	train-mlogloss:0.371266	test-mlogloss:0.713320
# [206]	train-mlogloss:0.370278	test-mlogloss:0.713038
# [207]	train-mlogloss:0.369219	test-mlogloss:0.712829
# [208]	train-mlogloss:0.368120	test-mlogloss:0.712628
# [209]	train-mlogloss:0.367073	test-mlogloss:0.712497
# [210]	train-mlogloss:0.365893	test-mlogloss:0.712207
# [211]	train-mlogloss:0.364771	test-mlogloss:0.712042
# [212]	train-mlogloss:0.363562	test-mlogloss:0.711820
# [213]	train-mlogloss:0.362477	test-mlogloss:0.711669
# [214]	train-mlogloss:0.361404	test-mlogloss:0.711583
# [215]	train-mlogloss:0.360370	test-mlogloss:0.711461
# [216]	train-mlogloss:0.359333	test-mlogloss:0.711204
# [217]	train-mlogloss:0.358279	test-mlogloss:0.711076
# [218]	train-mlogloss:0.357100	test-mlogloss:0.710847
# [219]	train-mlogloss:0.355961	test-mlogloss:0.710456
# [220]	train-mlogloss:0.354972	test-mlogloss:0.710332
# [221]	train-mlogloss:0.353995	test-mlogloss:0.710202
# [222]	train-mlogloss:0.352944	test-mlogloss:0.710103
# [223]	train-mlogloss:0.351817	test-mlogloss:0.709880
# [224]	train-mlogloss:0.350672	test-mlogloss:0.709660
# [225]	train-mlogloss:0.349692	test-mlogloss:0.709471
# [226]	train-mlogloss:0.348580	test-mlogloss:0.709369
# [227]	train-mlogloss:0.347593	test-mlogloss:0.709125
# [228]	train-mlogloss:0.346627	test-mlogloss:0.708938
# [229]	train-mlogloss:0.345841	test-mlogloss:0.708731
# [230]	train-mlogloss:0.344794	test-mlogloss:0.708517
# [231]	train-mlogloss:0.343909	test-mlogloss:0.708340
# [232]	train-mlogloss:0.342948	test-mlogloss:0.708135
# [233]	train-mlogloss:0.341878	test-mlogloss:0.707961
# [234]	train-mlogloss:0.340957	test-mlogloss:0.707711
# [235]	train-mlogloss:0.339964	test-mlogloss:0.707514
# [236]	train-mlogloss:0.338990	test-mlogloss:0.707337
# [237]	train-mlogloss:0.337975	test-mlogloss:0.707115
# [238]	train-mlogloss:0.337023	test-mlogloss:0.707037
# [239]	train-mlogloss:0.336060	test-mlogloss:0.706941
# [240]	train-mlogloss:0.335106	test-mlogloss:0.706837
# [241]	train-mlogloss:0.334146	test-mlogloss:0.706551
# [242]	train-mlogloss:0.333265	test-mlogloss:0.706376
# [243]	train-mlogloss:0.332363	test-mlogloss:0.706245
# [244]	train-mlogloss:0.331481	test-mlogloss:0.706086
# [245]	train-mlogloss:0.330597	test-mlogloss:0.705921
# [246]	train-mlogloss:0.329724	test-mlogloss:0.705750
# [247]	train-mlogloss:0.328751	test-mlogloss:0.705604
# [248]	train-mlogloss:0.327960	test-mlogloss:0.705528
# [249]	train-mlogloss:0.327172	test-mlogloss:0.705375
# [250]	train-mlogloss:0.326300	test-mlogloss:0.705332
# [251]	train-mlogloss:0.325466	test-mlogloss:0.705184
# [252]	train-mlogloss:0.324439	test-mlogloss:0.704927
# [253]	train-mlogloss:0.323516	test-mlogloss:0.704693
# [254]	train-mlogloss:0.322464	test-mlogloss:0.704466
# [255]	train-mlogloss:0.321626	test-mlogloss:0.704357
# [256]	train-mlogloss:0.320773	test-mlogloss:0.704368
# [257]	train-mlogloss:0.319797	test-mlogloss:0.704208
# [258]	train-mlogloss:0.318813	test-mlogloss:0.704043
# [259]	train-mlogloss:0.317777	test-mlogloss:0.703873
# [260]	train-mlogloss:0.316947	test-mlogloss:0.703755
# [261]	train-mlogloss:0.316118	test-mlogloss:0.703616
# [262]	train-mlogloss:0.315127	test-mlogloss:0.703395
# [263]	train-mlogloss:0.314340	test-mlogloss:0.703433
# [264]	train-mlogloss:0.313465	test-mlogloss:0.703287
# [265]	train-mlogloss:0.312536	test-mlogloss:0.703216
# [266]	train-mlogloss:0.311676	test-mlogloss:0.703136
# [267]	train-mlogloss:0.310831	test-mlogloss:0.703094
# [268]	train-mlogloss:0.310012	test-mlogloss:0.702957
# [269]	train-mlogloss:0.309288	test-mlogloss:0.702816
# [270]	train-mlogloss:0.308405	test-mlogloss:0.702723
# [271]	train-mlogloss:0.307505	test-mlogloss:0.702649
# [272]	train-mlogloss:0.306627	test-mlogloss:0.702532
# [273]	train-mlogloss:0.305780	test-mlogloss:0.702348
# [274]	train-mlogloss:0.304999	test-mlogloss:0.702272
# [275]	train-mlogloss:0.304230	test-mlogloss:0.702184
# [276]	train-mlogloss:0.303407	test-mlogloss:0.702004
# [277]	train-mlogloss:0.302539	test-mlogloss:0.701922
# [278]	train-mlogloss:0.301765	test-mlogloss:0.701864
# [279]	train-mlogloss:0.301011	test-mlogloss:0.701761
# [280]	train-mlogloss:0.300195	test-mlogloss:0.701685
# [281]	train-mlogloss:0.299401	test-mlogloss:0.701574
# [282]	train-mlogloss:0.298572	test-mlogloss:0.701457
# [283]	train-mlogloss:0.297777	test-mlogloss:0.701315
# [284]	train-mlogloss:0.296996	test-mlogloss:0.701311
# [285]	train-mlogloss:0.296139	test-mlogloss:0.701164
# [286]	train-mlogloss:0.295344	test-mlogloss:0.701093
# [287]	train-mlogloss:0.294488	test-mlogloss:0.700916
# [288]	train-mlogloss:0.293688	test-mlogloss:0.700717
# [289]	train-mlogloss:0.292894	test-mlogloss:0.700572
# [290]	train-mlogloss:0.292105	test-mlogloss:0.700531
# [291]	train-mlogloss:0.291299	test-mlogloss:0.700349
# [292]	train-mlogloss:0.290592	test-mlogloss:0.700184
# [293]	train-mlogloss:0.289812	test-mlogloss:0.700229
# [294]	train-mlogloss:0.288964	test-mlogloss:0.700050
# [295]	train-mlogloss:0.288212	test-mlogloss:0.699973
# [296]	train-mlogloss:0.287488	test-mlogloss:0.699962
# [297]	train-mlogloss:0.286798	test-mlogloss:0.699850
# [298]	train-mlogloss:0.286020	test-mlogloss:0.699828
# [299]	train-mlogloss:0.285248	test-mlogloss:0.699670
# [300]	train-mlogloss:0.284523	test-mlogloss:0.699647
# [301]	train-mlogloss:0.283771	test-mlogloss:0.699559
# [302]	train-mlogloss:0.282910	test-mlogloss:0.699507
# [303]	train-mlogloss:0.282109	test-mlogloss:0.699460
# [304]	train-mlogloss:0.281337	test-mlogloss:0.699250
# [305]	train-mlogloss:0.280573	test-mlogloss:0.699154
# [306]	train-mlogloss:0.279906	test-mlogloss:0.698991
# [307]	train-mlogloss:0.279262	test-mlogloss:0.698886
# [308]	train-mlogloss:0.278530	test-mlogloss:0.698748
# [309]	train-mlogloss:0.277758	test-mlogloss:0.698657
# [310]	train-mlogloss:0.277054	test-mlogloss:0.698512
# [311]	train-mlogloss:0.276346	test-mlogloss:0.698324
# [312]	train-mlogloss:0.275631	test-mlogloss:0.698190
# [313]	train-mlogloss:0.274987	test-mlogloss:0.698132
# [314]	train-mlogloss:0.274334	test-mlogloss:0.698099
# [315]	train-mlogloss:0.273661	test-mlogloss:0.698118
# [316]	train-mlogloss:0.272878	test-mlogloss:0.698111
# [317]	train-mlogloss:0.272142	test-mlogloss:0.698099
# [318]	train-mlogloss:0.271414	test-mlogloss:0.698086
# [319]	train-mlogloss:0.270656	test-mlogloss:0.698114
# [320]	train-mlogloss:0.269986	test-mlogloss:0.698090
# [321]	train-mlogloss:0.269322	test-mlogloss:0.697948
# [322]	train-mlogloss:0.268781	test-mlogloss:0.697909
# [323]	train-mlogloss:0.268067	test-mlogloss:0.697918
# [324]	train-mlogloss:0.267416	test-mlogloss:0.697819
# [325]	train-mlogloss:0.266676	test-mlogloss:0.697681
# [326]	train-mlogloss:0.265962	test-mlogloss:0.697592
# [327]	train-mlogloss:0.265343	test-mlogloss:0.697557
# [328]	train-mlogloss:0.264681	test-mlogloss:0.697467
# [329]	train-mlogloss:0.263985	test-mlogloss:0.697497
# [330]	train-mlogloss:0.263313	test-mlogloss:0.697417
# [331]	train-mlogloss:0.262669	test-mlogloss:0.697364
# [332]	train-mlogloss:0.261955	test-mlogloss:0.697237
# [333]	train-mlogloss:0.261351	test-mlogloss:0.697165
# [334]	train-mlogloss:0.260717	test-mlogloss:0.697114
# [335]	train-mlogloss:0.260036	test-mlogloss:0.697057
# [336]	train-mlogloss:0.259453	test-mlogloss:0.697043
# [337]	train-mlogloss:0.258936	test-mlogloss:0.696965
# [338]	train-mlogloss:0.258372	test-mlogloss:0.696941
# [339]	train-mlogloss:0.257719	test-mlogloss:0.696855
# [340]	train-mlogloss:0.257125	test-mlogloss:0.696837
# [341]	train-mlogloss:0.256487	test-mlogloss:0.696854
# [342]	train-mlogloss:0.255908	test-mlogloss:0.696760
# [343]	train-mlogloss:0.255185	test-mlogloss:0.696771
# [344]	train-mlogloss:0.254558	test-mlogloss:0.696835
# [345]	train-mlogloss:0.253870	test-mlogloss:0.696793
# [346]	train-mlogloss:0.253210	test-mlogloss:0.696651
# [347]	train-mlogloss:0.252540	test-mlogloss:0.696654
# [348]	train-mlogloss:0.251955	test-mlogloss:0.696625
# [349]	train-mlogloss:0.251273	test-mlogloss:0.696609
# [350]	train-mlogloss:0.250626	test-mlogloss:0.696632
# [351]	train-mlogloss:0.249977	test-mlogloss:0.696560
# [352]	train-mlogloss:0.249459	test-mlogloss:0.696554
# [353]	train-mlogloss:0.248796	test-mlogloss:0.696609
# [354]	train-mlogloss:0.248220	test-mlogloss:0.696503
# [355]	train-mlogloss:0.247605	test-mlogloss:0.696449
# [356]	train-mlogloss:0.246986	test-mlogloss:0.696517
# [357]	train-mlogloss:0.246355	test-mlogloss:0.696515
# [358]	train-mlogloss:0.245636	test-mlogloss:0.696383
# [359]	train-mlogloss:0.245056	test-mlogloss:0.696380
# [360]	train-mlogloss:0.244475	test-mlogloss:0.696268
# [361]	train-mlogloss:0.243855	test-mlogloss:0.696201
# [362]	train-mlogloss:0.243243	test-mlogloss:0.696184
# [363]	train-mlogloss:0.242654	test-mlogloss:0.696065
# [364]	train-mlogloss:0.242063	test-mlogloss:0.695938
# [365]	train-mlogloss:0.241496	test-mlogloss:0.695830
# [366]	train-mlogloss:0.240939	test-mlogloss:0.695689
# [367]	train-mlogloss:0.240327	test-mlogloss:0.695566
# [368]	train-mlogloss:0.239762	test-mlogloss:0.695514
# [369]	train-mlogloss:0.239256	test-mlogloss:0.695477
# [370]	train-mlogloss:0.238619	test-mlogloss:0.695462
# [371]	train-mlogloss:0.238075	test-mlogloss:0.695510
# [372]	train-mlogloss:0.237520	test-mlogloss:0.695515
# [373]	train-mlogloss:0.236978	test-mlogloss:0.695558
# [374]	train-mlogloss:0.236473	test-mlogloss:0.695487
# [375]	train-mlogloss:0.235951	test-mlogloss:0.695437
# [376]	train-mlogloss:0.235385	test-mlogloss:0.695356
# [377]	train-mlogloss:0.234778	test-mlogloss:0.695367
# [378]	train-mlogloss:0.234206	test-mlogloss:0.695296
# [379]	train-mlogloss:0.233544	test-mlogloss:0.695250
# [380]	train-mlogloss:0.233031	test-mlogloss:0.695280
# [381]	train-mlogloss:0.232493	test-mlogloss:0.695208
# [382]	train-mlogloss:0.231929	test-mlogloss:0.695230
# [383]	train-mlogloss:0.231442	test-mlogloss:0.695158
# [384]	train-mlogloss:0.230893	test-mlogloss:0.695203
# [385]	train-mlogloss:0.230346	test-mlogloss:0.695205
# [386]	train-mlogloss:0.229749	test-mlogloss:0.695154
# [387]	train-mlogloss:0.229299	test-mlogloss:0.695242
# [388]	train-mlogloss:0.228819	test-mlogloss:0.695236
# [389]	train-mlogloss:0.228219	test-mlogloss:0.695160
# [390]	train-mlogloss:0.227705	test-mlogloss:0.695119
# [391]	train-mlogloss:0.227180	test-mlogloss:0.695101
# [392]	train-mlogloss:0.226635	test-mlogloss:0.695044
# [393]	train-mlogloss:0.226085	test-mlogloss:0.695001
# [394]	train-mlogloss:0.225658	test-mlogloss:0.694921
# [395]	train-mlogloss:0.225131	test-mlogloss:0.694959
# [396]	train-mlogloss:0.224649	test-mlogloss:0.695015
# [397]	train-mlogloss:0.224191	test-mlogloss:0.695070
# [398]	train-mlogloss:0.223685	test-mlogloss:0.695044
# [399]	train-mlogloss:0.223077	test-mlogloss:0.694937
# [400]	train-mlogloss:0.222508	test-mlogloss:0.695039
# [401]	train-mlogloss:0.221950	test-mlogloss:0.694967
# [402]	train-mlogloss:0.221484	test-mlogloss:0.695046
# [403]	train-mlogloss:0.220911	test-mlogloss:0.695077
# [404]	train-mlogloss:0.220411	test-mlogloss:0.695155
# [405]	train-mlogloss:0.219888	test-mlogloss:0.695175
# [406]	train-mlogloss:0.219378	test-mlogloss:0.695168
# [407]	train-mlogloss:0.218872	test-mlogloss:0.695186
# [408]	train-mlogloss:0.218268	test-mlogloss:0.695147
# [409]	train-mlogloss:0.217679	test-mlogloss:0.695126
# [410]	train-mlogloss:0.217162	test-mlogloss:0.695148
# [411]	train-mlogloss:0.216648	test-mlogloss:0.695211
# [412]	train-mlogloss:0.216142	test-mlogloss:0.695238
