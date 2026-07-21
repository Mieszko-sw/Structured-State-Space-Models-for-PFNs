import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load data from text block
data_str = """
  did                      dataset_name         model  num_samples  num_features  eval_position      mean     median        std        min        max  count  mean_metric
   11                     balance-scale hybrid_8l          625             4            972 0.038379 0.037995 0.001448 0.036809 0.041655     10     0.995021
   11                     balance-scale     hydra          625             4            972 0.046002 0.045420 0.001468 0.044583 0.048698     10     0.988137
   11                     balance-scale    tabpfn          625             4            972 0.044140 0.044040 0.001003 0.043024 0.046621     10     0.998240
   14                     mfeat-fourier hybrid_8l         2000            76            972 0.344327 0.344559 0.005945 0.336523 0.358572     10     0.973904
   14                     mfeat-fourier     hydra         2000            76            972 0.365584 0.366942 0.008171 0.349608 0.376054     10     0.960277
   14                     mfeat-fourier    tabpfn         2000            76            972 0.359691 0.359550 0.008851 0.347010 0.376945     10     0.975816
   15                          breast-w hybrid_8l          683             9            972 0.053085 0.053112 0.001046 0.050807 0.054581     10     0.993802
   15                          breast-w     hydra          683             9            972 0.060190 0.060203 0.001051 0.058740 0.062239     10     0.994672
   15                          breast-w    tabpfn          683             9            972 0.060232 0.060295 0.001152 0.057937 0.061768     10     0.994680
   16                     mfeat-karhunen hybrid_8l         2000            64            972 0.315890 0.295992 0.064565 0.292539 0.499543     10     0.996645
   16                     mfeat-karhunen     hydra         2000            64            972 0.340498 0.319531 0.065063 0.316744 0.525464     10     0.989891
   16                     mfeat-karhunen    tabpfn         2000            64            972 0.337879 0.311697 0.067404 0.308725 0.526878     10     0.996786
   18               mfeat-morphological hybrid_8l         2000             6            972 0.095419 0.094160 0.003136 0.093222 0.102000     10     0.964042
   18               mfeat-morphological     hydra         2000             6            972 0.131244 0.113736 0.028866 0.113036 0.182100     10     0.958057
   18               mfeat-morphological    tabpfn         2000             6            972 0.108694 0.107459 0.003447 0.106581 0.117834     10     0.963835
   22                      mfeat-zernike hybrid_8l         2000            47            972 0.253762 0.249715 0.015642 0.239990 0.295268     10     0.980661
   22                      mfeat-zernike     hydra         2000            47            972 0.269118 0.272459 0.006475 0.259032 0.276470     10     0.967980
   22                      mfeat-zernike    tabpfn         2000            47            972 0.266308 0.265739 0.009141 0.253226 0.284111     10     0.982242
   23                                cmc hybrid_8l         1473             9            972 0.068450 0.064764 0.011844 0.062737 0.101954     10     0.706750
   23                                cmc     hydra         1473             9            972 0.081551 0.076752 0.015672 0.074645 0.126049     10     0.693118
   23                                cmc    tabpfn         1473             9            972 0.080083 0.074336 0.014003 0.072526 0.118377     10     0.712318
   29                    credit-approval hybrid_8l          653            15            972 0.095052 0.103977 0.018082 0.068947 0.111933     10     0.926676
   29                    credit-approval     hydra          653            15            972 0.097430 0.106395 0.018872 0.073638 0.118307     10     0.930668
   29                    credit-approval    tabpfn          653            15            972 0.097035 0.107614 0.017172 0.075197 0.114003     10     0.933494
   31                           credit-g hybrid_8l         1000            20            972 0.113924 0.110477 0.018807 0.095222 0.136124     10     0.792168
   31                           credit-g     hydra         1000            20            972 0.120792 0.106413 0.020244 0.104089 0.146844     10     0.779825
   31                           credit-g    tabpfn         1000            20            972 0.116957 0.105895 0.018131 0.104591 0.144861     10     0.792060
   37                           diabetes hybrid_8l          768             8            972 0.063069 0.052114 0.019403 0.049304 0.101327     10     0.825548
   37                           diabetes     hydra          768             8            972 0.070621 0.059828 0.017915 0.056595 0.098792     10     0.834808
   37                           diabetes    tabpfn          768             8            972 0.072983 0.059867 0.018507 0.056684 0.097316     10     0.835945
   50                        tic-tac-toe hybrid_8l          958             9            972 0.054201 0.054051 0.001487 0.052234 0.056770     10     0.964488
   50                        tic-tac-toe     hydra          958             9            972 0.062542 0.062713 0.002059 0.059617 0.065434     10     0.801097
   50                        tic-tac-toe    tabpfn          958             9            972 0.062445 0.062453 0.001915 0.059627 0.065843     10     0.972544
   54                            vehicle hybrid_8l          846            18            972 0.099099 0.099363 0.001173 0.097087 0.100656     10     0.950290
   54                            vehicle     hydra          846            18            972 0.120278 0.114575 0.015886 0.112143 0.164613     10     0.927589
   54                            vehicle    tabpfn          846            18            972 0.110654 0.111031 0.001772 0.107699 0.112994     10     0.958254
  188                          eucalyptus hybrid_8l          641            19            972 0.101197 0.100998 0.001066 0.099814 0.102914     10     0.910933
  188                          eucalyptus     hydra          641            19            972 0.117449 0.116733 0.002550 0.114827 0.122460     10     0.901261
  188                          eucalyptus    tabpfn          641            19            972 0.111981 0.111885 0.000911 0.110548 0.113496     10     0.914725
  458              analcatdata_authorship hybrid_8l          841            70            972 0.310370 0.284397 0.041267 0.277124 0.374162     10     0.999993
  458              analcatdata_authorship     hydra          841            70            972 0.344083 0.342879 0.049534 0.292964 0.393931     10     0.999914
  458              analcatdata_authorship    tabpfn          841            70            972 0.334138 0.318149 0.044947 0.289348 0.391874     10     0.999995
  469                    analcatdata_dmft hybrid_8l          797             4            972 0.062337 0.061996 0.001951 0.060393 0.067466     10     0.581094
  469                    analcatdata_dmft     hydra          797             4            972 0.077262 0.077365 0.000732 0.075757 0.078267     10     0.565801
  469                    analcatdata_dmft    tabpfn          797             4            972 0.074225 0.073261 0.002639 0.072537 0.081453     10     0.572532
 1049                                 pc4 hybrid_8l         1458            37            972 0.161193 0.159798 0.005030 0.156866 0.174284     10     0.929582
 1049                                 pc4     hydra         1458            37            972 0.182652 0.171151 0.032539 0.165419 0.272127     10     0.914074
 1049                                 pc4    tabpfn         1458            37            972 0.171312 0.168231 0.008858 0.163058 0.194196     10     0.931262
 1050                                 pc3 hybrid_8l         1563            37            972 0.164742 0.165925 0.004742 0.157824 0.174166     10     0.826344
 1050                                 pc3     hydra         1563            37            972 0.176899 0.174066 0.009077 0.167064 0.195055     10     0.811030
 1050                                 pc3    tabpfn         1563            37            972 0.177451 0.173036 0.015427 0.167748 0.219748     10     0.825854
 1063                                 kc2 hybrid_8l          522            21            972 0.087125 0.087246 0.001728 0.084530 0.089432     10     0.806450
 1063                                 kc2     hydra          522            21            972 0.092178 0.092216 0.001779 0.089184 0.094741     10     0.845177
 1063                                 kc2    tabpfn          522            21            972 0.092468 0.092807 0.001844 0.089202 0.095019     10     0.849974
 1068                                 pc1 hybrid_8l         1109            21            972 0.106438 0.102487 0.011246 0.100413 0.137595     10     0.880654
 1068                                 pc1     hydra         1109            21            972 0.111803 0.111106 0.001994 0.109690 0.115192     10     0.821710
 1068                                 pc1    tabpfn         1109            21            972 0.112534 0.112266 0.001933 0.109912 0.116021     10     0.880834
 1462          banknote-authentication hybrid_8l         1372             4            972 0.039051 0.038785 0.001034 0.038165 0.041675     10     1.000000
 1462          banknote-authentication     hydra         1372             4            972 0.046603 0.046238 0.001384 0.045604 0.050158     10     1.000000
 1462          banknote-authentication    tabpfn         1372             4            972 0.046769 0.046439 0.001474 0.045745 0.050727     10     1.000000
 1464 blood-transfusion-service-center hybrid_8l          748             4            972 0.034624 0.034710 0.000739 0.033553 0.035493     10     0.749097
 1464 blood-transfusion-service-center     hydra          748             4            972 0.040788 0.040878 0.000832 0.039052 0.042138     10     0.763685
 1464 blood-transfusion-service-center    tabpfn          748             4            972 0.040510 0.040613 0.000732 0.039476 0.041335     10     0.759915
 1480                                ilpd hybrid_8l          583            10            972 0.049789 0.049744 0.000602 0.048717 0.050966     10     0.729350
 1480                                ilpd     hydra          583            10            972 0.055806 0.055717 0.000771 0.054322 0.057090     10     0.740224
 1480                                ilpd    tabpfn          583            10            972 0.054081 0.054224 0.000465 0.052978 0.054668     10     0.740725
 1494                      qsar-biodeg hybrid_8l         1055            41            972 0.176263 0.175477 0.003360 0.172728 0.184689     10     0.931395
 1494                      qsar-biodeg     hydra         1055            41            972 0.184527 0.183904 0.003204 0.180799 0.191346     10     0.913201
 1494                      qsar-biodeg    tabpfn         1055            41            972 0.184866 0.184196 0.002993 0.181530 0.191307     10     0.934326
 1510                                wdbc hybrid_8l          569            30            972 0.121544 0.121875 0.001470 0.119109 0.123254     10     0.995753
 1510                                wdbc     hydra          569            30            972 0.127188 0.127604 0.001159 0.125618 0.128618     10     0.995329
 1510                                wdbc    tabpfn          569            30            972 0.127418 0.127249 0.001811 0.125413 0.131414     10     0.995813
 6332                   cylinder-bands hybrid_8l          277            37            972 0.115772 0.115186 0.003677 0.110189 0.122203     10     0.790760
 6332                   cylinder-bands     hydra          277            37            972 0.123031 0.123210 0.003394 0.117200 0.128663     10     0.748485
 6332                   cylinder-bands    tabpfn          277            37            972 0.116566 0.116428 0.003155 0.111170 0.122626     10     0.781278
23381                    dresses-sales hybrid_8l           99            12            972 0.049757 0.049730 0.000945 0.048472 0.051287     10     0.484916
23381                    dresses-sales     hydra           99            12            972 0.056689 0.056698 0.001257 0.055065 0.059233     10     0.475014
23381                    dresses-sales    tabpfn           99            12            972 0.050259 0.049701 0.002170 0.047703 0.055337     10     0.468156
40966                      MiceProtein hybrid_8l          552            77            972 0.296730 0.288666 0.021052 0.280004 0.352258     10     1.000000
40966                      MiceProtein     hydra          552            77            972 0.311564 0.301484 0.024148 0.293283 0.370688     10     0.997710
40966                      MiceProtein    tabpfn          552            77            972 0.301547 0.299668 0.008647 0.291425 0.323773     10     0.999989
40975                              car hybrid_8l         1728             6            972 0.067485 0.064709 0.008314 0.063350 0.090548     10     0.983252
40975                              car     hydra         1728             6            972 0.080957 0.079807 0.003114 0.078861 0.089169     10     0.944707
40975                              car    tabpfn         1728             6            972 0.077728 0.076648 0.002728 0.076155 0.085086     10     0.986350
40982               steel-plates-fault hybrid_8l         1941            27            972 0.154490 0.152642 0.005606 0.149260 0.168192     10     0.951551
40982               steel-plates-fault     hydra         1941            27            972 0.179913 0.173742 0.018866 0.168854 0.231340     10     0.920159
40982               steel-plates-fault    tabpfn         1941            27            972 0.171197 0.164778 0.020938 0.162326 0.230615     10     0.954483
40994 climate-model-simulation-crashes hybrid_8l          540            18            972 0.097656 0.106682 0.017407 0.074144 0.115217     10     0.941784
40994 climate-model-simulation-crashes     hydra          540            18            972 0.111761 0.117968 0.017268 0.080011 0.130311     10     0.934011
40994 climate-model-simulation-crashes    tabpfn          540            18            972 0.108940 0.118164 0.018730 0.080009 0.125901     10     0.943450
"""

# 2. Parse into Pandas Dataframe
df = pd.read_csv(io.StringIO(data_str.strip()), sep=r"\s+")

# Pivot data so each row is a unique Dataset ID and columns represent model metrics
pivot_df = df.pivot(
    index="did", columns="model", values=["mean", "mean_metric"]
)
pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
pivot_df = pivot_df.reset_index()

# Sort by numeric dataset ID for cleaner layout formatting
pivot_df = pivot_df.sort_values(by="did").reset_index(drop=True)

# 3. Calculate Deltas relative to TabPFN baseline
# For accuracy/metric: positive is better than tabpfn
pivot_df["hybrid_8l_score_delta"] = (
    pivot_df["hybrid_8l_mean_metric"] - pivot_df["tabpfn_mean_metric"]
)
pivot_df["hydra_score_delta"] = (
    pivot_df["hydra_mean_metric"] - pivot_df["tabpfn_mean_metric"]
)

# For time: positive means FASTER (lower mean execution latency) than tabpfn
pivot_df["hybrid_8l_speedup"] = (
    (pivot_df["tabpfn_mean"] - pivot_df["hybrid_8l_mean"])
    / pivot_df["tabpfn_mean"]
) * 100
pivot_df["hydra_speedup"] = (
    (pivot_df["tabpfn_mean"] - pivot_df["hydra_mean"]) / pivot_df["tabpfn_mean"]
) * 100

# 4. Set up Plot Constants
num_datasets = len(pivot_df)
x = np.arange(num_datasets)
width = 0.35  # Stripe width grouping

# Color choices
c_hybrid = "#2b5c8f"  # Blue shade
c_hydra = "#d95f02"  # Orange shade

# Create stacked visual canvas sharing the exact same X indices
fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1, figsize=(14, 10), sharex=True, dpi=100
)

# --- TOP SUBPLOT: Inference Footprint Change ---
rects1_time = ax1.bar(
    x - width / 2,
    pivot_df["hybrid_8l_speedup"],
    width,
    label="hybrid_8l",
    color=c_hybrid,
)
rects2_time = ax1.bar(
    x + width / 2,
    pivot_df["hydra_speedup"],
    width,
    label="hydra",
    color=c_hydra,
)

ax1.axhline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.7)
ax1.set_ylabel("Inference Speedup (% Faster than TabPFN)", fontsize=11)
ax1.set_title(
    "Benchmark Array Comparison: Deviations From TabPFN Baseline",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.legend(loc="upper right", frameon=True)

# --- BOTTOM SUBPLOT: Predictive Metric Shift ---
rects1_score = ax2.bar(
    x - width / 2,
    pivot_df["hybrid_8l_score_delta"],
    width,
    color=c_hybrid,
    alpha=0.9,
)
rects2_score = ax2.bar(
    x + width / 2,
    pivot_df["hydra_score_delta"],
    width,
    color=c_hydra,
    alpha=0.9,
)

ax2.axhline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.7)
ax2.set_ylabel("Predictive Score Deviation vs. TabPFN", fontsize=11)
ax2.set_xlabel("Dataset ID (did)", fontsize=12, labelpad=10)
ax2.grid(axis="y", linestyle="--", alpha=0.5)

# Coordinate clean text alignments mapping IDs directly onto shared base ticks
ax2.set_xticks(x)
ax2.set_xticklabels(pivot_df["did"].astype(str), rotation=45, ha="right")

plt.tight_layout()
plt.show()
plt.savefig("result_csvs/plot_res.png", dpi=300, bbox_inches="tight")


