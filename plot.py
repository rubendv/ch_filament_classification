import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.stats

with open("results/results_without_hmi.json") as f:
    results = json.load(f)

plt.figure()
plt.title("Performance")
plt.boxplot(results["tss"].values())
plt.xticks([1, 2, 3, 4], results["tss"].keys())
plt.xlabel("Classifier")
plt.ylabel("TSS")
plt.ylim([0,1])
plt.savefig("tss_without_hmi.pdf")

colors = ["r", "g", "c", "m"]

plt.figure(figsize=(16,16))
for i, alg_name in enumerate(results["fpr"].keys()):
    ax = plt.subplot(2, 2, i+1)
    plt.title("{}".format(alg_name))
    tpr = np.array(results["tpr"][alg_name])
    fpr = np.array(results["fpr"][alg_name])
    X, Y = np.mgrid[0:1:128j,0:1:128j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([tpr, fpr])
    print values.shape, positions.shape
    kernel = scipy.stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    plt.imshow(Z, origin="lower", extent=(0, 1, 0, 1), cmap="gray")
    plt.plot(fpr, tpr, colors[i]+'+', label=alg_name)
    plt.plot([0, 1], [0, 1], color="w", linestyle='-')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
plt.savefig("fpr_vs_tpr_without_hmi.pdf")
plt.show()
