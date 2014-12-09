import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.stats
import os

for prefix in ("results",):
    for infix in ("with", "without"):
        with open(os.path.join(prefix, "results_{}_hmi.json".format(infix))) as f:
            results = json.load(f)

        plt.figure()
        plt.title("Performance")
        plt.boxplot([results["tss"][alg_name] for alg_name in sorted(results["tss"].keys())])
        plt.xticks([1, 2, 3, 4], sorted(results["tss"].keys()))
        plt.xlabel("Classifier")
        plt.ylabel("TSS")
        plt.ylim([-1,1])
        plt.savefig(os.path.join(prefix, "tss_{}_hmi.pdf".format(infix)))

        colors = ["r", "g", "c", "m"]

        plt.figure(figsize=(16,16))
        for i, alg_name in enumerate(sorted(results["fpr"].keys())):
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
        plt.savefig(os.path.join(prefix, "fpr_vs_tpr_{}_hmi.pdf".format(infix)))
