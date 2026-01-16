import numpy as np
import os


def Estimation_WhiteMatterGrayMatter(name):
    infant_dict = {
        "Preterm Infant": r"F:\PretermInfant\ExperimentResults\PretermInfant",
        "Term Infant": r"F:\PretermInfant\ExperimentResults\TermInfant"
    }
    seg_dir = infant_dict[name]
    wm_list, gm_list, csf_list, bv_list, wm_ratio_list, gm_ratio_list = [], [], [], [], [], []
    num_valid = 0

    for item in os.listdir(seg_dir):
        if not item.endswith('.npy'): continue
        label_ndarray = np.load(os.path.join(seg_dir, item))
        wm = int(np.sum(label_ndarray == 3))
        gm = int(np.sum(label_ndarray == 2))
        csf = int(np.sum(label_ndarray == 1))
        bv = wm + gm
        if wm == 0 and gm == 0:
            continue
        wm_list.append(wm)
        gm_list.append(gm)
        csf_list.append(csf)
        bv_list.append(bv)
        if bv > 0:
            wm_ratio_list.append(wm / bv)
            gm_ratio_list.append(gm / bv)
        num_valid += 1

    if num_valid == 0:
        print(f"{name}: No valid samples after filtering!")
        return 0, 0

    average_whitematter = np.mean(wm_list)
    average_graymatter = np.mean(gm_list)
    average_csf = np.mean(csf_list)
    average_brain_volume = np.mean(bv_list)
    average_whitematter_ratio = np.mean(wm_ratio_list)
    average_graymatter_ratio = np.mean(gm_ratio_list)

    print("{}:The average White Matter is {:.6f} over {} samples".format(name, average_whitematter, num_valid))
    print("{}:The average Gray Matter is {:.6f} over {} samples".format(name, average_graymatter, num_valid))
    print("{}:The average Cerebrospinal Fluid is {:.6f} over {} samples".format(name, average_csf, num_valid))
    print("{}:The average Brain Volume is {:.6f} over {} samples".format(name, average_brain_volume, num_valid))
    print("{}:The average White Matter Ratio is {:.6f} over {} samples".format(name, average_whitematter_ratio, num_valid))
    print("{}:The average Gray Matter Ratio is {:.6f} over {} samples".format(name, average_graymatter_ratio, num_valid))
    return average_brain_volume


if __name__ == "__main__":
    preterm_brain_volume = Estimation_WhiteMatterGrayMatter("Preterm Infant")
    term_brain_volume = Estimation_WhiteMatterGrayMatter("Term Infant")

    if term_brain_volume > preterm_brain_volume:
        print("From the aspect of Brain Volume(White Matter+Gray Matter), Term infant is better than preterm infants!")
    else:
        print("From the aspect of Brain Volume(White Matter+Gray Matter), Term infant is worse than preterm infants!")
    print("Finish!")
