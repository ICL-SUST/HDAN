import os, glob, numpy as np


def audit_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.npy")))
    total = len(files)
    invalid = []
    report = []
    for f in files:
        try:
            arr = np.load(f)
            wm = int(np.sum(arr == 3))
            gm = int(np.sum(arr == 2))
            csf = int(np.sum(arr == 1))
            has_nan = np.isnan(arr).any()
            uniq = np.unique(arr)
            if (wm == 0 and gm == 0 and csf == 0) or has_nan:
                invalid.append((os.path.basename(f), wm, gm, csf, has_nan, arr.shape, uniq[:8]))
            report.append((os.path.basename(f), wm, gm, csf))
        except Exception as e:
            invalid.append((os.path.basename(f), "LOAD_ERROR", str(e), None, None, None, None))
    print(f"\nFolder: {folder}")
    print(f"Total .npy: {total}")
    print(f"Invalid (excluded by your logic): {len(invalid)}")
    for item in invalid:
        print("  ->", item)
    return total, invalid, report


audit_folder(r"F:\PretermInfant\ExperimentResults\PretermInfant")
audit_folder(r"F:\PretermInfant\ExperimentResults\TermInfant")
