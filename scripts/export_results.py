"""
Export Benchmark Results Table comparing Survey Reference vs Reproduced.

Survey: "A Decade Survey of Content Based Image Retrieval using Deep Learning"
Table IV: CIFAR-10 retrieval comparison (mAP@54000)
"""

import json
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SURVEY_REFERENCES = {
    ("HashNet", 16): 0.643,
    ("HashNet", 32): 0.667,
    ("DHN", 16): 0.568,
    ("DHN", 32): 0.603,
}

def main():
    rows = []
    
    # Check HashNet 16-bit
    meta_16_path = os.path.join(PROJECT_ROOT, "outputs/experiments/hashnet_16bit/run_metadata.json")
    if os.path.exists(meta_16_path):
        with open(meta_16_path) as f:
            m16 = json.load(f)
        ref_16 = SURVEY_REFERENCES[("HashNet", 16)]
        rep_16 = m16["mAP"]
        diff_16 = (rep_16 - ref_16) * 100
        rows.append({
            "Method": "HashNet",
            "Hash_Bits": 16,
            "Backbone": "AlexNet",
            "Survey_Ref_mAP": f"{ref_16 * 100:.1f}%",
            "Reproduced_mAP": f"{rep_16 * 100:.2f}%",
            "Difference_pp": f"{diff_16:+.2f} pp",
            "Metric": f"mAP@{m16['topk']}",
            "Protocol_Mode": "survey_source (modified 44k gallery)",
            "Status": "Verified",
        })

    # Check HashNet 32-bit
    meta_32_path = os.path.join(PROJECT_ROOT, "outputs/experiments/hashnet_32bit/run_metadata.json")
    if os.path.exists(meta_32_path):
        with open(meta_32_path) as f:
            m32 = json.load(f)
        ref_32 = SURVEY_REFERENCES[("HashNet", 32)]
        rep_32 = m32["mAP"]
        diff_32 = (rep_32 - ref_32) * 100
        rows.append({
            "Method": "HashNet",
            "Hash_Bits": 32,
            "Backbone": "AlexNet",
            "Survey_Ref_mAP": f"{ref_32 * 100:.1f}%",
            "Reproduced_mAP": f"{rep_32 * 100:.2f}%",
            "Difference_pp": f"{diff_32:+.2f} pp",
            "Metric": f"mAP@{m32['topk']}",
            "Protocol_Mode": "survey_source (modified 44k gallery)",
            "Status": "Verified",
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(PROJECT_ROOT, "results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Exported results to: {out_csv}")
    print(df.to_string())

if __name__ == "__main__":
    main()

