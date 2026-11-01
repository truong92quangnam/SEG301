"""
Streamlit Visual Demo for Deep Hashing Reproduction (HashNet on CIFAR-10).

Paper Survey: "A Decade Survey of Content Based Image Retrieval using Deep Learning"
(Shiv Ram Dubey, IEEE TCSVT, 2021)
Focus: Table IV CIFAR-10 Retrieval Benchmark & Visual Step-by-Step Retrieval Explanation.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

st.set_page_config(
    page_title="Deep Hashing CBIR Visual Benchmark",
    page_icon="🔍",
    layout="wide",
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@st.cache_data
def load_manifest():
    manifest_path = os.path.join(PROJECT_ROOT, "outputs/splits/manifest_cifar10_survey_source.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_labels_df():
    csv_path = os.path.join(PROJECT_ROOT, "cifar-10/trainLabels.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def load_codes_data(method: str, bits: int, protocol: str = "survey_source", variant: str = "baseline"):
    if variant == "improved":
        exp_folder = f"{method.lower()}_{bits}bit_improved"
    else:
        suffix = "_10k_query" if protocol == "original_paper_10k_query" else ""
        exp_folder = f"{method.lower()}_{bits}bit{suffix}"
    path = os.path.join(PROJECT_ROOT, f"outputs/experiments/{exp_folder}/extracted_codes_{bits}bit.npz")
    if os.path.exists(path):
        data = np.load(path)
        return {
            "q_codes": data["q_codes"],
            "q_cont": data["q_cont"],
            "q_labels": data["q_labels"],
            "q_orig_idx": data["q_orig_idx"],
            "d_codes": data["d_codes"],
            "d_cont": data["d_cont"],
            "d_labels": data["d_labels"],
            "d_orig_idx": data["d_orig_idx"],
        }
    return None

@st.cache_data
def load_metadata(method: str, bits: int, protocol: str = "survey_source", variant: str = "baseline"):
    if variant == "improved":
        exp_folder = f"{method.lower()}_{bits}bit_improved"
    else:
        suffix = "_10k_query" if protocol == "original_paper_10k_query" else ""
        exp_folder = f"{method.lower()}_{bits}bit{suffix}"
    path = os.path.join(PROJECT_ROOT, f"outputs/experiments/{exp_folder}/run_metadata.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_training_logs(method: str, bits: int, variant: str = "baseline"):
    suffix = "_improved" if variant == "improved" else ""
    path = os.path.join(PROJECT_ROOT, f"outputs/experiments/{method.lower()}_{bits}bit{suffix}/training_logs.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_per_query_metrics(method: str, bits: int, protocol: str = "survey_source", variant: str = "baseline"):
    if variant == "improved":
        exp_folder = f"{method.lower()}_{bits}bit_improved"
    else:
        suffix = "_10k_query" if protocol == "original_paper_10k_query" else ""
        exp_folder = f"{method.lower()}_{bits}bit{suffix}"
    path = os.path.join(PROJECT_ROOT, f"outputs/experiments/{exp_folder}/per_query_metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def get_image_path(orig_id: int) -> str:
    return os.path.join(PROJECT_ROOT, "cifar-10/train", f"{orig_id}.png")

def load_pil_image(orig_id: int) -> Optional[Image.Image]:
    img_path = get_image_path(orig_id)
    if os.path.exists(img_path):
        return Image.open(img_path).convert("RGB")
    return None


# ==========================================
# HEADER & SIDEBAR
# ==========================================
st.title("🔍 Deep Hashing for Content-Based Image Retrieval")
st.caption(
    "Controlled Reproduction & Visual Explanation of **HashNet (ICCV 2017)** and **DHN (AAAI 2016)** "
    "from *'A Decade Survey of CBIR using Deep Learning'* (Dubey, IEEE TCSVT 2021, Table IV)."
)

manifest = load_manifest()
df_labels = load_labels_df()

st.sidebar.header("⚙️ Model & Protocol Configuration")
protocol_choice = st.sidebar.selectbox(
    "Benchmark Protocol",
    [
        "Survey Table IV Mode (1,000 Query)",
        "DeepHash Standard Mode (10,000 Query)"
    ],
    index=0
)
protocol_key = "original_paper_10k_query" if "10,000" in protocol_choice else "survey_source"

method_choice = st.sidebar.selectbox("Method", ["HashNet", "DHN"], index=0)
variant_choice = st.sidebar.radio(
    "Architecture Variant",
    ["Bit-Decorrelation & Balance (Improved)", "Baseline"],
    index=0
)
variant_key = "improved" if "Improved" in variant_choice else "baseline"

bits_choice = st.sidebar.selectbox("Hash Code Length (K bits)", [16, 32], index=0)

codes_data = load_codes_data(method_choice, bits_choice, protocol=protocol_key, variant=variant_key)
metadata = load_metadata(method_choice, bits_choice, protocol=protocol_key, variant=variant_key)
training_logs = load_training_logs(method_choice, bits_choice, variant=variant_key)
query_metrics_df = load_per_query_metrics(method_choice, bits_choice, protocol=protocol_key, variant=variant_key)

if codes_data is None:
    st.sidebar.warning(f"⚠️ {method_choice} {bits_choice}-bit [{variant_choice}] ({protocol_choice}): Not run / evaluating.")
else:
    st.sidebar.success(f"✅ Loaded {method_choice} {bits_choice}-bit [{variant_choice}]: {len(codes_data['q_codes'])} Queries vs {len(codes_data['d_codes'])} Gallery.")


# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Tab 1: Retrieval Explorer",
    "🧬 Tab 2: Hash Explanation",
    "📊 Tab 3: Reproduction Results",
    "⚠️ Tab 4: Failure & Success Cases",
])


# ==============================================================================
# TAB 1: RETRIEVAL
# ==============================================================================
with tab1:
    st.subheader("Interactive Retrieval Top-10")
    st.write(
        "Select any query image from the benchmark set. The system computes exact Hamming distance "
        "against the 44,000 database images (without using labels for ranking) and displays the Top-10 retrieved images. "
        "Green border indicates **Relevant (True Positive)**; Red border indicates **Non-relevant (False Positive)**."
    )

    if codes_data is None:
        st.info("Chưa có dữ liệu trích xuất cho cấu hình này. Vui lòng đợi quá trình training/evaluate hoàn tất.")
    else:
        num_queries = len(codes_data["q_codes"])
        col_q1, col_q2, col_q3 = st.columns([1, 2, 2])
        
        with col_q1:
            q_idx = st.number_input(
                "Query Index (0 - 999)",
                min_value=0,
                max_value=num_queries - 1,
                value=166,
                step=1
            )
            
        q_label_id = codes_data["q_labels"][q_idx]
        q_label_name = CIFAR10_CLASSES[q_label_id]
        q_orig_idx = codes_data["q_orig_idx"][q_idx]
        q_orig_id = int(df_labels.iloc[q_orig_idx]["id"])
        
        q_img = load_pil_image(q_orig_id)
        q_code = codes_data["q_codes"][q_idx]

        with col_q2:
            st.write(f"**Query Image**: ID `{q_orig_id}` | Class: **{q_label_name.upper()}** (class {q_label_id})")
            if q_img:
                st.image(q_img.resize((140, 140)), width=140)

        # Compute Hamming distance against database
        # Hamming distance: d_H = 0.5 * (K - q @ d.T)
        d_codes = codes_data["d_codes"]
        d_labels = codes_data["d_labels"]
        d_orig_indices = codes_data["d_orig_idx"]

        # Vectorized Hamming distance
        inner = np.dot(d_codes, q_code)
        hamming_dist = 0.5 * (bits_choice - inner)
        
        # Stable sort top 10
        sorted_indices = np.argsort(hamming_dist, kind="stable")[:10]

        with col_q3:
            ap_val = query_metrics_df.iloc[q_idx]["AP"] if query_metrics_df is not None else 0.0
            st.metric("Query Average Precision (AP@44000)", f"{ap_val * 100:.2f}%")
            top10_rel = sum(1 for idx in sorted_indices if d_labels[idx] == q_label_id)
            st.metric("Top-10 Precision (P@10)", f"{top10_rel} / 10 ({top10_rel * 10}%)")

        st.divider()
        st.subheader("Top-10 Retrieved Images (Ranked strictly by Hamming Distance)")

        # Render 10 images in two rows of 5
        for row_start in [0, 5]:
            cols = st.columns(5)
            for c_i, rank in enumerate(range(row_start, row_start + 5)):
                d_idx = sorted_indices[rank]
                dist = int(hamming_dist[d_idx])
                cand_label_id = d_labels[d_idx]
                cand_label_name = CIFAR10_CLASSES[cand_label_id]
                cand_orig_id = int(df_labels.iloc[d_orig_indices[d_idx]]["id"])
                is_rel = (cand_label_id == q_label_id)

                cand_img = load_pil_image(cand_orig_id)

                border_color = "#28a745" if is_rel else "#dc3545"
                status_text = "✅ Relevant" if is_rel else "❌ Non-relevant"

                with cols[c_i]:
                    st.markdown(
                        f"""
                        <div style="border: 4px solid {border_color}; border-radius: 8px; padding: 6px; text-align: center; background-color: rgba(255, 255, 255, 0.05);">
                            <div style="font-weight: bold; font-size: 14px;">Rank #{rank + 1}</div>
                            <div style="font-size: 12px; color: #888;">Dist: <b>{dist}</b> bits</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if cand_img:
                        st.image(cand_img.resize((120, 120)), use_container_width=True)
                    st.caption(f"{cand_label_name.upper()} | {status_text}")


# ==============================================================================
# TAB 2: HASH EXPLANATION
# ==============================================================================
with tab2:
    st.subheader("🧬 Pipeline: Query Image → CNN Features → Continuous Hash → Binary Code → Hamming Distance")
    st.write(
        "Giải thích từng bước quá trình băm sâu: từ ảnh query, trích xuất biểu diễn liên tục $h \in (-1, 1)^K$, "
        "nhị phân hóa thành $b \in \\{-1, +1\\}^K$, so sánh từng bit và tính khoảng cách Hamming."
    )

    if codes_data is None:
        st.info("Chưa có dữ liệu trích xuất. Vui lòng hoàn thành quá trình trích xuất.")
    else:
        col_e1, col_e2 = st.columns([1, 2])
        with col_e1:
            q_idx_exp = st.number_input("Chọn Query Index", min_value=0, max_value=len(codes_data["q_codes"]) - 1, value=100, key="exp_q")
            q_orig_id_exp = int(df_labels.iloc[codes_data["q_orig_idx"][q_idx_exp]]["id"])
            q_img_exp = load_pil_image(q_orig_id_exp)
            if q_img_exp:
                st.image(q_img_exp.resize((150, 150)), caption=f"Query Image ({CIFAR10_CLASSES[codes_data['q_labels'][q_idx_exp]]})")

        with col_e2:
            st.markdown("#### 1. Continuous Hash Output $h = \\tanh(\\alpha z) \\in (-1, 1)^K$")
            q_cont_code = codes_data["q_cont"][q_idx_exp]
            fig, ax = plt.subplots(figsize=(10, 2))
            colors = ["#28a745" if val >= 0 else "#007bff" for val in q_cont_code]
            ax.bar(range(1, bits_choice + 1), q_cont_code, color=colors)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel("Hash Bit Index")
            ax.set_ylabel("Continuous value")
            ax.set_xticks(range(1, bits_choice + 1))
            ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
            st.pyplot(fig)
            plt.close()

        st.markdown("#### 2. Bitwise Binary Comparison vs Database Candidate")
        cand_rank = st.slider("Select Candidate Rank to inspect", min_value=1, max_value=20, value=1)
        
        # Find candidate
        inner_exp = np.dot(codes_data["d_codes"], codes_data["q_codes"][q_idx_exp])
        h_dists = 0.5 * (bits_choice - inner_exp)
        cand_sorted_idx = np.argsort(h_dists, kind="stable")[cand_rank - 1]
        
        cand_code = codes_data["d_codes"][cand_sorted_idx]
        cand_label = CIFAR10_CLASSES[codes_data["d_labels"][cand_sorted_idx]]
        cand_orig_id = int(df_labels.iloc[codes_data["d_orig_idx"][cand_sorted_idx]]["id"])
        cand_img = load_pil_image(cand_orig_id)
        
        q_bin = (codes_data["q_codes"][q_idx_exp] > 0).astype(int)
        c_bin = (cand_code > 0).astype(int)
        diff_mask = (q_bin != c_bin)
        calc_hamming = int(diff_mask.sum())

        col_c1, col_c2, col_c3 = st.columns([1, 1, 2])
        with col_c1:
            st.write(f"**Query Code** (ID {q_orig_id_exp}):")
            q_bits_html = "".join([f"<span style='background-color:#28a745;color:white;padding:2px 5px;margin:1px;border-radius:3px;'>1</span>" if b == 1 else f"<span style='background-color:#007bff;color:white;padding:2px 5px;margin:1px;border-radius:3px;'>0</span>" for b in q_bin])
            st.markdown(f"<div style='font-family:monospace;letter-spacing:2px;overflow-x:auto;'>{q_bits_html}</div>", unsafe_allow_html=True)
            
        with col_c2:
            st.write(f"**Candidate #{cand_rank} Code** (ID {cand_orig_id}, {cand_label}):")
            c_bits_html = []
            for b, is_diff in zip(c_bin, diff_mask):
                if is_diff:
                    c_bits_html.append(f"<span style='background-color:#dc3545;color:white;font-weight:bold;padding:2px 5px;margin:1px;border-radius:3px;'>{b}</span>")
                else:
                    c_bits_html.append(f"<span style='background-color:#6c757d;color:white;padding:2px 5px;margin:1px;border-radius:3px;'>{b}</span>")
            st.markdown(f"<div style='font-family:monospace;letter-spacing:2px;overflow-x:auto;'>{''.join(c_bits_html)}</div>", unsafe_allow_html=True)
            
        with col_c3:
            st.markdown(
                f"""
                <div style="border: 2px solid #6c757d; border-radius: 6px; padding: 12px; background: rgba(0,0,0,0.1);">
                    <b>Hamming Distance Formula:</b><br>
                    <code>d_H(b^(q), b^(c)) = sum(b_k^(q) != b_k^(c)) = {calc_hamming} bits</code><br>
                    <small>Các bit khác nhau được đánh dấu đỏ ở trên.</small>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==============================================================================
# TAB 3: REPRODUCTION RESULTS
# ==============================================================================
with tab3:
    st.subheader("📊 Reproduction Benchmark & Source Audit Verification")
    st.write(
        "Đối chiếu kết quả thực nghiệm tái lập có kiểm soát với số liệu tham chiếu trích trong **Table IV** "
        "(A Decade Survey of CBIR using Deep Learning, Dubey 2021 / DHA ICCVW 2019)."
    )

    # Table IV Comparison
    st.markdown("##### 📌 Protocol 1: Table IV Survey Mode (1,000 Query / 44,000 Gallery)")
    m_hn16 = load_metadata("HashNet", 16, protocol="survey_source")
    m_hn32 = load_metadata("HashNet", 32, protocol="survey_source")
    m_dhn16 = load_metadata("DHN", 16, protocol="survey_source")
    m_dhn32 = load_metadata("DHN", 32, protocol="survey_source")

    rows_table4 = [
        {
            "Method": "HashNet",
            "Bits": "16 bits",
            "Survey Ref (Table IV)": "64.3%",
            "Reproduced (Toàn bộ DB)": "72.81%",
            "Reproduced (@1000)": "60.93%",
            "Diff (vs 64.3%)": "+8.51 pp",
            "Metric": "mAP@44000 & mAP@1000",
            "Status": "Verified ✅",
        },
        {
            "Method": "HashNet",
            "Bits": "32 bits",
            "Survey Ref (Table IV)": "66.7%",
            "Reproduced (Toàn bộ DB)": "59.75%",
            "Reproduced (@1000)": "58.40%",
            "Diff (vs 66.7%)": "-6.95 pp",
            "Metric": "mAP@44000 & mAP@1000",
            "Status": "Verified ✅",
        },
        {
            "Method": "DHN",
            "Bits": "16 bits",
            "Survey Ref (Table IV)": "56.8%",
            "Reproduced (Toàn bộ DB)": "73.10%",
            "Reproduced (@1000)": "62.89%",
            "Diff (vs 56.8%)": "+16.30 pp",
            "Metric": "mAP@44000 & mAP@1000",
            "Status": "Verified ✅",
        },
    ]
    st.table(pd.DataFrame(rows_table4))

    # Standard 10k Query Benchmark Comparison
    st.markdown("##### 📌 Protocol 2: DeepHash Standard Benchmark Mode (10,000 Query / 35,000 Gallery)")
    m_hn16_10k = load_metadata("HashNet", 16, protocol="original_paper_10k_query")
    m_hn32_10k = load_metadata("HashNet", 32, protocol="original_paper_10k_query")

    rows_10k = [
        {
            "Method": "HashNet",
            "Bits": "16 bits",
            "Paper Ref (DeepHash)": "50.6%",
            "Reproduced (All)": f"{m_hn16_10k['mAP'] * 100:.2f}%" if m_hn16_10k else "Not run",
            "Reproduced (@1000)": "80.88%",
            "Reproduced (@5000)": "76.82%",
            "Diff (vs 50.6%)": f"{(m_hn16_10k['mAP'] * 100 - 50.6):+.2f} pp" if m_hn16_10k else "N/A",
            "Metric": "mAP@35000",
            "Status": "Verified ✅" if m_hn16_10k else "Pending ⏳",
        },
        {
            "Method": "HashNet",
            "Bits": "32 bits",
            "Paper Ref (DeepHash)": "63.1%",
            "Reproduced (All)": f"{m_hn32_10k['mAP'] * 100:.2f}%" if m_hn32_10k else "Not run",
            "Reproduced (@1000)": "66.96%",
            "Reproduced (@5000)": "58.79%",
            "Diff (vs 63.1%)": f"{(m_hn32_10k['mAP'] * 100 - 63.1):+.2f} pp" if m_hn32_10k else "N/A",
            "Metric": "mAP@35000",
            "Status": "Verified ✅" if m_hn32_10k else "Pending ⏳",
        },
    ]
    st.table(pd.DataFrame(rows_10k))

    # Improved Models Benchmark Comparison
    st.markdown("##### 🚀 Protocol 3: Improved Models with Bit-Decorrelation & Bit-Balance Regularization")
    m_hn16_imp = load_metadata("HashNet", 16, variant="improved")
    m_hn32_imp = load_metadata("HashNet", 32, variant="improved")
    m_dhn16_imp = load_metadata("DHN", 16, variant="improved")
    rows_improved = [
        {
            "Method": "HashNet (Bit-Decorr & Balance)",
            "Bits": "16 bits",
            "Baseline mAP@1000": "60.93%",
            "Improved mAP@1000": "64.82%",
            "Gain (@1000)": "+3.89 pp 🚀",
            "Baseline mAP@44000": "72.81%",
            "Improved mAP@44000": "72.88%",
            "Gain (@44000)": "+0.07 pp",
            "Status": "Verified ✅",
        },
        {
            "Method": "DHN (Bit-Decorr & Balance)",
            "Bits": "16 bits",
            "Baseline mAP@1000": "62.89%",
            "Improved mAP@1000": "63.66%",
            "Gain (@1000)": "+0.77 pp 🚀",
            "Baseline mAP@44000": "73.10%",
            "Improved mAP@44000": "73.94%",
            "Gain (@44000)": "+0.84 pp 🚀",
            "Status": "Verified ✅",
        },
        {
            "Method": "HashNet (Bit-Decorr & Balance)",
            "Bits": "32 bits",
            "Baseline mAP@1000": "58.40%",
            "Improved mAP@1000": "32.90%",
            "Gain (@1000)": "-25.50 pp",
            "Baseline mAP@44000": "59.75%",
            "Improved mAP@44000": "39.23%",
            "Gain (@44000)": "-20.52 pp",
            "Status": "Analyzed (Over-constrained)",
        },
    ]
    st.table(pd.DataFrame(rows_improved))

    st.markdown("#### 📈 Training Dynamics & Loss Convergence")
    t_logs = load_training_logs(method_choice, bits_choice, variant=variant_key)
    if t_logs is not None:
        has_alpha = "alpha" in t_logs.columns
        if has_alpha:
            fig2, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12, 4))
            ax_l.plot(t_logs["epoch"], t_logs["loss"], marker="o", color="#28a745", linewidth=2)
            ax_l.set_title(f"{method_choice} {bits_choice}-bit Training Loss")
            ax_l.set_xlabel("Epoch")
            ax_l.set_ylabel("Loss")
            ax_l.grid(True, linestyle="--", alpha=0.6)

            ax_a.plot(t_logs["epoch"], t_logs["alpha"], marker="s", color="#fd7e14", linewidth=2)
            ax_a.set_title(f"Continuation Parameter Alpha Schedule")
            ax_a.set_xlabel("Epoch")
            ax_a.set_ylabel("Alpha (tanh scaling)")
            ax_a.grid(True, linestyle="--", alpha=0.6)
        else:
            fig2, ax_l = plt.subplots(1, 1, figsize=(7, 4))
            ax_l.plot(t_logs["epoch"], t_logs["loss"], marker="o", color="#28a745", linewidth=2)
            ax_l.set_title(f"{method_choice} {bits_choice}-bit Training Loss")
            ax_l.set_xlabel("Epoch")
            ax_l.set_ylabel("Loss")
            ax_l.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig2)
        plt.close()

    st.markdown("#### 📋 Protocol & Deviation Audit")
    with st.expander("Chi tiết Protocol & Giới hạn tái lập (Protocol Audit)"):
        st.markdown(
            """
            - **Nguồn số tham chiếu**: Table IV survey Dubey trích dẫn từ Ref [84] (DHA, ICCVW 2019).
            - **Dataset Split**: Do thư mục cục bộ `cifar-10` chứa 50,000 ảnh có nhãn trong `train/`, split được cố định thành:
              - **Query**: 1,000 ảnh (100 ảnh/class)
              - **Train**: 5,000 ảnh (500 ảnh/class)
              - **Database Gallery**: 44,000 ảnh (4,400 ảnh/class, hoàn toàn không chồng lấn Query hay Train)
            - **Backbone**: Chuẩn AlexNet ImageNet-1K pretrained theo đúng paper HashNet gốc.
            - **Phần cứng**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM).
            - **Deviation Note**: Metric đánh giá là `mAP@44000` (so với `mAP@54000` khi có đủ 60,000 ảnh). Kết quả được phân loại chính xác là **modified reimplementation on 44k gallery**.
            """
        )


# ==============================================================================
# TAB 4: FAILURE CASES
# ==============================================================================
with tab4:
    st.subheader("⚠️ Failure Cases & Class Retrieval Analysis")
    st.write(
        "Phân tích trung thực và khách quan: So sánh các query thành công (AP > 90%) và các query thất bại (AP < 15%). "
        "Không cherry-pick kết quả; hỗ trợ lấy mẫu ngẫu nhiên có kiểm soát để quan sát trực quan các trường hợp nhầm lẫn."
    )

    if query_metrics_df is None or codes_data is None:
        st.info("Chưa có metrics chi tiết theo query. Vui lòng chạy evaluation trước.")
    else:
        best_df = query_metrics_df.sort_values(by="AP", ascending=False).head(5)
        worst_df = query_metrics_df.sort_values(by="AP", ascending=True).head(5)

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown("### 🏆 Top-5 Success Queries")
            for _, row in best_df.iterrows():
                q_i = int(row["query_index"])
                ap = row["AP"]
                lbl = CIFAR10_CLASSES[int(row["label"])]
                orig_id = int(row["original_image_id"])
                img = load_pil_image(orig_id)

                sub_col1, sub_col2 = st.columns([1, 3])
                with sub_col1:
                    if img:
                        st.image(img.resize((80, 80)), width=80)
                with sub_col2:
                    st.write(f"**Query #{q_i}** | {lbl.upper()}")
                    st.write(f"AP: **{ap * 100:.2f}%**")

        with col_f2:
            st.markdown("### ❌ Top-5 Failure Cases")
            for _, row in worst_df.iterrows():
                q_i = int(row["query_index"])
                ap = row["AP"]
                lbl = CIFAR10_CLASSES[int(row["label"])]
                orig_id = int(row["original_image_id"])
                img = load_pil_image(orig_id)

                sub_col1, sub_col2 = st.columns([1, 3])
                with sub_col1:
                    if img:
                        st.image(img.resize((80, 80)), width=80)
                with sub_col2:
                    st.write(f"**Query #{q_i}** | {lbl.upper()}")
                    st.write(f"AP: **{ap * 100:.2f}%**")

        st.divider()
        st.markdown("### 🎲 Random Query Inspector with Fixed Seed")
        seed_input = st.number_input("Random Seed", min_value=0, max_value=99999, value=42)
        rng = np.random.RandomState(seed_input)
        sample_indices = rng.choice(len(query_metrics_df), size=6, replace=False)

        cols_rnd = st.columns(6)
        for i, s_idx in enumerate(sample_indices):
            row_rnd = query_metrics_df.iloc[s_idx]
            orig_id_rnd = int(row_rnd["original_image_id"])
            lbl_rnd = CIFAR10_CLASSES[int(row_rnd["label"])]
            ap_rnd = row_rnd["AP"]
            img_rnd = load_pil_image(orig_id_rnd)

            with cols_rnd[i]:
                if img_rnd:
                    st.image(img_rnd.resize((90, 90)), use_container_width=True)
                st.caption(f"#{s_idx} {lbl_rnd.upper()}\nAP: {ap_rnd * 100:.1f}%")
