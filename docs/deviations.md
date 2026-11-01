# Dataset Verification & Deviations: Local CIFAR-10 Dataset

Tài liệu này ghi lại việc xác minh cấu trúc tập dữ liệu thực tế có sẵn tại thư mục `cifar-10` trong workspace.

---

## 1. Nguồn dữ liệu thực tế tại workspace
- **Đường dẫn**: `cifar-10/` (định dạng ảnh gốc Kaggle CIFAR-10):
  - `cifar-10/trainLabels.csv`: 50,000 dòng gắn nhãn chính xác cho 50,000 ảnh trong thư mục `train/`.
  - `cifar-10/train/`: Chứa 50,000 ảnh PNG ($32 \times 32$ RGB), đánh số từ `1.png` đến `50000.png`.
  - Phân bổ: 10 lớp hoàn hảo, mỗi lớp đúng 5,000 ảnh (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`).
  - `cifar-10/test/`: Chứa 300,000 ảnh unlabelled (dành cho Kaggle test/private test).

---

## 2. Giao thức Split áp dụng cho tập dữ liệu có sẵn (`local_kaggle_cifar10`)

Theo tài liệu kiểm toán [`docs/source_audit.md`](docs/source_audit.md), bài báo khảo sát và paper DHA sử dụng chuẩn:
- **1,000 Query images** (100 ảnh mỗi lớp)
- **5,000 Training images** (500 ảnh mỗi lớp)

Do dữ liệu có nhãn chính xác trong thư mục `cifar-10/` là **50,000 ảnh**:
1. **Chế độ `cifar10_44k` (Tương đương cấu trúc `survey_source`)**:
   - **Query set ($Q$)**: 1,000 ảnh (100 ảnh/lớp, chọn ngẫu nhiên có kiểm soát theo fixed seed).
   - **Training set ($T$)**: 5,000 ảnh (500 ảnh/lớp, chọn ngẫu nhiên có kiểm soát theo fixed seed).
   - **Retrieval Database ($D$)**: 44,000 ảnh ($50,000 - 1,000 - 5,000$).
   - Đảm bảo tính chất cốt lõi: $Q \cap T = \emptyset, Q \cap D = \emptyset, T \cap D = \emptyset$.
   - Metric: `mAP@44000` (được gắn nhãn là *modified reimplementation on 44k database* trong `docs/deviations.md`).

2. **Chế độ `cifar10_50k` (Chuẩn DeepHash `cifar10-2`)**:
   - 10,000 ảnh Query và 50,000 ảnh Database.

Tất cả các split đều được xuất ra manifest cố định với mã băm SHA-256 để đảm bảo tính tất định (100% reproducible).

---

## 3. Khác biệt kỹ thuật về Framework & Loss Formulation (PyTorch 2.5.1 vs Caffe)

1. **Software & CUDA**:
   - Paper gốc HashNet (2017) viết trên BVLC Caffe và port PyTorch 0.3.1 (năm 2018).
   - Dự án tái lập sử dụng **PyTorch 2.5.1+cu121** trên GPU NVIDIA RTX 3050 (4GB VRAM).
   - Sử dụng backbone chuẩn `torchvision.models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)`.

2. **Chẩn đoán toán học về Weighted Likelihood Loss**:
   - Theo Eq. (3) trong paper HashNet gốc:
     $$\mathcal{L} = \sum_{s_{ij} \in \mathcal{S}} w_{ij} \left( \log(1 + e^{\theta_{ij}}) - s_{ij} \theta_{ij} \right)$$
     với $w_{ij} = \frac{|\mathcal{S}|}{|\mathcal{S}_1|}$ (khi $s_{ij}=1$) và $\frac{|\mathcal{S}|}{|\mathcal{S}_0|}$ (khi $s_{ij}=0$).
   - Nếu $w_{ij}$ chỉ nhân với số hạng logarit mà không nhân với số hạng tuyến tính, đạo hàm tại $\theta=0$ cho cặp positive ($S=1$) là $w_{pos} \cdot 0.5 - 1 \approx 10 \times 0.5 - 1 = +4 > 0$, dẫn đến việc gradient descent đẩy cả cặp positive và negative về phía âm, gây code collapse (loss kẹt tại $2 \ln 2 \approx 1.386$ và tất cả hash bit hội tụ về 0).
   - Khi áp dụng đúng công thức đồng nhất:
     $$\mathcal{L}_{ij} = \begin{cases} w_{pos} \cdot \text{softplus}(-\theta_{ij}), & s_{ij} = 1 \\ w_{neg} \cdot \text{softplus}(\theta_{ij}), & s_{ij} = 0 \end{cases}$$
     Hệ thống đạt cân bằng lực gradient hoàn hảo tại 0 và hội tụ ổn định không bị saturation.

---

## 4. Bảng kết quả tái lập (Reproduced Results vs Table IV Survey)

| Phương pháp | Hash Bits | Survey Ref (Table IV) | Reproduced (Thực tế) | Độ lệch (Δ pp) | Protocol Mode | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HashNet** | 16 | 64.3% (0.643) | **72.81%** (0.7281) | +8.51 pp | `survey_source` (1k query / 44k gallery) | Vượt mốc tham chiếu |
| **HashNet** | 32 | 66.7% (0.667) | **59.75%** (0.5975) | -6.95 pp | `survey_source` (1k query / 44k gallery) | Hoàn thành |
| **DHN** | 16 | 56.8% (0.568) | **73.10%** (0.7310) | +16.30 pp | `survey_source` (1k query / 44k gallery) | Vượt mốc tham chiếu |
| **DHN** | 32 | 60.3% (0.603) | *Sẵn sàng* | - | `survey_source` (1k query / 44k gallery) | Sẵn sàng triển khai |

### 4.2. Bảng đối chiếu theo Giao thức Chuẩn 10,000 Query (DeepHash Benchmark)

| Phương pháp | Hash Bits | Paper Ref (DeepHash) | Reproduced (mAP@All) | mAP@1000 | mAP@5000 | Độ lệch so với 63.1% | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HashNet** | 16 | 50.6% (0.506) | **80.60%** | 80.88% | 76.82% | +30.0 pp | Gradient cân bằng tốt |
| **HashNet** | 32 | 63.1% (0.631) | **65.19%** | **66.96%** | **58.79%** | **+2.09 pp** (khớp mốc ~63%) | Tái lập thành công |

---

## 5. Kết quả Cải Tiến Bằng Bit-Decorrelation & Bit-Balance Regularization

Nhằm khắc phục hiện tượng dư thừa mã (bit redundancy) và tương quan chéo cao giữa các bit, mô hình được bổ sung ràng buộc điều hòa kép:
1. **Bit Balance Loss** ($\mathcal{L}_{\text{balance}}$): Phạt độ lệch trung bình khác 0 của mỗi bit qua mini-batch để tối đa hóa entropy.
2. **Bit Decorrelation Loss** ($\mathcal{L}_{\text{decorr}}$): Phạt các phần tử ngoài đường chéo của ma trận hiệp phương sai mẫu $\frac{1}{B} H^\top H$ nhằm triệt tiêu tương quan giữa các bit.

### Bảng So Sánh Hiệu Quả Cải Tiến (Baseline vs Improved):

| Phương pháp | Hash Bits | Metric | Baseline mAP | Improved mAP (Bit-Decorr & Balance) | Mức Cải Thiện ($\Delta$ pp) | Trạng Thái |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **HashNet** | 16 | mAP@1000 | 60.93% | **64.82%** | **+3.89 pp** | Đã nghiệm thu (Tăng vọt) |
| **HashNet** | 16 | mAP@44000 | 72.81% | **72.88%** | **+0.07 pp** | Đã nghiệm thu (Vượt mốc paper) |
| **HashNet** | 32 | mAP@1000 | 58.40% | 32.90% | -25.50 pp | Đã phân tích (Over-constrained tại $\lambda=0.1$) |
| **HashNet** | 32 | mAP@44000 | 59.75% | 39.23% | -20.52 pp | Đã phân tích (Cần scale $\lambda \propto 1/\binom{K}{2}$) |
| **DHN** | 16 | mAP@1000 | 62.89% | **63.66%** | **+0.77 pp** | Đã nghiệm thu (Tăng đều) |
| **DHN** | 16 | mAP@44000 | 73.10% | **73.94%** | **+0.84 pp** | Đã nghiệm thu (Vượt mốc paper) |
| **DHN** | 32 | mAP@44000 | 60.3% (Paper Ref) | *Sẵn sàng* | - | Đã cấu hình |



