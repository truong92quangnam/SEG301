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
