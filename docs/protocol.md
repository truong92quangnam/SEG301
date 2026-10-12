# Evaluation & Experimental Protocol

Tài liệu này xác định và khóa toàn bộ các tham số thực nghiệm, quy ước metric và phân chia tập dữ liệu nhằm đảm bảo việc tái lập có kiểm soát (controlled reproduction).

---

## 1. Phân loại chế độ Protocol (Modes)

Dự án phân biệt rõ hai chế độ thực nghiệm:

- **Mode A (`survey_source`)** [CHẾ ĐỘ MẶC ĐỊNH]:
  Bám sát cấu hình thực nghiệm từ nguồn trích dẫn của Table IV (Survey Reference [84] - DHA 2019):
  - Database tách biệt hoàn toàn khỏi Training set.
  - Kích thước Database = 54,000 ảnh.
  - Kích thước Query = 1,000 ảnh (100 ảnh / lớp).
  - Kích thước Training = 5,000 ảnh (500 ảnh / lớp).
  - Metric báo cáo: **mAP@54000** (toàn bộ database).

- **Mode B (`original_paper`)**:
  Bám sát cấu hình từ code chính thức của THUML (`thuml/HashNet` / `thuml/DeepHash`):
  - Database = 59,000 ảnh (60,000 - 1,000 query).
  - Training = 5,000 ảnh lấy mẫu từ 59,000 ảnh database (có overlap giữa train và database).
  - Metric báo cáo: **mAP@5000** hoặc **mAP@All (59,000)**.

> **Quy tắc tuyệt đối**: Không trộn lẫn kết quả của Mode A và Mode B trong cùng một bảng đối chiếu hiệu năng.

---

## 2. Quy cách phân chia Dataset (CIFAR-10)

### 2.1. Nguồn dữ liệu
- CIFAR-10 chuẩn gồm 60,000 ảnh màu $32 \times 32$ thuộc 10 lớp cân bằng (6,000 ảnh/lớp).
- Dữ liệu được trích xuất từ định dạng local trong repo (`cifar10/plain_text/` hoặc `torchvision.datasets.CIFAR10`) và sinh ra file manifest cố định.

### 2.2. Phân chia trong Mode `survey_source`
- **Query set ($Q$)**: 1,000 ảnh (chọn ngẫu nhiên 100 ảnh mỗi lớp với seed cố định).
- **Training set ($T$)**: 5,000 ảnh (chọn ngẫu nhiên 500 ảnh mỗi lớp từ phần còn lại, seed cố định).
- **Retrieval database ($D$)**: 54,000 ảnh (toàn bộ các ảnh còn lại sau khi trừ $Q$ và $T$).
- **Kiểm tra tính toàn vẹn**:
  - $|Q| + |T| + |D| = 1,000 + 5,000 + 5,4000 = 60,000$.
  - $Q \cap T = \emptyset$.
  - $Q \cap D = \emptyset$.
  - $T \cap D = \emptyset$.
  - Manifest được lưu trữ vĩnh viễn dưới dạng `outputs/splits/manifest_cifar10_54k.json` kèm mã băm SHA256 để không bị thay đổi giữa các lần chạy.

### 2.3. Quy tắc Self-Match và Relevance
- **Định nghĩa Relevance ($S_{ij}$)**:
  Hai ảnh $i$ và $j$ được coi là tương đồng ($S_{ij} = 1$) nếu và chỉ nếu chúng có cùng nhãn lớp phân loại ($y_i = y_j$). Ngược lại $S_{ij} = 0$.
- **Quy tắc Self-match**:
  Trong retrieval phase, Query set $Q$ được tra cứu đối chiếu trên Retrieval Database $D$. Do $Q \cap D = \emptyset$, không tồn tại self-match trong danh sách retrieved items.

---

## 3. Kiến trúc mô hình và Tiền xử lý

### 3.1. Backbone
- **Kiến trúc**: AlexNet chuẩn (AlexNet-based Deep Hashing).
- **Trọng số Pretrained**: ImageNet-1K pretrained weights (PyTorch standard `models.AlexNet_Weights.IMAGENET1K_V1`).
- **Thay đổi kiến trúc**:
  - Giữ nguyên các lớp tích chập `features` (conv1 - conv5) và hai lớp fully-connected đầu tiên `classifier[1]` (fc6), `classifier[4]` (fc7).
  - Thay thế lớp phân loại `fc8` bằng một Hash Head: `nn.Linear(4096, hash_bits, bias=True)`.

### 3.2. Tiền xử lý & Data Augmentation
- **Training**:
  - Resize ảnh CIFAR-10 lên $256 \times 256$.
  - Random crop về kích thước $224 \times 224$.
  - Random horizontal flip ($p = 0.5$).
  - Chuẩn hóa theo ImageNet mean `[0.485, 0.456, 0.406]` và std `[0.229, 0.224, 0.225]`.
- **Inference (Query & Database extraction)**:
  - Resize ảnh lên $256 \times 256$.
  - Center crop về $224 \times 224$.
  - Chuẩn hóa theo ImageNet mean và std.

---

## 4. Hàm mất mát và Chiến lược tối ưu hóa

### 4.1. HashNet Formulation
- **Continuous Hash Output**:
  $$h_i = \tanh(\alpha \cdot z_i) \in (-1, 1)^K$$
  với $z_i$ là logit đầu ra từ Hash Head, $\alpha \ge 1$ là continuation scale parameter.
- **Pairwise Similarity Inner Product**:
  $$u_{ij} = \frac{1}{2} h_i^\top h_j$$
- **Weighted Cross-Entropy Loss (Xử lý Imbalance)**:
  $$\mathcal{L} = \sum_{s_{ij} \in \mathcal{S}} \left( w_{ij} \log(1 + e^{\alpha u_{ij}}) - s_{ij} \alpha u_{ij} \right)$$
  với:
  - $s_{ij} \in \{0, 1\}$ là nhãn semantic similarity.
  - Trọng số cân bằng $w_{ij}$:
    $$w_{ij} = \begin{cases} c_{pos} = \frac{|\mathcal{S}|}{|\mathcal{S}_{pos}|}, & \text{nếu } s_{ij} = 1 \\ c_{neg} = \frac{|\mathcal{S}|}{|\mathcal{S}_{neg}|}, & \text{nếu } s_{ij} = 0 \end{cases}$$
- **Continuation Schedule**:
  $$\alpha_t = \sqrt{1 + \gamma \cdot t}$$
  với $t$ là training step/iteration và $\gamma$ là hệ số kiểm soát tốc độ tăng trưởng (mặc định $\gamma = 0.001$).
- **Binarization**:
  $$b_i = \text{sgn}(h_i) = \text{sgn}(z_i) \in \{-1, +1\}^K$$

### 4.2. Siêu tham số huấn luyện (Hyperparameters)
- **Optimizer**: SGD với momentum $0.9$, weight decay $0.0005$.
- **Learning rate**: $0.001$ cho backbone, $0.01$ cho hash head (10x để hash layer hội tụ nhanh).
- **Learning rate scheduler**: StepLR hoặc Annealing theo iteration: $\text{lr}_t = \frac{\text{lr}_0}{(1 + \gamma \cdot t)^{0.75}}$.
- **Batch size**: 64 (hoặc 128 tùy thuộc vào dung lượng VRAM thực tế đo đạc được).
- **Epochs**: 40 epochs (khoảng 3,000 iterations với batch size 64 trên 5,000 ảnh training).

---

## 5. Metric và Quy tắc Retrieval Evaluation

### 5.1. Khoảng cách xếp hạng (Ranking Distance)
- **Hamming Distance**:
  $$D_H(b_q, b_d) = \frac{1}{2} (K - b_q^\top b_d) = \sum_{k=1}^K \mathbb{I}(b_{q,k} \neq b_{d,k})$$
- Xếp hạng cơ sở dữ liệu $D$ theo khoảng cách $D_H$ tăng dần từ nhỏ nhất đến lớn nhất.

### 5.2. Công thức Average Precision (AP@K)
Cho một query $q$ và danh sách ranking top-$K$ database items:
$$\text{AP}@K(q) = \frac{\sum_{k=1}^K P(k) \cdot \text{rel}(k)}{N_{rel@K}}$$
Trong đó:
- $P(k) = \frac{\sum_{i=1}^k \text{rel}(i)}{k}$ là Precision tại vị trí $k$.
- $\text{rel}(k) \in \{0, 1\}$: nhận giá trị 1 nếu item thứ $k$ cùng nhãn với query, ngược lại bằng 0.
- **Mẫu số chuẩn hóa**:
  $$N_{rel@K} = \min(R_q, K)$$
  với $R_q$ là tổng số item relevant của query $q$ có trong toàn bộ retrieval database $D$.
- **Cutoff**: $K = 54,000$ (toàn bộ database).
- **Mean Average Precision (mAP@54000)**:
  $$\text{mAP}@K = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP}@K(q)$$
- **Chính sách Zero-Relevant**: Nếu một query không có bất kỳ relevant item nào trong database ($R_q = 0$), định nghĩa $\text{AP}(q) = 0.0$.
- **Chính sách Hamming Ties**: Áp dụng stable sort (giữ nguyên thứ tự ID gốc nếu cùng Hamming distance) để bảo đảm tính tất định và reproducibility.
- **Đánh giá theo Chunk (Chunked Evaluation)**: Tránh tính toàn bộ ma trận $1,000 \times 54,000$ trên GPU cùng lúc để ngăn lỗi CUDA OOM.

