# Source Audit: Deep Hashing Benchmark on CIFAR-10

## 1. Tổng quan mục tiêu kiểm toán nguồn
Mục tiêu của tài liệu này là xác minh độc lập nguồn gốc học thuật, mã nguồn chính thức, cấu hình thực nghiệm và các điểm khác biệt liên quan đến các kết quả deep hashing được trích dẫn trong:
- **Survey Paper**: Shiv Ram Dubey, *"A Decade Survey of Content Based Image Retrieval using Deep Learning"*, IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol. 32, No. 5, pp. 2687–2704, 2022. (Preprint: arXiv:2012.00641, DOI: 10.1109/TCSVT.2021.3080920).
- **Trọng tâm đối chiếu**: **Table IV** trong survey paper — Performance comparison on CIFAR-10 (mAP@54000).

Các số tham chiếu trong Table IV cần kiểm toán:
- **HashNet**: 16 bits = 64.3% (0.643), 32 bits = 66.7% (0.667).
- **DHN**: 16 bits = 56.8% (0.568), 32 bits = 60.3% (0.603).

> **Lưu ý quan trọng**: Đây là các con số *tham chiếu (reference)* được survey tổng hợp, **không** phải là kết quả mặc định đã tái lập.

---

## 2. Thông tin chi tiết các bài báo nguồn

### 2.1. Nguồn trích dẫn trực tiếp trong Table IV (Survey Reference [84])
- **Tiêu đề**: *“DHA: Supervised Deep Learning to Hash with an Adaptive Loss Function”*
- **Tác giả**: Jiehao Xu, Chengyu Guo, Qingjie Liu, Jie Qin, Yunhong Wang, Li Liu (Beihang University & Inception Institute of AI).
- **Hội nghị**: IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2019, pp. 3054–3062.
- **DOI**: `10.1109/ICCVW.2019.00368`.
- **Vai trò trong kiểm toán**: Table IV của survey Dubey trích dẫn bảng so sánh hiệu năng trực tiếp từ Reference [84] (DHA). Cụ thể, các hàng dữ liệu so sánh các phương pháp tiền nhiệm gồm CNNH, DNNH, DHN, HashNet, HashGAN trên CIFAR-10 đều lấy từ bảng kết quả thực nghiệm trong paper DHA.
- **Repository chính thức**: Không công bố repository mã nguồn chính thức công khai.

### 2.2. Phương pháp HashNet (Survey Reference [71])
- **Tiêu đề**: *“HashNet: Deep Learning to Hash by Continuation”*
- **Tác giả**: Zhangjie Cao, Mingsheng Long, Jianmin Wang, Philip S. Yu (Tsinghua University & UIC).
- **Hội nghị**: IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2187–2196.
- **DOI**: `10.1109/ICCV.2017.239`.
- **Preprint**: `arXiv:1702.00758`.
- **Repository chính thức của nhóm tác giả (THUML)**:
  - URL: `https://github.com/thuml/HashNet`
  - Ngôn ngữ/Framework: Hỗ trợ Caffe (bản ban đầu) và PyTorch (trong thư mục `pytorch/`, ban đầu viết trên PyTorch 0.3.1).
  - Thuộc hệ sinh thái: `thuml/DeepHash` (`https://github.com/thuml/DeepHash`).

### 2.3. Phương pháp DHN (Survey Reference [67])
- **Tiêu đề**: *“Deep Hashing Network for Efficient Similarity Retrieval”*
- **Tác giả**: Han Zhu, Mingsheng Long, Jianmin Wang, Yue Cao (Tsinghua University).
- **Hội nghị**: AAAI Conference on Artificial Intelligence (AAAI), 2016, pp. 2472–2479.
- **Repository chính thức của nhóm tác giả (THUML)**:
  - Caffe release: `https://github.com/thuml/hash-caffe`
  - DeepHash package: `https://github.com/thuml/DeepHash`

### 2.4. Third-Party Implementations đã xác minh
- **`swuxyj/DeepHash-pytorch`** (`https://github.com/swuxyj/DeepHash-pytorch`):
  - Bộ benchmark mã nguồn mở cộng đồng chuyển đổi các thuật toán Deep Hashing từ Caffe/Torch sang PyTorch hiện đại.
  - Chứa cài đặt cụ thể cho CIFAR-10 với 3 biến thể phân chia dữ liệu (`cifar10` 54k database, `cifar10-1`, `cifar10-2`).
  - Được xem là tài liệu tham khảo bổ trợ để xác nhận tính khả thi của việc chạy mô hình trên PyTorch hiện đại, nhưng mọi công thức loss và cơ chế binarization phải đối chiếu chuẩn hóa với paper gốc và official repo `thuml/HashNet`.

---

## 3. Cấu hình thực nghiệm đã xác nhận qua các nguồn

### 3.1. Nguồn kết quả Table IV (chế độ `survey_source` / DHA Paper [84])
| Tham số | Giá trị xác nhận |
| :--- | :--- |
| Dataset | CIFAR-10 (60,000 ảnh 32x32, 10 lớp) |
| Query set size | 1,000 ảnh (100 ảnh mỗi lớp) |
| Training set size | 5,000 ảnh (500 ảnh mỗi lớp) |
| Retrieval Database size | 54,000 ảnh (toàn bộ phần còn lại sau khi trừ query set và training set) |
| Metric báo cáo | mAP@54000 (đánh giá trên toàn bộ 54,000 ảnh database) |
| Backbone mạng | AlexNet (ImageNet pretrained weights) |
| Hash lengths | 16, 32, 48, 64 bits |
| Kết quả HashNet | 16-bit: **64.3%**; 32-bit: **66.7%** |
| Kết quả DHN | 16-bit: **56.8%**; 32-bit: **60.3%** |

### 3.2. Phương pháp HashNet gốc (chế độ `original_paper` / ICCV 2017)
| Thành phần | Đặc tả kỹ thuật |
| :--- | :--- |
| Backbone | AlexNet (hoặc VGG-16; trong benchmark retrieval tiêu chuẩn của tác giả trên ImageNet/CIFAR-10 là AlexNet) |
| Hash Layer | Fully Connected layer thay thế `fc8`, số chiều ra bằng hash code length $K$ |
| Activation | Smooth tanh approximation with scale: $y = \tanh(\alpha z)$ (hoặc $\beta z$) |
| Continuation schedule | $\alpha = \sqrt{1 + \gamma \cdot t}$ (với $t$ là iteration hoặc epoch step, $\alpha$ tăng dần từ $1.0$ đến khi xấp xỉ $\text{sgn}$) |
| Pairwise Loss | Weighted Maximum Likelihood Loss giải quyết mất cân bằng similarity: $\mathcal{L} = \sum_{s_{ij} \in \mathcal{S}} \left( w_{ij} \log(1 + e^{\alpha u_{ij}}) - s_{ij} \alpha u_{ij} \right)$ với $u_{ij} = \frac{1}{2} h_i^\top h_j$, $w_{ij}$ đánh trọng số dựa trên tỷ lệ cặp positive / negative |
| Binarization | Inference: $b = \text{sgn}(z)$ hoặc $b = \text{sgn}(\tanh(\alpha z)) \in \{-1, +1\}^K$ |
| Optimizer | Mini-batch SGD, momentum = 0.9, weight decay = 0.0005 |
| Batch Size | 256 (hoặc mini-batch tiêu chuẩn theo bộ nhớ VRAM) |

### 3.3. Phương pháp DHN gốc (AAAI 2016)
| Thành phần | Đặc tả kỹ thuật |
| :--- | :--- |
| Backbone | AlexNet pretrained trên ImageNet |
| Hash Layer | FC layer với activation $\tanh(z)$ để giới hạn miền giá trị $[-1, 1]$ |
| Objective function | Pairwise cross-entropy similarity loss + Quantization loss: $\mathcal{L} = \mathcal{L}_{pair} + \lambda \mathcal{L}_{quant}$ |
| Quantization loss | $\mathcal{L}_{quant} = \sum_i \log \cosh (\| |u_i| - \mathbf{1} \|_1)$ hoặc $L_2$ quantization $\|u_i - \text{sgn}(u_i)\|^2$ |
| Optimizer | SGD với learning rate phân cấp (hash layer học nhanh hơn 10x so với backbone) |

---

## 4. Các điểm khác biệt và xung đột giữa các nguồn (Deviations & Nuances)

1. **Khác biệt về phân chia Retrieval Database trên CIFAR-10**:
   - **Giao thức 54,000 (Table IV & DHA [84])**:
     - Tổng cộng: 60,000 ảnh.
     - Query: 1,000 ảnh (100 ảnh/class).
     - Train: 5,000 ảnh (500 ảnh/class).
     - Database: 54,000 ảnh = $60,000 - 1,000 - 5,000$. Training set hoàn toàn tách biệt khỏi Database.
   - **Giao thức 59,000 (DeepHash standard / Caffe script)**:
     - Query: 1,000 ảnh.
     - Database: 59,000 ảnh (toàn bộ phần còn lại).
     - Train: 5,000 ảnh được lấy mẫu từ bên trong 59,000 database images.
   - **Giao thức 50,000 / mAP@50000**:
     - Query: 10,000 ảnh (test split gốc).
     - Database / Train: 50,000 ảnh (train split gốc).
   - **Đánh giá kiểm toán**: Table IV báo cáo rõ ràng là **mAP@54000**, do đó giao thức chính xác của nguồn số liệu survey là giao thức **54,000 database** (training images không nằm trong database hoặc cutoff đúng 54,000 gallery images).

2. **Khác biệt về Backbone giữa các bài báo khảo sát**:
   - Một số bài báo sau này chuyển sang ResNet-18/50 để đạt mAP cao hơn (mAP có thể nhảy từ 64% lên 80%+).
   - Tuy nhiên, trong Table IV của survey Dubey và paper DHA [84], các kết quả HashNet 64.3% / 66.7% và DHN 56.8% / 60.3% đều sử dụng backbone kinh điển **AlexNet**. Nếu ta tự ý chuyển sang ResNet-50 thì kết quả mAP tăng vọt không phản ánh đúng thí nghiệm của paper gốc.

3. **Chi tiết chưa xác định (Ambiguities)**:
   - File cấu hình trọng số pretrained cụ thể của AlexNet trong DHA (PyTorch torchvision `AlexNet_Weights.IMAGENET1K_V1` vs BVLC Caffe pretrained AlexNet). Caffe AlexNet và PyTorch torchvision AlexNet có sai khác nhỏ về cách chuẩn hóa LRN vs MaxPool order.
   - Chiến lược tie-breaking khi nhiều ảnh có cùng Hamming distance (Hash collision). Thường trong code đánh giá chuẩn, tie-breaking giữ nguyên thứ tự index ổn định (stable sort).

---

## 5. Kết luận xác minh và phân loại chế độ (Protocol Modes)

Nhằm đảm bảo tính minh bạch và trung thực tuyệt đối của nghiên cứu:
- **Chế độ A (`survey_source`)**: Bám sát giao thức số liệu Table IV trong Survey Dubey (nguồn DHA [84]): AlexNet backbone, CIFAR-10 split 1,000 Query / 5,000 Train / 54,000 Database, tính mAP@54000.
- **Chế độ B (`original_paper`)**: Thử nghiệm đối chiếu nếu áp dụng theo code của official repository THUML (nếu có khác biệt về split 59k vs 54k).

Dự án này khóa chặt mục tiêu đối chiếu chính vào **Chế độ A (`survey_source`)** để thẩm định trực tiếp các số liệu 64.3% và 66.7% của Table IV.

