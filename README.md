# Deep Hashing Reproduction Benchmark: HashNet & DHN on CIFAR-10

Project tái lập có kiểm soát một phần kết quả trong bài báo:
> **“A Decade Survey of Content Based Image Retrieval using Deep Learning”**  
> Shiv Ram Dubey, IEEE TCSVT, Vol. 32, No. 5, 2022. DOI: `10.1109/TCSVT.2021.3080920`.

Tập trung vào kiểm toán và tái lập kết quả Deep Hashing trên **CIFAR-10 (Table IV)**:
- **HashNet** (ICCV 2017): 16 bits (64.3%), 32 bits (66.7%).
- **DHN** (AAAI 2016): 16 bits (56.8%), 32 bits (60.3%).

---

## Cấu trúc thư mục dự án

```text
├── configs/              # Cấu hình thực nghiệm (YAML) cho 16-bit và 32-bit
├── docs/                 # Tài liệu học thuật và phân tích kiểm toán
│   ├── source_audit.md   # Báo cáo kiểm toán nguồn và xác minh số liệu
│   ├── protocol.md       # Đặc tả khóa protocol, dataset split, metric AP
│   └── deviations.md     # Nhật ký các sai khác và tương thích phần cứng
├── src/                  # Mã nguồn chính
│   ├── datasets/         # CIFAR-10 data loader & fixed manifest manager
│   ├── models/           # Backbone AlexNet & HashNet / DHN architecture
│   ├── losses/           # Pairwise loss, quantization loss, continuation
│   ├── retrieval/        # Hamming ranking, chunked distance calculation
│   └── evaluation/       # AP, mAP@54000 reference & optimized evaluators
├── scripts/              # Các kịch bản chạy
│   ├── prepare_data.py   # Chuẩn bị dữ liệu và sinh manifest hash cố định
│   ├── train.py          # Huấn luyện mô hình Deep Hashing
│   ├── extract_codes.py  # Trích xuất binary hash codes
│   └── evaluate.py       # Đánh giá mAP@54000 từ binary codes đã trích xuất
├── tests/                # Unit tests cho metrics, ranking, splits
├── app/                  # Trực quan hóa Streamlit Dashboard (Demo)
│   └── streamlit_app.py
├── outputs/              # Manifests, logs, checkpoints, kết quả csv
├── requirements.txt      # Thư viện phụ thuộc
└── README.md
```

---

## Các chế độ Protocol (Modes)

- **`survey_source`** (Chế độ mặc định đối chiếu Table IV):
  - Kích thước: 1,000 Query, 5,000 Train, 54,000 Retrieval Database.
  - Metric: `mAP@54000` trên toàn bộ retrieval database.
  - Backbone: AlexNet ImageNet-pretrained.
- **`original_paper`**:
  - Giao thức theo repository gốc của tác giả (THUML).

Xem chi tiết tại [docs/protocol.md](docs/protocol.md) và [docs/source_audit.md](docs/source_audit.md).

