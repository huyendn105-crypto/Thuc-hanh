# ⚖️ Ứng Dụng Dự Báo Gian Lận Báo Cáo Tài Chính (BCTC)
### Áp dụng Mô hình Beneish M-Score & Hồi quy Logistic trên nền tảng Streamlit

Ứng dụng web tương tác phục vụ mục đích thẩm định tín dụng, kiểm toán và đầu tư, giúp phát hiện sớm các dấu hiệu thao túng số liệu và gian lận trên Báo cáo tài chính doanh nghiệp thông qua bộ **8 chỉ số Beneish M-Score** và thuật toán học máy **Logistic Regression**.

---

## 🌟 Tính Năng Nổi Bật

1. **📈 Khám Phá & Phân Tích Dữ Liệu (EDA):**
   - Thống kê mô tả số lượng quan sát, tỷ lệ doanh nghiệp bình thường so với doanh nghiệp gian lận.
   - Trực quan hóa ma trận tương quan giữa 8 biến Beneish và cờ gian lận (`FRAUD_FLAG`).
   - So sánh trực quan giá trị trung bình của 8 chỉ số giữa nhóm doanh nghiệp gian lận và nhóm an toàn.

2. **🤖 Huấn Luyện & Đánh Giá Mô Hình:**
   - Pipeline chuẩn hóa tự động (`StandardScaler`) kết hợp thuật toán `LogisticRegression`.
   - Tùy chỉnh linh hoạt: Tỷ lệ phân chia Train/Test, Random State, Ngưỡng quyết định xác suất (Decision Threshold).
   - Đo lường toàn diện: **Accuracy**, **Precision**, **Recall**, **F1-Score**, **Specificity**, **ROC-AUC**, **FPR**, **FNR**.
   - Biểu diễn ma trận nhầm lẫn (Confusion Matrix) và đường cong ROC trực quan.
   - Hiển thị phương trình hồi quy, hệ số $\beta$ và tỷ số chênh (Odds Ratio $e^\beta$).
   - **Xuất báo cáo Excel chuyên nghiệp (`.xlsx`)**: Tải về trọn bộ kết quả mô hình gồm 5 trang tính chi tiết.

3. **🔍 Dự Báo Gian Lận Thực Tế:**
   - **Dự báo đơn lẻ cho 1 doanh nghiệp**: Nhập trực tiếp 8 chỉ số với các kịch bản mẫu (gian lận / an toàn) để nhận diện xác suất rủi ro ngay lập tức kèm biểu đồ phân tích yếu tố đóng góp.
   - **Đối chứng với thang điểm Beneish M-score gốc** (ngưỡng quy ước $-1.78$).
   - **Dự báo hàng loạt theo lô (Batch Prediction)**: Tải lên danh sách file CSV nhiều doanh nghiệp và tải về bảng xếp hạng rủi ro hoàn chỉnh.

4. **📖 Cẩm Nang Kiến Thức 8 Chỉ Số Beneish:**
   - Giải nghĩa bản chất tài chính, công thức và dấu hiệu cảnh báo của từng chỉ số: DSRI, GMI, AQI, SGI, DEPI, SGAI, TATA, LVGI.

---

## 📁 Cấu Trúc Thư Mục

```text
├── app.py                     # Mã nguồn ứng dụng Streamlit chính
├── requirements.txt           # Danh sách các thư viện cần thiết
├── README.md                  # Hướng dẫn chi tiết sử dụng & triển khai
└── MScore_data.csv            # Tập dữ liệu mẫu 8 chỉ số Beneish & FRAUD_FLAG
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Trên Máy Cục Bộ (Local)

### 1. Yêu cầu hệ thống
- Đã cài đặt **Python 3.9 - 3.12** trên máy tính ([Tải Python](https://www.python.org/downloads/)).
- Hệ điều hành: Windows, macOS hoặc Linux.

### 2. Các bước khởi chạy

1. **Mở Terminal / PowerShell / Command Prompt** tại thư mục chứa dự án:
   ```bash
   cd "d:/AI/TAO APP THUC HANH"
   ```

2. **(Khuyến nghị) Tạo môi trường ảo:**
   - Trên Windows:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - Trên macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Cài đặt các thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Khởi chạy ứng dụng Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Sau khi chạy lệnh trên, trình duyệt web sẽ tự động mở địa chỉ: `http://localhost:8501`.

---

## 🌐 Hướng Dẫn Đẩy Lên GitHub & Deploy Miễn Phí Trên Streamlit Cloud

### Bước 1: Tải mã nguồn lên GitHub

#### Cách A: Sử dụng giao diện web GitHub (Đơn giản nhất, không cần cài Git)
1. Đăng nhập vào tài khoản [GitHub](https://github.com/).
2. Nhấn nút **"New"** (hoặc dấu `+` ở góc trên bên phải) để tạo một **Repository** mới.
   - Đặt tên kho lưu trữ (Repository name), ví dụ: `financial-fraud-detection`.
   - Chọn chế độ **Public**.
   - Nhấn **"Create repository"**.
3. Tại trang Repository vừa tạo, chọn mục **"uploading an existing file"**.
4. Kéo thả 4 tệp sau vào cửa sổ trình duyệt:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `MScore_data.csv`
5. Nhấn **"Commit changes"** để hoàn tất tải lên.

#### Cách B: Sử dụng dòng lệnh Git (Dành cho lập trình viên)
```bash
# Khởi tạo git trong thư mục dự án
git init

# Thêm tất cả tệp vào danh sách chuẩn bị commit
git add app.py requirements.txt README.md MScore_data.csv

# Tạo commit đầu tiên
git commit -m "Khoi tao web app du bao gian lan BCTC tren Streamlit"

# Đặt tên nhánh chính là main
git branch -M main

# Liên kết với kho lưu trữ GitHub của bạn (thay bằng URL repo của bạn)
git remote add origin https://github.com/[TEN_GITHUB_CUA_BAN]/financial-fraud-detection.git

# Đẩy code lên GitHub
git push -u origin main
```

---

### Bước 2: Triển khai (Deploy) trên Streamlit Community Cloud

1. Truy cập vào trang quản lý: [Streamlit Community Cloud](https://share.streamlit.io/) và nhấn **"Sign in"** bằng tài khoản GitHub của bạn.
2. Nhấn nút **"New app"** (hoặc **"Create app"**).
3. Điền các thông tin triển khai:
   - **Repository:** Chọn kho lưu trữ bạn vừa tạo (ví dụ: `[TEN_GITHUB_CUA_BAN]/financial-fraud-detection`).
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (tùy chọn):** Đặt tên miền tùy chỉnh cho web app của bạn (ví dụ: `du-bao-gian-lan-bctc.streamlit.app`).
4. Nhấn **"Deploy!"**.
5. Đợi khoảng 1 - 2 phút để Streamlit Cloud tự động cài đặt các gói trong `requirements.txt` và xây dựng ứng dụng. Khi hoàn tất, web app của bạn sẽ chính thức hoạt động công khai trên toàn cầu!

---

## 📊 Cấu Trúc File Dữ Liệu Đầu Vào

Tập dữ liệu chuẩn yêu cầu định dạng `.csv` bao gồm 9 cột như sau:

| Tên cột | Ý nghĩa tài chính | Định dạng giá trị |
| :--- | :--- | :--- |
| `DSRI` | Chỉ số số ngày thu tiền khách hàng | Số thực (Float) |
| `GMI` | Chỉ số tỷ suất lợi nhuận gộp | Số thực (Float) |
| `AQI` | Chỉ số chất lượng tài sản phi hiện vật | Số thực (Float) |
| `SGI` | Chỉ số tốc độ tăng trưởng doanh thu | Số thực (Float) |
| `DEPI` | Chỉ số tốc độ trích khấu hao | Số thực (Float) |
| `SGAI` | Chỉ số chi phí bán hàng & quản lý doanh nghiệp | Số thực (Float) |
| `TATA` | Biến tổng dồn tích trên tổng tài sản | Số thực (Float) |
| `LVGI` | Chỉ số đòn bẩy tài chính | Số thực (Float) |
| `FRAUD_FLAG` | Nhãn thực tế: `0` (Bình thường) hoặc `1` (Gian lận) | Số nguyên nhị phân (`0` hoặc `1`) |

*Ghi chú: Khi thực hiện dự báo theo lô (Batch Prediction) cho danh sách công ty mới, cột `FRAUD_FLAG` không bắt buộc phải có.*

---

## 📜 Giấy Phép & Đóng Góp
- Dự án mã nguồn mở phục vụ mục đích nghiên cứu, học thuật và thực hành quản trị rủi ro tài chính.
- Mọi góp ý hoặc yêu cầu tính năng mới, vui lòng tạo **Issue** hoặc **Pull Request** trên GitHub.
