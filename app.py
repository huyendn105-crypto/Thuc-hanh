# -*- coding: utf-8 -*-
"""
Web App: Dự báo Gian lận Báo cáo Tài chính (BCTC) sử dụng Mô hình Beneish M-Score & Logistic Regression
Nền tảng: Streamlit
Tương thích: Streamlit Community Cloud & Local Deployment
"""

import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    auc
)

# ==============================================================================
# CẤU HÌNH TRANG WEB STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Dự báo Gian lận BCTC | Beneish M-Score",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện thẩm mỹ & chuyên nghiệp phong cách tài chính
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-fraud {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-safe {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .instruction-box {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HẰNG SỐ & ĐỊNH NGHĨA 8 CHỈ SỐ BENEISH
# ==============================================================================
FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]
TARGET = "FRAUD_FLAG"

FEATURE_METADATA = {
    "DSRI": {
        "vn_name": "Chỉ số Số ngày thu tiền khách hàng (DSRI)",
        "full_name": "Days Sales in Receivables Index",
        "desc": "Tỷ lệ số ngày thu tiền kỳ này so với kỳ trước. DSRI > 1 cho thấy số ngày thu tiền tăng, có thể do chính sách nới lỏng tín dụng hoặc nguy cơ ghi nhận doanh thu ảo.",
        "default": 1.25,
        "step": 0.05
    },
    "GMI": {
        "vn_name": "Chỉ số Tỷ suất Lợi nhuận gộp (GMI)",
        "full_name": "Gross Margin Index",
        "desc": "Tỷ suất lợi nhuận gộp kỳ trước so với kỳ này. GMI > 1 báo hiệu biên lợi nhuận gộp suy giảm, tạo động lực 'thổi phồng' lợi nhuận để duy trì hình ảnh tốt.",
        "default": 1.10,
        "step": 0.05
    },
    "AQI": {
        "vn_name": "Chỉ số Chất lượng Tài sản (AQI)",
        "full_name": "Asset Quality Index",
        "desc": "Tỷ lệ tài sản phi hiện vật kỳ này so với kỳ trước. AQI > 1 cho thấy doanh nghiệp tăng vốn hóa chi phí hoặc chuyển chi phí vào tài sản để làm đẹp sổ sách.",
        "default": 1.05,
        "step": 0.05
    },
    "SGI": {
        "vn_name": "Chỉ số Tăng trưởng Doanh thu (SGI)",
        "full_name": "Sales Growth Index",
        "desc": "Tăng trưởng doanh thu kỳ này so với kỳ trước. SGI cao thường gặp ở công ty tăng trưởng nóng, chịu áp lực phải duy trì kỳ vọng lợi nhuận.",
        "default": 1.30,
        "step": 0.05
    },
    "DEPI": {
        "vn_name": "Chỉ số Khấu hao Tài sản (DEPI)",
        "full_name": "Depreciation Index",
        "desc": "Tỷ lệ khấu hao kỳ trước so với kỳ này. DEPI > 1 cho thấy tốc độ trích khấu hao chậm lại hoặc kéo dài thời gian khấu hao nhằm tăng lợi nhuận kỳ báo cáo.",
        "default": 1.00,
        "step": 0.05
    },
    "SGAI": {
        "vn_name": "Chỉ số Chi phí Quản lý & Bán hàng (SGAI)",
        "full_name": "Sales, General and Administrative expenses Index",
        "desc": "Tỷ lệ chi phí bán hàng & QLDN trên doanh thu. SGAI > 1 phản ánh chi phí bán hàng & quản lý tăng nhanh hơn doanh thu, hiệu quả vận hành giảm.",
        "default": 1.15,
        "step": 0.05
    },
    "TATA": {
        "vn_name": "Tổng Biến dồn tích trên Tổng tài sản (TATA)",
        "full_name": "Total Accruals to Total Assets",
        "desc": "Đo lường mức độ dồn tích kế toán so với dòng tiền thực. TATA càng cao, lợi nhuận càng phụ thuộc vào ghi nhận kế toán thay vì dòng tiền thực thu.",
        "default": 0.08,
        "step": 0.01
    },
    "LVGI": {
        "vn_name": "Chỉ số Đòn bẩy Tài chính (LVGI)",
        "full_name": "Leverage Index",
        "desc": "Tỷ lệ tổng nợ trên tổng tài sản kỳ này so với kỳ trước. LVGI > 1 cho thấy đòn bẩy tài chính tăng, làm tăng rủi ro tài chính và vi phạm cam kết nợ.",
        "default": 1.12,
        "step": 0.05
    }
}

# ==============================================================================
# HÀM BỔ TRỢ & CACHING
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_default_data():
    """Tải dữ liệu mặc định từ MScore_data.csv nếu có"""
    possible_paths = [
        "MScore_data.csv",
        os.path.join(os.path.dirname(__file__), "MScore_data.csv")
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except Exception:
                pass
    return None

def clean_data(df):
    """Làm sạch và kiểm tra cấu trúc dữ liệu theo 8 biến Beneish"""
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        return None, f"Tập dữ liệu thiếu các cột bắt buộc: {', '.join(missing_cols)}"
    
    clean_df = df[FEATURES + [TARGET]].copy()
    for col in FEATURES + [TARGET]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
    
    initial_len = len(clean_df)
    clean_df = clean_df.dropna().reset_index(drop=True)
    clean_df[TARGET] = clean_df[TARGET].astype(int)
    
    valid_targets = set(clean_df[TARGET].unique()).issubset({0, 1})
    if not valid_targets:
        return None, "Cột 'FRAUD_FLAG' bắt buộc chỉ nhận hai giá trị nhị phân: 0 (Không gian lận) hoặc 1 (Gian lận)."
    
    dropped_count = initial_len - len(clean_df)
    msg = f"Đã làm sạch dữ liệu thành công: {len(clean_df)} quan sát hợp lệ"
    if dropped_count > 0:
        msg += f" (loại bỏ {dropped_count} dòng chứa dữ liệu trống)."
    return clean_df, msg

@st.cache_resource(show_spinner=False)
def build_and_train_pipeline(X_train_vals, y_train_vals, random_state=42):
    """Xây dựng và huấn luyện Pipeline: StandardScaler + LogisticRegression"""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=5000, random_state=random_state))
    ])
    pipe.fit(X_train_vals, y_train_vals)
    return pipe

def compute_beneish_traditional(row_dict):
    """
    Tính chỉ số Beneish M-Score chuẩn theo phương trình gốc của GS. Messod Beneish:
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.037*TATA + 0.0327*LVGI
    Ngưỡng phát hiện: M > -1.78 => Có rủi ro gian lận
    """
    score = (
        -4.84
        + 0.920 * row_dict.get("DSRI", 0)
        + 0.528 * row_dict.get("GMI", 0)
        + 0.404 * row_dict.get("AQI", 0)
        + 0.892 * row_dict.get("SGI", 0)
        + 0.115 * row_dict.get("DEPI", 0)
        - 0.172 * row_dict.get("SGAI", 0)
        + 4.037 * row_dict.get("TATA", 0)
        + 0.0327 * row_dict.get("LVGI", 0)
    )
    return score

def generate_excel_export(coef_df, cm_df, metrics_df, interp_df, model_info_df):
    """Xuất đầy đủ kết quả báo cáo ra tệp Excel (.xlsx) với các sheet chi tiết"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        coef_df.to_excel(writer, sheet_name="Coefficients", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        interp_df.to_excel(writer, sheet_name="Interpretation", index=False)
        model_info_df.to_excel(writer, sheet_name="Model_Info", index=False)
    buffer.seek(0)
    return buffer.getvalue()

# ==============================================================================
# THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=65)
    st.markdown("### ⚖️ Thiết lập Mô hình")
    
    data_option = st.radio(
        "Nguồn dữ liệu huấn luyện:",
        ["Dữ liệu mặc định (MScore_data.csv)", "Tải lên file CSV mới"],
        index=0
    )
    
    uploaded_file = None
    if data_option == "Tải lên file CSV mới":
        uploaded_file = st.file_uploader("Chọn file CSV:", type=["csv"])
    
    st.markdown("---")
    st.markdown("#### ⚙️ Tham số huấn luyện")
    test_size_pct = st.slider("Tỷ lệ phân chia tập Test (%):", min_value=10, max_value=40, value=20, step=5)
    test_size = test_size_pct / 100.0
    
    random_seed = st.number_input("Random State (Tái lập):", min_value=0, max_value=9999, value=42, step=1)
    
    decision_threshold = st.slider(
        "Ngưỡng quyết định (Decision Threshold):",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Xác suất dự báo P(FRAUD=1) lớn hơn hoặc bằng ngưỡng này sẽ được phân loại là Gian lận."
    )
    
    st.markdown("---")
    st.markdown("""
    **Về ứng dụng:**
    - Phát triển trên nền tảng **Streamlit**.
    - Sử dụng mô hình **Logistic Regression**.
    - Bộ 8 chỉ số tài chính **Beneish M-Score**.
    """)

# ==============================================================================
# NẠP DỮ LIỆU CHÍNH
# ==============================================================================
raw_df = None
if data_option == "Dữ liệu mặc định (MScore_data.csv)":
    raw_df = load_default_data()
    if raw_df is None:
        st.warning("⚠️ Không tìm thấy file `MScore_data.csv` mặc định trong thư mục. Vui lòng tải file lên ở thanh bên.")
else:
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Lỗi khi đọc file tải lên: {e}")

if raw_df is None:
    st.info("👋 Vui lòng kiểm tra file dữ liệu `MScore_data.csv` hoặc tải lên file dữ liệu ở thanh bên trái để bắt đầu.")
    st.stop()

clean_df, clean_msg = clean_data(raw_df)
if clean_df is None:
    st.error(f"❌ {clean_msg}")
    st.stop()

# ==============================================================================
# TIẾN HÀNH HUẤN LUYỆN MÔ HÌNH
# ==============================================================================
X = clean_df[FEATURES]
y = clean_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=int(random_seed),
    stratify=y
)

model = build_and_train_pipeline(X_train, y_train, random_state=int(random_seed))
logistic_step = model.named_steps["logistic"]
intercept = float(logistic_step.intercept_[0])
coefficients = logistic_step.coef_[0]

# Dự báo trên tập kiểm tra với ngưỡng tùy chỉnh
y_prob_test = model.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= decision_threshold).astype(int)

# Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

# Các chỉ tiêu đánh giá
acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

# Bảng trọng số & Odds ratio
coef_df = pd.DataFrame({
    "Chỉ số": FEATURES,
    "Tên đầy đủ": [FEATURE_METADATA[f]["vn_name"] for f in FEATURES],
    "Hệ số (Beta)": coefficients,
    "Odds Ratio (exp(Beta))": np.exp(coefficients),
    "Xu hướng tác động": ["Tăng nguy cơ gian lận" if c > 0 else "Giảm nguy cơ gian lận" for c in coefficients]
})

cm_df = pd.DataFrame(
    cm,
    index=["Thực tế: Bình thường (0)", "Thực tế: Gian lận (1)"],
    columns=["Dự báo: Bình thường (0)", "Dự báo: Gian lận (1)"]
)

metrics_df = pd.DataFrame({
    "Chỉ tiêu đánh giá": [
        "Độ chính xác (Accuracy)",
        "Độ chuẩn xác (Precision)",
        "Độ nhạy / Thu hồi (Recall / Sensitivity)",
        "F1-Score",
        "Độ đặc hiệu (Specificity)",
        "Tỷ lệ dương tính giả (FPR)",
        "Tỷ lệ âm tính giả (FNR)"
    ],
    "Giá trị thập phân": [acc, prec, rec, f1, spec, fpr, fnr],
    "Tỷ lệ phần trăm": [f"{v * 100:.2f}%" for v in [acc, prec, rec, f1, spec, fpr, fnr]]
})

interp_list = []
for name, b in zip(FEATURES, coefficients):
    or_val = np.exp(b)
    if b > 0:
        meaning = f"{name} tăng 1 độ lệch chuẩn làm tăng log-odds gian lận {b:.4f}; tỷ lệ odds gian lận được nhân {or_val:.4f} lần (giữ nguyên các biến khác)."
    elif b < 0:
        meaning = f"{name} tăng 1 độ lệch chuẩn làm giảm log-odds gian lận {abs(b):.4f}; tỷ lệ odds gian lận được nhân {or_val:.4f} lần (giữ nguyên các biến khác)."
    else:
        meaning = f"{name} không có ảnh hưởng tới mô hình."
    interp_list.append({"Chỉ số": name, "Hệ số": b, "Odds Ratio": or_val, "Ý nghĩa kinh tế": meaning})
interp_df = pd.DataFrame(interp_list)

model_info_df = pd.DataFrame({
    "Tham số": ["Intercept", "Decision Threshold", "Train Size", "Test Size", "Random State"],
    "Giá trị": [intercept, decision_threshold, len(X_train), len(X_test), random_seed]
})

# ==============================================================================
# GIAO DIỆN CHÍNH - TIÊU ĐỀ & TABS
# ==============================================================================
st.markdown('<div class="main-title">📊 Ứng Dụng Dự Báo Gian Lận Báo Cáo Tài Chính</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ứng dụng học máy Logistic Regression kết hợp bộ 8 chỉ số Beneish M-Score để phát hiện rủi ro thao túng số liệu kế toán</div>', unsafe_allow_html=True)

tab_eda, tab_model, tab_predict, tab_knowledge = st.tabs([
    "📈 1. Khám Phá Dữ Liệu",
    "🤖 2. Huấn Luyện & Đánh Giá Mô Hình",
    "🔍 3. Dự Báo Thực Tế (Đơn lẻ & Theo Lô)",
    "📖 4. Cẩm Nang 8 Chỉ Số Beneish"
])

# ==============================================================================
# TAB 1: KHÁM PHÁ DỮ LIỆU (EDA)
# ==============================================================================
with tab_eda:
    st.markdown("### 📋 Tổng Quan Dữ Liệu Phân Tích")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    total_samples = len(clean_df)
    fraud_count = int(clean_df[TARGET].sum())
    normal_count = total_samples - fraud_count
    fraud_rate = (fraud_count / total_samples) * 100
    
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tổng số quan sát</div>
            <div class="metric-value">{total_samples:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Doanh nghiệp Bình thường</div>
            <div class="metric-value" style="color: #16A34A;">{normal_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Doanh nghiệp Gian lận</div>
            <div class="metric-value" style="color: #DC2626;">{fraud_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tỷ lệ gian lận</div>
            <div class="metric-value" style="color: #D97706;">{fraud_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Hiển thị bảng dữ liệu xem trước
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.markdown("##### 🔍 Dữ liệu xem trước (Sample Data)")
        st.dataframe(clean_df.head(10), use_container_width=True)
    with col_t2:
        st.markdown("##### 📐 Thống kê mô tả (Descriptive Statistics)")
        st.dataframe(clean_df[FEATURES].describe().round(3).T[["mean", "std", "min", "50%", "max"]], use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Trực Quan Hóa Tương Quan & Phân Phối")

    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        st.markdown("##### Phân bố nhãn Doanh nghiệp (FRAUD_FLAG)")
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        sns.countplot(
            data=clean_df,
            x=TARGET,
            palette=["#10B981", "#EF4444"],
            ax=ax_bar
        )
        ax_bar.set_xticklabels(["0: Bình thường / An toàn", "1: Thao túng / Gian lận"])
        ax_bar.set_xlabel("Phân loại")
        ax_bar.set_ylabel("Số lượng doanh nghiệp")
        for p in ax_bar.patches:
            height = p.get_height()
            ax_bar.annotate(f'{int(height)} ({height/total_samples*100:.1f}%)',
                            (p.get_x() + p.get_width() / 2., height / 2),
                            ha='center', va='center', color='white', fontweight='bold', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_bar)

    with col_plot2:
        st.markdown("##### Ma trận hệ số tương quan (Correlation Heatmap)")
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4.4))
        corr_matrix = clean_df[FEATURES + [TARGET]].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax_corr, cbar=True, annot_kws={"size": 8})
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_corr)

    # So sánh giá trị trung bình 8 chỉ số giữa 2 nhóm
    st.markdown("##### 📈 So sánh trung bình 8 chỉ số Beneish giữa Nhóm Gian lận và Không gian lận")
    mean_comparison = clean_df.groupby(TARGET)[FEATURES].mean().T
    mean_comparison.columns = ["Bình thường (0)", "Gian lận (1)"]
    mean_comparison["Chênh lệch (1 - 0)"] = mean_comparison["Gian lận (1)"] - mean_comparison["Bình thường (0)"]

    fig_comp, ax_comp = plt.subplots(figsize=(11, 4))
    mean_comparison[["Bình thường (0)", "Gian lận (1)"]].plot(kind="bar", ax=ax_comp, color=["#10B981", "#EF4444"], width=0.7)
    ax_comp.set_title("Giá trị trung bình 8 chỉ số Beneish theo nhóm nhãn", fontsize=12, fontweight="bold")
    ax_comp.set_ylabel("Giá trị trung bình")
    ax_comp.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig_comp)

# ==============================================================================
# TAB 2: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH
# ==============================================================================
with tab_model:
    st.markdown("### 🎯 Kết Quả Huấn Luyện & Hiệu Năng Mô Hình")
    
    # KPI Metrics
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    mcol1.metric("Độ chính xác (Accuracy)", f"{acc*100:.2f}%")
    mcol2.metric("Độ chuẩn xác (Precision)", f"{prec*100:.2f}%")
    mcol3.metric("Độ nhạy (Recall)", f"{rec*100:.2f}%")
    mcol4.metric("F1-Score", f"{f1*100:.2f}%")
    mcol5.metric("Độ đặc hiệu (Specificity)", f"{spec*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_m_left, col_m_right = st.columns(2)

    with col_m_left:
        st.markdown("##### 🧩 Ma trận nhầm lẫn (Confusion Matrix)")
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Dự báo: 0 (An toàn)", "Dự báo: 1 (Gian lận)"],
            yticklabels=["Thực tế: 0 (An toàn)", "Thực tế: 1 (Gian lận)"],
            ax=ax_cm, annot_kws={"size": 14, "weight": "bold"}
        )
        ax_cm.set_ylabel("Nhãn thực tế", fontweight="bold")
        ax_cm.set_xlabel("Nhãn dự báo", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_cm)

        st.markdown(f"""
        - **Đúng bình thường (TN)**: `{tn}` | **Đúng gian lận (TP)**: `{tp}`
        - **Báo động giả (FP - Sai loại 1)**: `{fp}` (Tỷ lệ FPR: `{fpr*100:.2f}%`)
        - **Bỏ lọt gian lận (FN - Sai loại 2)**: `{fn}` (Tỷ lệ FNR: `{fnr*100:.2f}%`)
        """)

    with col_m_right:
        st.markdown("##### 📉 Đường cong ROC (Receiver Operating Characteristic)")
        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_prob_test)
        roc_auc_val = auc(fpr_curve, tpr_curve)

        fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4.2))
        ax_roc.plot(fpr_curve, tpr_curve, color="#2563EB", lw=2.5, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
        ax_roc.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Ngẫu nhiên (AUC = 0.50)")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("Tỷ lệ dương tính giả (FPR)", fontweight="bold")
        ax_roc.set_ylabel("Tỷ lệ dương tính thật (TPR)", fontweight="bold")
        ax_roc.set_title("Đường cong ROC & Điểm AUC", fontsize=11, fontweight="bold")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_roc)

    st.markdown("---")
    st.markdown("### 🧮 Phương Trình Hồi Quy & Tác Động Của Từng Chỉ Số")

    # Hiển thị phương trình Logit
    eq_str = f"\\text{{logit}}(P(\\text{{FRAUD}}=1)) = {intercept:.4f}"
    for name, b in zip(FEATURES, coefficients):
        eq_str += f" {'+' if b >= 0 else '-'} {abs(b):.4f} \\times \\text{{{name}}}_{{std}}"
    st.latex(eq_str)

    st.markdown("##### 📊 Bảng Hệ số Hồi quy (Beta) & Tỷ số Chênh (Odds Ratio)")
    st.dataframe(
        coef_df.style.format({
            "Hệ số (Beta)": "{:.4f}",
            "Odds Ratio (exp(Beta))": "{:.4f}"
        }).background_gradient(subset=["Hệ số (Beta)"], cmap="RdYlGn_r"),
        use_container_width=True
    )

    with st.expander("🔎 Xem Diễn giải Chi tiết Ý nghĩa Kinh tế của Từng Hệ số"):
        for _, row in interp_df.iterrows():
            st.markdown(f"- **{row['Chỉ số']}**: {row['Ý nghĩa kinh tế']}")

    # Xuất báo cáo Excel
    st.markdown("---")
    st.markdown("### 📥 Tải Báo Cáo Kết Quả Mô Hình (Excel)")
    excel_data = generate_excel_export(coef_df, cm_df, metrics_df, interp_df, model_info_df)
    st.download_button(
        label="📥 Tải xuống Báo cáo Kết quả Toàn diện (.xlsx)",
        data=excel_data,
        file_name="Logistic_Regression_MScore_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Xuất toàn bộ bảng hệ số, ma trận nhầm lẫn, chỉ số đánh giá và thông tin mô hình vào file Excel."
    )

# ==============================================================================
# TAB 3: DỰ BÁO GIAN LẬN THỰC TẾ (ĐƠN LẺ & THEO LÔ)
# ==============================================================================
with tab_predict:
    st.markdown("### 🔎 Dự Báo Nguy Cơ Gian Lận BCTC")
    subtab_single, subtab_batch = st.tabs(["👤 Dự báo cho 1 Doanh nghiệp", "📁 Dự báo theo Lô từ Tệp tải lên"])

    # --- DỰ BÁO ĐƠN LẺ ---
    with subtab_single:
        st.markdown("""
        Nhập 8 chỉ số Beneish tính toán từ Báo cáo tài chính 2 năm liền kề của doanh nghiệp để ước tính xác suất thao túng số liệu.
        """)

        # Điền nhanh các kịch bản mẫu
        sample_choice = st.selectbox(
            "⚡ Chọn kịch bản mẫu để kiểm tra nhanh:",
            ["Tùy chỉnh (Nhập thủ công)", "Doanh nghiệp có nguy cơ cao (Mẫu Gian lận điển hình)", "Doanh nghiệp lành mạnh (Mẫu An toàn)"]
        )

        sample_presets = {
            "Tùy chỉnh (Nhập thủ công)": {k: v["default"] for k, v in FEATURE_METADATA.items()},
            "Doanh nghiệp có nguy cơ cao (Mẫu Gian lận điển hình)": {
                "DSRI": 1.78, "GMI": 1.65, "AQI": 1.45, "SGI": 1.60,
                "DEPI": 1.25, "SGAI": 1.35, "TATA": 0.16, "LVGI": 1.38
            },
            "Doanh nghiệp lành mạnh (Mẫu An toàn)": {
                "DSRI": 0.85, "GMI": 0.95, "AQI": 0.78, "SGI": 0.98,
                "DEPI": 0.90, "SGAI": 0.92, "TATA": 0.02, "LVGI": 0.88
            }
        }
        current_preset = sample_presets[sample_choice]

        col_in1, col_in2 = st.columns(2)
        input_data = {}

        for idx, (feat, meta) in enumerate(FEATURE_METADATA.items()):
            target_col = col_in1 if idx < 4 else col_in2
            with target_col:
                val = st.number_input(
                    label=f"**{feat}** - {meta['vn_name']}",
                    min_value=-5.0,
                    max_value=15.0,
                    value=float(current_preset.get(feat, meta["default"])),
                    step=float(meta["step"]),
                    help=meta["desc"],
                    key=f"input_{feat}"
                )
                input_data[feat] = val

        if st.button("🚀 Thực hiện Dự báo Nguy cơ", type="primary", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            
            # Dự báo với mô hình hồi quy Logistic
            prob_fraud = float(model.predict_proba(input_df)[0, 1])
            is_fraud = prob_fraud >= decision_threshold

            # Tính chỉ số Beneish truyền thống đối chứng
            traditional_mscore = compute_beneish_traditional(input_data)
            trad_fraud = traditional_mscore > -1.78

            st.markdown("---")
            st.markdown("#### 🎯 Kết quả Đánh giá Rủi ro")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown("##### 1. Đánh giá theo Mô hình Học máy (Logistic Regression)")
                st.progress(prob_fraud)
                st.markdown(f"**Xác suất gian lận ước tính:** <span style='font-size: 1.5rem; font-weight: bold;'>{prob_fraud*100:.2f}%</span> (Ngưỡng phân loại: {decision_threshold*100:.0f}%)", unsafe_allow_html=True)
                
                if is_fraud:
                    st.error("🚨 **CẢNH BÁO NGUY CƠ CAO:** Doanh nghiệp có dấu hiệu thao túng, gian lận số liệu trên Báo cáo tài chính!")
                else:
                    st.success("✅ **AN TOÀN:** Xác suất gian lận nằm dưới ngưỡng rủi ro cho phép.")

            with res_col2:
                st.markdown("##### 2. Đối chứng Điểm Beneish M-Score Truyền thống")
                st.metric("Điểm Beneish M-Score", f"{traditional_mscore:.4f}", help="Ngưỡng quy ước truyền thống là -1.78. M > -1.78 là rủi ro cao.")
                if trad_fraud:
                    st.markdown(f"<span class='badge-fraud'>Nguy cơ cao (M-Score > -1.78)</span>", unsafe_allow_html=True)
                    st.caption("Theo lý thuyết Beneish chuẩn, công ty có xác suất cao đã thao túng lợi nhuận.")
                else:
                    st.markdown(f"<span class='badge-safe'>Vùng an toàn (M-Score ≤ -1.78)</span>", unsafe_allow_html=True)
                    st.caption("Theo lý thuyết Beneish chuẩn, công ty ít có nguy cơ gian lận.")

            # Phân tích mức độ đóng góp của từng chỉ số vào điểm Logit
            st.markdown("##### 🔍 Đóng góp tương đối của từng biến vào kết quả dự báo:")
            scaler = model.named_steps["scaler"]
            std_values = scaler.transform(input_df)[0]
            contributions = coefficients * std_values

            contrib_df = pd.DataFrame({
                "Chỉ số": FEATURES,
                "Giá trị nhập": [input_data[f] for f in FEATURES],
                "Đóng góp vào Logit": contributions
            }).sort_values(by="Đóng góp vào Logit", ascending=False)

            fig_contrib, ax_contrib = plt.subplots(figsize=(8, 3.5))
            colors = ["#EF4444" if x > 0 else "#10B981" for x in contrib_df["Đóng góp vào Logit"]]
            ax_contrib.barh(contrib_df["Chỉ số"], contrib_df["Đóng góp vào Logit"], color=colors)
            ax_contrib.axvline(0, color="gray", linestyle="--", alpha=0.7)
            ax_contrib.set_xlabel("Giá trị đóng góp (Beta * X_std)")
            ax_contrib.set_title("Chỉ số đẩy tăng nguy cơ (Đỏ) vs Chỉ số kéo giảm nguy cơ (Xanh)")
            ax_contrib.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig_contrib)

    # --- DỰ BÁO THEO LÔ ---
    with subtab_batch:
        st.markdown("##### 📂 Dự báo Hàng loạt Danh sách Doanh nghiệp")
        st.markdown("""
        Tải lên tệp **CSV** chứa các chỉ số tài chính của nhiều doanh nghiệp để hệ thống tự động quét và phân loại toàn bộ danh sách.
        Tệp cần chứa ít nhất 8 cột: `DSRI`, `GMI`, `AQI`, `SGI`, `DEPI`, `SGAI`, `TATA`, `LVGI`.
        """)

        # Nút tải file mẫu
        sample_batch_template = pd.DataFrame([
            {"Mã_CK": "AAA", "DSRI": 1.25, "GMI": 1.10, "AQI": 1.05, "SGI": 1.20, "DEPI": 1.02, "SGAI": 1.08, "TATA": 0.05, "LVGI": 1.12},
            {"Mã_CK": "BBB", "DSRI": 1.82, "GMI": 1.68, "AQI": 1.48, "SGI": 1.55, "DEPI": 1.30, "SGAI": 1.40, "TATA": 0.18, "LVGI": 1.42},
            {"Mã_CK": "CCC", "DSRI": 0.88, "GMI": 0.92, "AQI": 0.80, "SGI": 0.95, "DEPI": 0.91, "SGAI": 0.89, "TATA": -0.02, "LVGI": 0.92}
        ])
        csv_template = sample_batch_template.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Tải tệp CSV mẫu để điền dữ liệu",
            data=csv_template,
            file_name="Template_Danh_Sach_Doanh_Nghiep.csv",
            mime="text/csv"
        )

        batch_file = st.file_uploader("Chọn tệp danh sách cần dự báo (CSV):", type=["csv"], key="batch_uploader")
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                missing_in_batch = [f for f in FEATURES if f not in batch_df.columns]
                
                if missing_in_batch:
                    st.error(f"❌ Tệp tải lên thiếu các cột chỉ số bắt buộc: {', '.join(missing_in_batch)}")
                else:
                    st.success(f"Đã đọc thành công {len(batch_df)} doanh nghiệp từ tệp tải lên.")
                    
                    # Trích xuất 8 biến và chuyển dạng số
                    X_batch = batch_df[FEATURES].copy()
                    for f in FEATURES:
                        X_batch[f] = pd.to_numeric(X_batch[f], errors="coerce").fillna(0.0)
                    
                    # Dự báo
                    batch_probs = model.predict_proba(X_batch)[:, 1]
                    batch_preds = (batch_probs >= decision_threshold).astype(int)
                    
                    # Thêm kết quả vào DataFrame
                    result_df = batch_df.copy()
                    result_df["Xác suất Gian lận (%)"] = np.round(batch_probs * 100, 2)
                    result_df["Dự báo Mô hình"] = ["CẢNH BÁO GIAN LẬN" if p == 1 else "AN TOÀN" for p in batch_preds]
                    result_df["Điểm Beneish M-Score"] = [compute_beneish_traditional(row) for _, row in X_batch.iterrows()]
                    
                    st.markdown("##### 📊 Kết quả Dự báo Chi tiết:")
                    st.dataframe(result_df, use_container_width=True)

                    # Tóm tắt
                    num_batch_fraud = sum(batch_preds)
                    st.info(f"Phát hiện **{num_batch_fraud}/{len(result_df)}** ({num_batch_fraud/len(result_df)*100:.1f}%) doanh nghiệp có nguy cơ gian lận BCTC.")

                    # Tải kết quả về máy
                    res_csv = result_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 Tải tệp kết quả dự báo (.csv)",
                        data=res_csv,
                        file_name="Ket_Qua_Du_Bao_Gian_Lan_BCTC.csv",
                        mime="text/csv"
                    )
            except Exception as ex:
                st.error(f"Có lỗi xảy ra trong quá trình xử lý: {ex}")

# ==============================================================================
# TAB 4: CẨM NANG & KIẾN THỨC BENEISH M-SCORE
# ==============================================================================
with tab_knowledge:
    st.markdown("### 📚 Cẩm Nang Kiến Thức Về Mô Hình Beneish M-Score")
    st.markdown("""
    **Mô hình Beneish M-Score** được xây dựng bởi Giáo sư Messod Beneish (Đại học Indiana, Hoa Kỳ) vào năm 1999.
    Đây là một trong những công cụ định lượng kinh điển và hiệu quả nhất được các kiểm toán viên, chuyên viên phân tích tín dụng và nhà đầu tư toàn cầu sử dụng để phát hiện sớm các dấu hiệu thao túng lợi nhuận và gian lận báo cáo tài chính.
    """)

    st.markdown("#### 1. Chi tiết 8 Chỉ số Thành phần")
    for feat, meta in FEATURE_METADATA.items():
        with st.expander(f"📌 **{feat}** - {meta['vn_name']}"):
            st.markdown(f"- **Tên tiếng Anh:** `{meta['full_name']}`")
            st.markdown(f"- **Mô tả bản chất & dấu hiệu bất thường:** {meta['desc']}")

    st.markdown("---")
    st.markdown("#### 2. So sánh Mô hình Chuẩn Beneish (1999) và Mô hình Logistic Hiệu chỉnh")
    st.markdown("""
    | Đặc điểm | Mô hình Beneish (1999) Gốc | Mô hình Logistic Regression Trong Ứng Dụng Này |
    | :--- | :--- | :--- |
    | **Phương pháp** | Probit Regression cố định theo trọng số gốc 1999 | Logistic Regression có bước chuẩn hóa `StandardScaler` |
    | **Dữ liệu huấn luyện**| Dữ liệu doanh nghiệp Mỹ giai đoạn 1982 - 1992 | Huấn luyện trực tiếp trên tập dữ liệu thực tế tại Việt Nam/Thị trường mục tiêu |
    | **Ngưỡng quyết định** | Cố định tại **-1.78** (M-Score > -1.78 là rủi ro) | Linh hoạt điều chỉnh ngưỡng xác suất từ 10% đến 90% tùy khẩu vị rủi ro |
    | **Đầu ra dự báo** | Điểm số M-Score tổng hợp | Xác suất toán học chính xác $P \in [0\%, 100\%]$ kèm phân tích đóng góp từng biến |
    """)

    st.markdown("---")
    st.markdown("#### 3. Khuyến nghị Nghiệp vụ cho Kiểm toán viên & Thẩm định viên")
    st.info("""
    💡 **Lời khuyên ứng dụng trong thực tế:**
    1. Khi chỉ số **DSRI** hoặc **SGI** tăng đột biến trong khi dòng tiền từ hoạt động kinh doanh (CFO) âm, cần kiểm tra sâu các hợp đồng bán hàng vào thời điểm cuối niên độ tài chính.
    2. Chỉ số **TATA** phản ánh chênh lệch giữa lợi nhuận kế toán và dòng tiền thực; giá trị TATA dương lớn là một tín hiệu cảnh báo chất lượng lợi nhuận kém.
    3. Khi kết quả mô hình đưa ra cảnh báo **Nguy cơ cao**, doanh nghiệp cần được đưa vào danh sách kiểm toán/thẩm định đặc biệt trước khi cấp tín dụng hoặc ra quyết định đầu tư.
    """)

# ==============================================================================
# PHẦN CHÂN TRANG (FOOTER)
# ==============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>"
    "Hệ thống Dự báo Gian lận BCTC | Xây dựng trên nền tảng Streamlit & Scikit-Learn | Sẵn sàng triển khai trên Streamlit Community Cloud"
    "</div>",
    unsafe_allow_html=True
)
