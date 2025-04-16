import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Thiết lập tiêu đề trang
st.set_page_config(page_title="Phân cụm khách hàng", layout="wide")

# Tạo khung cho tên trang web (vẫn giữ nguyên ở phần chính)
st.markdown(
    """
    <style>
    .header-container {
        position: relative;
        text-align: center;
        margin-bottom: 80px;
    }

    .header-title {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: rgba(255, 213, 128, 0.92);
        color: #1f1f1f;
        padding: 25px 100px;
        border-radius: 16px;
        font-size: 42px;
        font-weight: 700;
        white-space: nowrap;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.15);
    }

    .header-image {
        width: 50%;
        height: 100px;
        object-fit: cover;
        border-radius: 12px;
    }
    </style>

    <div class="header-container">
        <img src="image.png" class="header-image"/>
        <div class="header-title">CUSTOMER SEGMENTATION PROJECT</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Tạo menu bên trái (sidebar)
with st.sidebar:
    menu = ["🏠 Trang Chủ", "📊 Phương Pháp Phân Cụm", "🔍 Khám Phá Dữ Liệu", "📈 Kết Quả Dự Án","🧪 Trải Nghiệm sản phẩm"]
    choice = st.sidebar.selectbox('Menu', menu)
    st.sidebar.markdown("---")
    st.sidebar.image("pc_app.png", use_container_width=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("👩‍🏫 **Giảng viên:**")
    st.sidebar.info("Cô: Khuất Thùy Phương")
    st.sidebar.markdown("🎖️ **Thực hiện bởi:**")
    st.sidebar.info("Dương Đại Dũng")
    st.sidebar.info("Nguyễn Thị Cẩm Thu")
    st.sidebar.markdown("📅 **Ngày báo cáo:** 19/04/2025")

    # --- Hiển thị nội dung dựa trên lựa chọn menu ---
if choice == "🏠 Trang Chủ":
    st.header("Giới thiệu dự án")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image_home.png", caption="Minh họa dự án phân cụm khách hàng", use_container_width=True)

    st.write("""

    **📌 Tổng quan về dự án**

    Dự án này nhằm phân tích dữ liệu khách hàng và áp dụng kỹ thuật **phân cụm (clustering)** để tìm ra các nhóm khách hàng có hành vi tương đồng.

    **🎯 Mục tiêu**:
    - Hiểu rõ hơn về các nhóm khách hàng khác nhau.
    - Tối ưu hóa chiến lược marketing và chăm sóc khách hàng.
    - Trải nghiệm công nghệ phân tích dữ liệu thực tế.

    **🧩 Bài toán**:

    Dự án thực hiện phân cụm khách hàng cho một cửa hàng bán lẻ ở Mỹ.

    Ngôn ngữ và các thư viện sử dụng: **Python, Streamlit, scikit-learn, pandas, matplotlib**
    """)

elif choice == "📊 Phương Pháp Phân Cụm":
    st.header("Phương pháp phân cụm khách hàng")
    st.subheader("🔖Phân cụm dựa vào tập luật")
    st.write("""
    Dữ liệu được tiền xử lí như loại bỏ các giá trị null, các giá trị thiếu
    và tính toán các đại lượng **Requency (R)**, **Frequency (F)** và **Monetary (M)**.

    Sau đó áp dụng tập luật để phân cụm. Tập luật chia khách hàng thành 6 nhóm bao gồm
    1. Khách hàng **VIP: R+F+M ≥ 14**
    2. Khách hàng trung thành **Loyal Customers: IF F≥4 & M≥4 & R≥3**
    3. Khách hàng mang ít lợi nhuận **Light Customers: (R=2|3) & (F=2|3) & (M=2|3)**
    4. Khách hàng mới **New Customers: R=5 & F=1 & M=1**
    5. Khách hàng có nguy cơ rời đi **At-risk Customers: R ≤2**
    6. Khách hàng bình thường **Regulars: Còn lại**
    """)
    st.subheader("🧠Phân cụm dựa vào thuật toán KMeans")
    st.write("""
    Dữ liệu được tiền xử lý và chuẩn hóa trước khi áp dụng mô hình phân cụm.

    Các bước chính:
    1. Tiền xử lý dữ liệu: loại bỏ giá trị thiếu, chuẩn hóa dữ liệu.
    2. Chọn số lượng cụm tối ưu dựa trên phương pháp **Elbow**.
    3. Áp dụng mô hình **KMeans** từ thư viện `scikit-learn`.
    4. Trực quan hóa kết quả bằng biểu đồ 2D.

    Thư viện và thuật toán sử dụng: `KMeans`, `StandardScaler`, `matplotlib`, `seaborn`
    """)

elif choice == "🔍 Khám Phá Dữ Liệu":
    @st.cache_data
    def load_rfm_data():
        return pd.read_csv("rfm_df.csv")

    st.header("🔍 Khám phá dữ liệu khách hàng")

    st.markdown("""
    Trong bước này, chúng ta tiến hành **khám phá dữ liệu ban đầu** để hiểu rõ hơn về hành vi mua hàng của khách hàng:

    - **Tổng quan về dữ liệu của cửa hàng**
    - **Thống kê mô tả Recency, Frequency, Monetary**
    - **Phân phối các biến và mối tương quan**
    """)

    rfm = load_rfm_data()
    st.subheader("📌 Tổng quan về dữ liệu của cửa hàng")
    st.write("""

    - Cửa hàng có 3898 khách hàng với thời gian giao dịch từ đầu năm 2014 đến cuối năm 2015.
    - Cửa hàng có tổng cộng 167 sản phẩm, được chia làm 11 loại sản phẩm bao gồm Fresh Food, Dairy, Bakery & Sweets, Household & Hygiene, Beverages, Frozen & Processed Food, Pantry Staples, Specialty & Seasonal, Pet Care, Personal Care, Snacks.

    """)
    st.subheader("🔍Thống kê dữ liệu RFM")
    st.dataframe(rfm.head())

    st.subheader("📊 Thống kê mô tả")
    st.write(rfm.describe())

    st.subheader("📅 Tổng hợp một số thống kê theo năm")

    data_summary = {
        "Nội dung": [
            "Tháng có doanh thu cao nhất năm",
            "Tháng có doanh thu thấp nhất",
            "Sản phẩm bán nhiều nhất tháng 8",
            "Sản phẩm bán ít nhất tháng 11",
            "Sản phẩm có doanh thu cao nhất tháng 8",
            "Sản phẩm có doanh thu thấp nhất tháng 11"
        ],
        "Năm 2014": [
            "8 (13617$)",
            "11 (11793$)",
            "Whole milk",
            "Cleaner",
            "Beef",
            "Popcorn"
        ],
        "Năm 2015": [
            "8 (13617$)",
            "11 (12744$)",
            "Whole milk",
            "Instant food products",
            "Beef",
            "Instant food products"
        ]
    }

    df_summary = pd.DataFrame(data_summary)
    st.dataframe(df_summary)

    st.subheader("📊 Phân bố và Boxplot các giá trị R/F/M")
    st.image("rfm_distribution_boxplot.png", caption="Phân bố và Boxplot của Recency, Frequency, Monetary", use_container_width=True)

    st.subheader("🛒 Top 10 sản phẩm bán chạy và bán chậm")

    st.image("top_10_best_worst_products.png", caption="Top 10 sản phẩm bán chạy nhất và bán chậm nhất", use_container_width=True)

    st.subheader("🔍 Phân tích thành phần chuỗi thời gian theo tháng")
    st.image("decomposition_monthly_blue.png", caption="Decomposition theo tháng", use_container_width=True)
    st.subheader("🔍 Phân tích thành phần chuỗi thời gian theo tuần")
    st.image("decomposition_weekly_salmon.png", caption="Decomposition theo tuần ", use_container_width=True)
    st.subheader("🧠 Heatmap tương quan giữa các chỉ số RFM")

    st.image("rfm_correlation_heatmap_fixed.png", caption="Tương quan giữa Recency, Frequency và Monetary", use_container_width=True)


elif choice == "📈 Kết Quả Dự Án":
    st.header("Kết quả của dự án")
    st.subheader("🔖 Kết quả phân cụm bằng tập luật")
    st.write("""
    Biểu đồ phân cụm:""")
    st.image("Buble_Tapluat.png", caption="Biểu đồ phân cụm 2D", use_container_width =True)
    #st.image("RFM Segments.png", caption="Biểu đồ phân cụm 2D", use_container_width =True)
    st.subheader("🧠 Kết quả phân cụm bằng KMeans")
    st.write("""
    Sau khi phân cụm, dữ liệu khách hàng được chia thành các nhóm cụ thể như sau:

    - **Cluster 0**: khách hàng **VIP** với tần suất mua nhiều nhất và chi tiêu cao nhất.
    - **Cluster 1**: khách hàng **NON-ACTIVE** đã lâu không quay lại (~1.5 năm), có thể đã rời bỏ cửa hàng hoặc khách vãng lai.
    - **Cluster 2**: chiếm số đông nhất, thời điểm mua hàng gần nhất, tần suất mua hàng trung bình và chi tiêu trung bình nên thuộc nhóm khách hàng trung thành **LOYAL CUSTOMERS**.
    - **Cluster 3**: gần một năm chưa quay lại nên thuộc nhóm có nguy cơ rời đi **AT-RISK CUSTOMERS**.
    - **Cluster 4**: thời điểm mua hàng cũng khá lâu, tần suất mua hàng trung bình và chi tiêu trung bình nên thuộc nhóm mang lại lợi nhuận ít cho cửa hàng **LIGHT CUSTOMERS**.


    Biểu đồ phân cụm:
    """)
    # (Bạn có thể chèn hình ảnh hoặc biểu đồ ở đây nếu có)
    st.image("Buble_KMeans.png", caption="Biểu đồ phân cụm 2D", use_container_width =True)
    #st.image("Unsupervised Segments.png", caption="Biểu đồ phân cụm 2D", use_container_width =True)

elif choice == "🧪 Trải Nghiệm sản phẩm":
    # Load dữ liệu RFM từ file
    @st.cache_data
    def load_rfm_data():
        return pd.read_csv("rfm_df.csv")

    # Load mô hình KMeans đã huấn luyện
    @st.cache_data
    def load_kmeans_model():
        with open("customer_segmentation_kmeans_model.pkl", "rb") as f:
            return pickle.load(f)

    rfm_data = load_rfm_data()
    kmeans_model = load_kmeans_model()

    # Hàm phân nhóm theo RFM
    def rfm_level(df):
        if df['RFM_Score'] >= 14:
            return '👑 VIP'  # Thêm icon vương miện cho khách hàng VIP
        elif df['F'] >= 4 and df['M'] >= 4 and df['R'] >= 3:
            return '❤️ Loyal Customers' # Thêm icon trái tim cho khách hàng trung thành
        elif (df['R'] in [2, 3]) and (df['F'] in [2, 3]) and (df['M'] in [2, 3]):
            return '🛒 Light' # Thêm icon giỏ hàng cho khách hàng mua ít
        elif df['R'] <= 2:
            return '⚠️ At-Risk Customers' # Thêm icon cảnh báo cho khách hàng có nguy cơ rời đi
        elif df['M'] == 1 and df['F'] == 1 and df['R'] == 5:
            return '✨ New Customers' # Thêm icon ngôi sao cho khách hàng mới
        else:
            return '🚶‍♂️ Regulars' # Thêm icon người đi bộ cho khách hàng thông thường

    st.header("Trải nghiệm phân cụm")
    st.write("Chọn cách nhập dữ liệu để dự đoán nhóm phân cụm của khách hàng:")

    mode = st.radio("Chọn hình thức nhập", [
        "🔢 Nhập thủ công",
        "🆔 Chọn từ danh sách ID",
        "📥 Tải danh sách nhiều khách hàng"
    ])

    input_data = None
    recency = frequency = monetary = None

    if mode == "🔢 Nhập thủ công":
        sub_mode = st.radio("Cách nhập thủ công", ["Nhập Recency/Frequency/Monetary", "Nhập Customer ID bằng tay"])

        if sub_mode == "Nhập Recency/Frequency/Monetary":
            recency = st.slider("Recency (số ngày kể từ lần mua gần nhất)", 0, 700, 90)
            frequency = st.slider("Frequency (số lần mua hàng)", 1, 35, 10)
            monetary = st.slider("Monetary (tổng giá trị mua hàng)", 1, 351, 100)
            input_data = np.array([[recency, frequency, monetary]])

        elif sub_mode == "Nhập Customer ID bằng tay":
            customer_id_input = st.text_input("Nhập Customer ID **1000-5000**:")
            if customer_id_input.strip():
                try:
                    customer_id = int(customer_id_input)
                    row = rfm_data[rfm_data["CustomerID"] == customer_id]
                    if not row.empty:
                        recency = row["Recency"].values[0]
                        frequency = row["Frequency"].values[0]
                        monetary = row["Monetary"].values[0]
                        st.markdown(f"**Recency:** {recency}  \n**Frequency:** {frequency}  \n**Monetary:** {monetary}")
                        input_data = np.array([[recency, frequency, monetary]])
                    else:
                        st.warning("Không tìm thấy khách hàng trong dữ liệu.")
                except ValueError:
                    st.error("Customer ID phải là số nguyên.")

    elif mode == "🆔 Chọn từ danh sách ID":
        customer_ids = rfm_data['CustomerID'].unique()
        selected_id = st.selectbox("Chọn ID khách hàng", customer_ids)
        customer_row = rfm_data[rfm_data['CustomerID'] == selected_id]
        if not customer_row.empty:
            recency = customer_row['Recency'].values[0]
            frequency = customer_row['Frequency'].values[0]
            monetary = customer_row['Monetary'].values[0]
            st.markdown(f"**Recency:** {recency}  \n**Frequency:** {frequency}  \n**Monetary:** {monetary}")
            input_data = np.array([[recency, frequency, monetary]])
        else:
            st.warning("Không tìm thấy thông tin khách hàng.")
            input_data = None

    elif mode == "📥 Tải danh sách nhiều khách hàng":
        uploaded_file = st.file_uploader("Tải lên file CSV chứa cột: CustomerID, Recency, Frequency, Monetary", type="csv")

        if uploaded_file is not None:
            show_results = st.button("🚀 Submit để phân cụm")
            if show_results:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    required_cols = {'CustomerID', 'Recency', 'Frequency', 'Monetary'}
                    if not required_cols.issubset(df_upload.columns):
                        st.error("❌ File phải chứa đủ các cột: CustomerID, Recency, Frequency, Monetary")
                    else:
                        st.success("✅ Đã đọc dữ liệu thành công!")
                        st.dataframe(df_upload)

                        input_values = df_upload[["Recency", "Frequency", "Monetary"]].values
                        df_upload["KMeans_Cluster"] = kmeans_model.predict(input_values)

                        r_quartiles = pd.qcut(rfm_data['Recency'], 5, retbins=True)[1]
                        f_quartiles = pd.qcut(rfm_data['Frequency'], 5, retbins=True)[1]
                        m_quartiles = pd.qcut(rfm_data['Monetary'], 5, retbins=True)[1]

                        def get_rank(v, bins, reverse=False):
                            for i in range(1, len(bins)):
                                if v <= bins[i]:
                                    return 6 - i if reverse else i
                            return 1 if reverse else 5

                        df_upload["R"] = df_upload["Recency"].apply(lambda x: get_rank(x, r_quartiles, reverse=True))
                        df_upload["F"] = df_upload["Frequency"].apply(lambda x: get_rank(x, f_quartiles))
                        df_upload["M"] = df_upload["Monetary"].apply(lambda x: get_rank(x, m_quartiles))
                        df_upload["RFM_Score"] = df_upload["R"] + df_upload["F"] + df_upload["M"]
                        df_upload["RFM_Segment"] = df_upload.apply(rfm_level, axis=1)

                        st.markdown("### 🧮 Kết quả phân cụm cho danh sách khách hàng")
                        st.dataframe(df_upload[["CustomerID", "Recency", "Frequency", "Monetary", "KMeans_Cluster", "RFM_Segment"]])

                        csv_result = df_upload.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Tải kết quả phân cụm", csv_result, file_name="rfm_kmeans_result.csv", mime="text/csv")

                except Exception as e:
                    st.error("⚠️ Đã xảy ra lỗi khi xử lý file.")
                    st.exception(e)

    if mode != "📥 Tải danh sách nhiều khách hàng" and input_data is not None:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🤖 Phân cụm bằng KMeans"):
                try:
                    cluster_label = kmeans_model.predict(input_data)[0]

                    st.subheader("📍 Kết quả từ mô hình KMeans:")
                    rfm_info = f"(Recency: {recency}, Frequency: {frequency}, Monetary: {monetary})"
                    if cluster_label == 0:
                        st.success(f"KMeans: Khách hàng là **👑 VIP** {rfm_info}")
                    elif cluster_label == 1:
                        st.success(f"KMeans: Khách hàng là **😴 NON-ACTIVE CUSTOMERS** {rfm_info}")
                    elif cluster_label == 2:
                        st.success(f"KMeans: Khách hàng là **❤️ LOYAL CUSTOMERS** {rfm_info}")
                    elif cluster_label == 3:
                        st.success(f"KMeans: Khách hàng là **⚠️ AT-RISK CUSTOMERS** {rfm_info}")
                    elif cluster_label == 4:
                        st.success(f"KMeans: Khách hàng là **🛒 LIGHT CUSTOMERS** {rfm_info}")
                    else:
                        st.info(f"KMeans: Cụm số {cluster_label} {rfm_info} (chưa đặt nhãn).")

                except Exception as e:
                    st.error("Lỗi khi phân cụm bằng KMeans.")
                    st.exception(e)

        with col2:
            if st.button("📊 Phân cụm bằng RFM"):
                try:
                    r_quartiles = pd.qcut(rfm_data['Recency'], 5, retbins=True)[1]
                    f_quartiles = pd.qcut(rfm_data['Frequency'], 5, retbins=True)[1]
                    m_quartiles = pd.qcut(rfm_data['Monetary'], 5, retbins=True)[1]

                    def get_rfm_rank(value, bins, reverse=False):
                        for i in range(1, len(bins)):
                            if value <= bins[i]:
                                return 6 - i if reverse else i
                        return 1 if reverse else 5

                    r = get_rfm_rank(recency, r_quartiles, reverse=True)
                    f = get_rfm_rank(frequency, f_quartiles)
                    m = get_rfm_rank(monetary, m_quartiles)
                    rfm_score = r + f + m

                    rfm_input = {'R': r, 'F': f, 'M': m, 'RFM_Score': rfm_score}
                    segment = rfm_level(rfm_input)

                    st.subheader("📊 Kết quả theo phương pháp RFM phân vị:")
                    rfm_values_info = f"(Recency: {recency}, Frequency: {frequency}, Monetary: {monetary})"
                    st.success(f"RFM: Khách hàng thuộc nhóm **{segment}** {rfm_values_info} (R={r}, F={f}, M={m}, RFM_Score={rfm_score})")

                except Exception as e:
                    st.error("Lỗi khi phân cụm bằng RFM.")
                    st.exception(e)