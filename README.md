# asm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# 1. TẠO DỮ LIỆU GIẢ LẬP (Để bạn có
data = {
    'gia_nha': [2500, 3100, -100, 4500, 2800, 3100, 5000, 2200, np.nan, 3500],
    'dien_tich': [50, 60, 70, 100, 55, 60, 120, 45, 80, 85],
    'so_phong': [2, 3, 0, 4, 2, 3, 5, 2, 3, 3],
    'vi_tri': ['Quận 1', 'Quận 3', 'Quận 1', 'Quận 7', 'Quận 3', 'Quận 3', 'Quận 1', 'Quận 7', 'Quận 1', 'Quận 3'],
    'mo_ta': ['nha dep lung linh', 'gan cho', 'gia re', 'biet thu sang trong', 'co cho dau xe', 'gan cho', 'view song', 'hem xe hoi', 'nha moi', 'luxury']
}
df = pd.DataFrame(data)

# --- GIAI ĐOẠN 1: LÀM SẠCH DỮ LIỆU ---

# 1.1 Xử lý dữ liệu vô lý (Giá âm, số phòng = 0)
df = df[(df['gia_nha'] > 0) & (df['so_phong'] > 0)]

# 1.2 Xử lý trùng lặp
df = df.drop_duplicates()

# 1.3 Tách Biến độc lập (X) và Biến mục tiêu (y)
X = df.drop('gia_nha', axis=1)
y = df['gia_nha']

# 1.4 Chia tập dữ liệu Train/Test (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- GIAI ĐOẠN 2: XÂY DỰNG PIPELINE TỰ ĐỘNG ---

# Định nghĩa các cột theo loại
numeric_features = ['dien_tich', 'so_phong']
categorical_features = ['vi_tri']

# Quy trình cho số: Điền thiếu bằng Median -> Chuẩn hóa bằng StandardScaler
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Quy trình cho chữ: Điền thiếu bằng 'missing' -> Chuyển thành số (One-Hot)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Gom tất cả vào bộ xử lý trung tâm
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Tạo Pipeline hoàn chỉnh: Tiền xử lý -> Mô hình (Random Forest)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# --- GIAI ĐOẠN 3: HUẤN LUYỆN VÀ ĐÁNH GIÁ ---

# Huấn luyện mô hình
model_pipeline.fit(X_train, y_train)

# Dự đoán trên tập Test
y_pred = model_pipeline.predict(X_test)

# In kết quả đánh giá
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# --- GIAI ĐOẠN 4: TRỰC QUAN HÓA (INSIGHT) ---

plt.figure(figsize=(10, 5))

# Biểu đồ phân phối giá nhà
plt.subplot(1, 2, 1)
sns.histplot(df['gia_nha'], kde=True, color='blue')
plt.title('Phân phối Giá Nhà')

# Biểu đồ so sánh Thực tế vs Dự đoán
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Giá thực tế')
plt.ylabel('Giá dự đoán')
plt.title('Thực tế vs Dự đoán')

plt.tight_layout()
plt.show()
