# -*- coding: utf-8 -*-
"""
Streamlit SVM 疾病预测网页

功能：
1. 加载 svm_model.pkl
2. 单条预测：分类变量默认空白，选择“是/否”；性别选择“男/女”
3. 数值变量默认空白，年龄限制为整数，其余实验室指标为浮点数
4. CSV 批量预测：支持 0/1，也支持 是/否、男/女
5. 下载 CSV 模板和预测结果
6. CSV 模板第一列为“患者姓名”，仅用于区分患者，不参与模型预测

运行命令：
streamlit run svm_streamlit.py
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. 页面和模型路径配置
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_model.pkl"

st.set_page_config(
    page_title="疾病预测模型[SVM]",
    page_icon="🩺",
    layout="wide",
)


# ============================================================
# 2. 加载模型
# ============================================================
@st.cache_resource
def load_model_package() -> dict[str, Any]:
    """
    加载训练好的 SVM 模型文件。

    要求 svm_model.pkl 是一个字典，至少包含：
    - model
    - scaler
    - feature_names

    你的 mysvm.py 中保存的 model_data 格式为：
    {
        "model": best_svm,
        "scaler": scaler,
        "feature_names": X.columns.tolist(),
        "best_params": grid_search.best_params_,
        "test_accuracy": accuracy,
        "test_auc": auc_score,
        "timestamp": ...
    }
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{MODEL_PATH}\n"
            "请确认 svm_model.pkl 和 svm_streamlit.py 在同一个文件夹。"
        )

    model_data = joblib.load(MODEL_PATH)

    if not isinstance(model_data, dict):
        raise TypeError(
            "svm_model.pkl 应该是一个 dict，且包含 model、scaler、feature_names。"
        )

    required_keys = ["model", "scaler", "feature_names"]
    missing_keys = [k for k in required_keys if k not in model_data]
    if missing_keys:
        raise KeyError(f"模型文件缺少字段：{missing_keys}")

    return model_data


def to_builtin(value: Any) -> Any:
    """
    把 numpy 类型转换成 Python 内置类型，方便 Streamlit 展示。
    """
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    return value


try:
    MODEL_DATA = load_model_package()
    MODEL = MODEL_DATA["model"]
    SCALER = MODEL_DATA["scaler"]
    FEATURE_NAMES = list(MODEL_DATA["feature_names"])

    MODEL_INFO_DISPLAY = {
        "best_params": to_builtin(MODEL_DATA.get("best_params", {})),
        "test_accuracy": to_builtin(MODEL_DATA.get("test_accuracy", None)),
        "test_auc": to_builtin(MODEL_DATA.get("test_auc", None)),
        "timestamp": to_builtin(MODEL_DATA.get("timestamp", None)),
    }

except Exception as e:
    st.error(str(e))
    st.stop()


# ============================================================
# 3. 特征类型定义
# ============================================================
# CSV 模板中的患者标识列：仅用于区分患者，不参与模型预测
PATIENT_NAME_COLUMN = "患者姓名"

# 0/1 二分类变量：0 = 否，1 = 是
YES_NO_FEATURES = [
    "糖尿病",
    "心力衰竭",
    "感染",
    "高血压",
    "心肌梗死",
    "高脂血症",
    "腹水",
    "高尿酸血症",
    "冠心病",
    "休克",
    "抗血小板药",
    "ACEI",
    "钙离子通道拮抗剂",
    "他汀类药物",
    "利尿剂",
    "β受体阻滞剂",
    "非甾体抗炎药",
    "维生素C",
    "β内酰胺类抗生素",
    "ARB类",
    "吸烟史",
    "是否手术",
]

# 性别：女 = 0，男 = 1
GENDER_FEATURE = "性别"

# 整数型数值变量
INTEGER_FEATURES = [
    "年龄",
]

# 只保留模型中真实存在的特征，避免特征名变化时报错
YES_NO_FEATURES = [f for f in YES_NO_FEATURES if f in FEATURE_NAMES]
INTEGER_FEATURES = [f for f in INTEGER_FEATURES if f in FEATURE_NAMES]
HAS_GENDER = GENDER_FEATURE in FEATURE_NAMES

BINARY_FEATURES = YES_NO_FEATURES.copy()
if HAS_GENDER:
    BINARY_FEATURES.append(GENDER_FEATURE)

# 除二分类变量和整数变量以外，其余全部按浮点数处理
FLOAT_FEATURES = [
    f for f in FEATURE_NAMES
    if f not in BINARY_FEATURES and f not in INTEGER_FEATURES
]


# ============================================================
# 4. 输入值转换和校验
# ============================================================
def parse_yes_no(value: Any, feature_name: str) -> int:
    """
    解析 0/1、是/否。
    0 = 否，1 = 是。
    """
    if pd.isna(value):
        raise ValueError(f"{feature_name} 不能为空")

    if isinstance(value, str):
        value = value.strip()

    mapping = {
        0: 0,
        1: 1,
        "0": 0,
        "1": 1,
        "否": 0,
        "是": 1,
        "无": 0,
        "有": 1,
        "False": 0,
        "True": 1,
        "false": 0,
        "true": 1,
    }

    if value in mapping:
        return mapping[value]

    raise ValueError(f"{feature_name} 只能填写 0/1 或 是/否，当前值为：{value}")


def parse_gender(value: Any) -> int:
    """
    解析性别。
    女 = 0，男 = 1。
    同时兼容 0/1。
    """
    if pd.isna(value):
        raise ValueError("性别不能为空")

    if isinstance(value, str):
        value = value.strip()

    mapping = {
        0: 0,
        1: 1,
        "0": 0,
        "1": 1,
        "女": 0,
        "男": 1,
        "女性": 0,
        "男性": 1,
        "F": 0,
        "M": 1,
        "f": 0,
        "m": 1,
    }

    if value in mapping:
        return mapping[value]

    raise ValueError(f"性别只能填写 女/男 或 0/1，当前值为：{value}")


def parse_integer(value: Any, feature_name: str) -> int:
    """
    解析整数型变量，例如年龄。
    """
    if value is None or pd.isna(value):
        raise ValueError(f"{feature_name} 不能为空")

    try:
        number = float(value)
    except Exception as exc:
        raise ValueError(f"{feature_name} 必须是整数，当前值为：{value}") from exc

    if not number.is_integer():
        raise ValueError(f"{feature_name} 必须是整数，当前值为：{value}")

    return int(number)


def parse_float(value: Any, feature_name: str) -> float:
    """
    解析浮点数变量。
    """
    if value is None or pd.isna(value):
        raise ValueError(f"{feature_name} 不能为空")

    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{feature_name} 必须是数字，当前值为：{value}") from exc


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    把用户输入转换成模型需要的 0/1、整数、浮点数。
    """
    missing = [f for f in FEATURE_NAMES if f not in record]
    if missing:
        raise ValueError(f"缺少特征：{missing}")

    normalized: dict[str, Any] = {}

    for feature in FEATURE_NAMES:
        value = record[feature]

        if feature in YES_NO_FEATURES:
            normalized[feature] = parse_yes_no(value, feature)
        elif feature == GENDER_FEATURE:
            normalized[feature] = parse_gender(value)
        elif feature in INTEGER_FEATURES:
            normalized[feature] = parse_integer(value, feature)
        else:
            normalized[feature] = parse_float(value, feature)

    return normalized


def make_dataframe_for_model(record: dict[str, Any]) -> pd.DataFrame:
    """
    将单条样本整理成模型需要的 DataFrame。
    注意：SVM 训练时是对全部特征 X 做 StandardScaler，
    因此预测时也要按完整 FEATURE_NAMES 顺序组装，然后整体 scaler.transform。
    """
    normalized = normalize_record(record)
    ordered = {f: normalized[f] for f in FEATURE_NAMES}
    return pd.DataFrame([ordered], columns=FEATURE_NAMES)


# ============================================================
# 5. 预测函数
# ============================================================
def predict_one(record: dict[str, Any]) -> dict[str, Any]:
    """
    单条预测。
    """
    df = make_dataframe_for_model(record)
    x_scaled = SCALER.transform(df)

    prediction = int(MODEL.predict(x_scaled)[0])

    result = {
        "prediction": prediction,
        "prediction_label": "是" if prediction == 1 else "否",
    }

    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(x_scaled)[0]
        result["negative_probability"] = float(proba[0])
        result["positive_probability"] = float(proba[1])
    else:
        result["negative_probability"] = None
        result["positive_probability"] = None

    return result


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    CSV 批量预测。
    输入：原始 CSV DataFrame。
    输出：原始数据 + 预测结果。

    注意：如果 CSV 中包含“患者姓名”列，该列只会保留在输出结果里，
    不会进入模型，不参与预测。
    """
    # 只检查模型训练时真正需要的特征列。
    # “患者姓名”等额外列允许存在，但不会参与模型预测。
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少特征列：{missing}")

    normalized_rows = []
    for idx, row in df.iterrows():
        try:
            raw_record = {f: row[f] for f in FEATURE_NAMES}
            normalized_rows.append(normalize_record(raw_record))
        except Exception as exc:
            raise ValueError(f"第 {idx + 2} 行数据错误：{exc}") from exc
            # idx + 2 是因为 CSV 第 1 行通常是表头

    x = pd.DataFrame(normalized_rows, columns=FEATURE_NAMES)
    x_scaled = SCALER.transform(x)

    prediction = MODEL.predict(x_scaled).astype(int)

    output = df.copy()
    output["prediction"] = prediction
    output["prediction_label"] = ["是" if p == 1 else "否" for p in prediction]

    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(x_scaled)
        output["negative_probability"] = proba[:, 0]
        output["positive_probability"] = proba[:, 1]

    return output


# ============================================================
# 6. CSV 工具函数
# ============================================================
def make_template_dataframe() -> pd.DataFrame:
    """
    生成空白 CSV 模板。
    第一列为“患者姓名”，仅用于区分患者，不参与模型预测。
    后面的列才是模型需要读取的特征列。
    """
    template_columns = [PATIENT_NAME_COLUMN] + FEATURE_NAMES
    return pd.DataFrame(columns=template_columns)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    DataFrame 转 CSV bytes。
    utf-8-sig 可以让 Excel 正确显示中文。
    """
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """
    读取上传 CSV，兼容 utf-8-sig / utf-8 / gbk。
    """
    content = uploaded_file.getvalue()
    encodings = ["utf-8-sig", "utf-8", "gbk"]

    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except Exception as e:
            last_error = e

    raise ValueError(f"CSV 读取失败，请尝试另存为 UTF-8 CSV。原始错误：{last_error}")


def make_feature_table() -> pd.DataFrame:
    """
    生成特征说明表。
    """
    rows = []
    for f in FEATURE_NAMES:
        if f in YES_NO_FEATURES:
            t = "二分类：否=0，是=1"
        elif f == GENDER_FEATURE:
            t = "性别：女=0，男=1"
        elif f in INTEGER_FEATURES:
            t = "整数"
        else:
            t = "浮点数"

        rows.append({"feature_name": f, "type": t})

    return pd.DataFrame(rows)


def make_linear_svm_importance() -> pd.DataFrame | None:
    """
    如果最佳 SVM 是线性核，可以展示 coef_ 的绝对值作为特征权重。
    如果是 RBF 等非线性核，则不展示该表。
    """
    if getattr(MODEL, "kernel", None) != "linear":
        return None

    if not hasattr(MODEL, "coef_"):
        return None

    coef = MODEL.coef_[0]
    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "coefficient": coef,
        "abs_coefficient": abs(coef),
    }).sort_values("abs_coefficient", ascending=False)

    return importance_df


# ============================================================
# 7. 页面主体
# ============================================================
st.title("疾病预测模型[SVM]")
st.caption("支持单条预测和 CSV 批量预测。模型文件：svm_model.pkl")

with st.sidebar:
    st.header("模型信息")
    st.write(f"特征数量：{len(FEATURE_NAMES)}")

    st.write("训练时保存的模型信息：")
    st.json(MODEL_INFO_DISPLAY)

    with st.expander("查看模型特征名"):
        st.write(FEATURE_NAMES)

    st.markdown("---")
    st.markdown("**变量编码说明**")
    st.write("疾病史 / 用药史 / 是否手术：否 = 0，是 = 1")
    st.write("性别：女 = 0，男 = 1")
    st.write("年龄：整数")
    st.write("实验室指标：浮点数")
    st.write("SVM：预测前对全部特征执行 StandardScaler 标准化")


tab_one, tab_batch, tab_feature, tab_help = st.tabs(
    ["单条预测", "CSV 批量预测", "特征信息", "使用说明"]
)


# ============================================================
# 8. 单条预测页面
# ============================================================
with tab_one:
    st.subheader("单条患者预测")
    st.write("所有输入框默认空白。请完整填写后再点击预测。")

    with st.form("single_prediction_form"):
        st.markdown("## 1）疾病史 / 用药史 / 分类变量")

        input_record: dict[str, Any] = {}

        binary_display_order = [f for f in FEATURE_NAMES if f in YES_NO_FEATURES]
        if HAS_GENDER:
            # 性别放在分类变量区域最后
            binary_display_order.append(GENDER_FEATURE)

        binary_cols = st.columns(3)
        for idx, feature in enumerate(binary_display_order):
            with binary_cols[idx % 3]:
                if feature == GENDER_FEATURE:
                    selected = st.selectbox(
                        label="性别",
                        options=["女", "男"],
                        index=None,
                        placeholder="请选择",
                        key=f"select_{feature}",
                    )
                    input_record[feature] = selected
                else:
                    selected = st.selectbox(
                        label=feature,
                        options=["否", "是"],
                        index=None,
                        placeholder="请选择",
                        key=f"select_{feature}",
                    )
                    input_record[feature] = selected

        st.markdown("## 2）实验室指标 / 年龄等数值变量")

        numeric_display_order = [
            f for f in FEATURE_NAMES
            if f in FLOAT_FEATURES or f in INTEGER_FEATURES
        ]

        numeric_cols = st.columns(3)
        for idx, feature in enumerate(numeric_display_order):
            with numeric_cols[idx % 3]:
                if feature in INTEGER_FEATURES:
                    value = st.number_input(
                        label=f"{feature}（整数）",
                        min_value=0,
                        max_value=130 if feature == "年龄" else None,
                        value=None,
                        step=1,
                        format="%d",
                        placeholder="请输入整数",
                        key=f"num_{feature}",
                    )
                    input_record[feature] = value
                else:
                    value = st.number_input(
                        label=f"{feature}（浮点数）",
                        value=None,
                        step=0.01,
                        format="%.4f",
                        placeholder="请输入数值",
                        key=f"num_{feature}",
                    )
                    input_record[feature] = value

        submitted = st.form_submit_button("开始预测", type="primary")

    # 按模型训练特征顺序整理，仅用于展示和预测
    ordered_input_record = {f: input_record.get(f) for f in FEATURE_NAMES}

    with st.expander("查看本次输入数据"):
        st.dataframe(pd.DataFrame([ordered_input_record]), use_container_width=True)

    if submitted:
        empty_features = [
            f for f, v in ordered_input_record.items()
            if v is None or (isinstance(v, str) and v.strip() == "")
        ]

        if empty_features:
            st.error(f"以下变量还没有填写，请补充完整后再预测：{empty_features}")
        else:
            try:
                result = predict_one(ordered_input_record)
                st.success("预测完成")

                metric_cols = st.columns(3)
                metric_cols[0].metric("预测类别", result["prediction_label"])

                if result["positive_probability"] is not None:
                    metric_cols[1].metric("是的概率", f"{result['positive_probability']:.2%}")
                    metric_cols[2].metric("否的概率", f"{result['negative_probability']:.2%}")
                else:
                    metric_cols[1].metric("是的概率", "无")
                    metric_cols[2].metric("否的概率", "无")

                st.json(result)

            except Exception as e:
                st.error(f"预测失败：{e}")


# ============================================================
# 9. CSV 批量预测页面
# ============================================================
with tab_batch:
    st.subheader("CSV 批量预测")

    template_df = make_template_dataframe()

    st.download_button(
        label="下载空白 CSV 模板",
        data=dataframe_to_csv_bytes(template_df),
        file_name="SVM_batch_prediction_template.csv",
        mime="text/csv",
    )

    st.info(
        "下载的 CSV 模板第一列为“患者姓名”，仅用于区分患者，不参与模型预测。"
        "CSV 文件必须包含模型训练时的全部特征列。"
        "二分类变量可填 0/1 或 是/否；性别可填 0/1 或 女/男；年龄必须是整数；实验室指标填写数字。"
    )

    uploaded_file = st.file_uploader("上传待预测 CSV 文件", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = read_uploaded_csv(uploaded_file)

            st.write("上传数据预览：")
            st.dataframe(input_df.head(20), use_container_width=True)

            missing_cols = [f for f in FEATURE_NAMES if f not in input_df.columns]

            if missing_cols:
                st.error(f"CSV 缺少以下列：{missing_cols}")
            else:
                if st.button("开始批量预测", type="primary"):
                    result_df = predict_dataframe(input_df)

                    st.success(f"批量预测完成，共预测 {len(result_df)} 条记录。")
                    st.dataframe(result_df, use_container_width=True)

                    st.download_button(
                        label="下载预测结果 CSV",
                        data=dataframe_to_csv_bytes(result_df),
                        file_name="SVM_prediction_results.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"处理失败：{e}")


# ============================================================
# 10. 特征信息页面
# ============================================================
with tab_feature:
    st.subheader("当前模型特征")
    st.dataframe(make_feature_table(), use_container_width=True)

    st.subheader("SVM 特征权重说明")

    importance_df = make_linear_svm_importance()
    if importance_df is not None:
        st.write("当前 SVM 是线性核，可以展示 coef_ 的绝对值作为特征权重：")
        st.dataframe(importance_df, use_container_width=True)
        st.bar_chart(importance_df.head(15).set_index("feature")["abs_coefficient"])
    else:
        st.info(
            "当前 SVM 不是线性核，或模型中没有 coef_。"
            "RBF 等非线性核不能直接用 coef_ 展示特征权重；如需特征重要性，"
            "建议在训练脚本中保存 permutation_importance 的结果。"
        )


# ============================================================
# 11. 使用说明页面
# ============================================================
with tab_help:
    st.subheader("运行方法")

    st.code("pip install -r requirements.txt", language="bash")
    st.code("streamlit run svm_streamlit.py", language="bash")

    st.subheader("项目文件结构")

    st.code(
        """
svm_streamlit_project/
├── svm_streamlit.py
├── svm_model.pkl
└── requirements.txt
        """.strip(),
        language="text",
    )

    st.subheader("输入编码说明")

    st.markdown(
        """
- CSV 模板第一列 **患者姓名**：仅用于区分患者，不参与模型预测
- 疾病史 / 用药史 / 是否手术：**否 = 0，是 = 1**
- 性别：**女 = 0，男 = 1**
- 年龄：**整数**
- 实验室指标：**浮点数**
- SVM 训练时对全部特征做了 `StandardScaler` 标准化，所以预测时也会按训练特征顺序整体标准化
        """.strip()
    )

    st.warning("注意：该模型输出结果只能作为辅助预测参考，不能直接替代临床诊断。")
