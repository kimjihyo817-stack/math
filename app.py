# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
from sympy import symbols, sympify, factor, diff, integrate
import warnings

# 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 함수 전처리 ---
def preprocess_func(func_str):
    func_str = func_str.replace('^', '**')
    func_str = re.sub(r'(\d)(x)', r'\1*\2', func_str)
    func_str = re.sub(r'(x)(\d)', r'\1*\2', func_str)
    return func_str

# --- Streamlit 앱 ---
st.title("함수 분석 그래프")
st.write("함수를 입력하면 원 함수, 인수분해, 미분, 적분 그래프를 보여줍니다.")

# 함수 입력
func_input = st.text_input("함수 입력 (예: x^2 + 5x + 6)", "x^2 + 2*x + 1")
func_processed = preprocess_func(func_input)

# SymPy 설정
x_sym = symbols('x')
try:
    expr = sympify(func_processed)
    factored_expr = factor(expr)
    diff_expr = diff(expr, x_sym)
    int_expr = integrate(expr, x_sym)
except Exception as e:
    st.error(f"함수 처리 오류: {e}")
    st.stop()

st.write(f"입력 함수: {func_input}")
st.write(f"인수분해 결과: {factored_expr}")
st.write(f"미분: {diff_expr}")
st.write(f"적분: {int_expr} + C")

# x 범위
x_vals = np.linspace(-10, 10, 400)

# 안전한 eval 환경
allowed_funcs = {
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs
}
safe_dict = allowed_funcs.copy()
safe_dict['x'] = x_vals

# y 값 계산
try:
    y_orig = eval(func_processed, {"__builtins__": None}, safe_dict)
    y_fact = np.array([float(factored_expr.subs(x_sym, val)) for val in x_vals])
    y_diff = np.array([float(diff_expr.subs(x_sym, val)) for val in x_vals])
    y_int = np.array([float(int_expr.subs(x_sym, val)) for val in x_vals])
except Exception as e:
    st.error(f"그래프 계산 오류: {e}")
    st.stop()

# --- 그래프 그리기 ---
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_vals, y_orig, label=f'원 함수: {func_input}', color='blue')
ax.plot(x_vals, y_fact, label=f'인수분해: {factored_expr}', color='green', linestyle='--')
ax.plot(x_vals, y_diff, label=f'미분: {diff_expr}', color='red', linestyle='-.')
ax.plot(x_vals, y_int, label=f'적분: {int_expr} + C', color='purple', linestyle=':')

ax.set_title('함수 그래프 및 인수분해/미분/적분', fontsize=14)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.grid(True)
ax.legend(fontsize=10)

st.pyplot(fig)

# Streamlit에는 클릭 좌표 기능이 따로 없어서 대체
st.info("※ 그래프 클릭 좌표 기능은 Streamlit에서는 지원되지 않습니다. 필요한 경우 Plotly 사용 가능")
