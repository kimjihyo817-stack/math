# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, factor, diff, integrate
import re
import matplotlib

# 한글 폰트 설정 (NanumGothic 사용, 없으면 기본폰트)
try:
    matplotlib.rc('font', family='NanumGothic')
except:
    matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 표시

# --- 함수 전처리 ---
def preprocess_func(func_str):
    func_str = func_str.replace('^', '**')
    func_str = re.sub(r'(\d)(x)', r'\1*\2', func_str)
    func_str = re.sub(r'(x)(\d)', r'\1*\2', func_str)
    return func_str

st.title("함수 분석 및 그래프 시각화 (한글 지원)")

# --- 사용자 입력 ---
func_input = st.text_input("함수 입력 (예: x^2 + 5x + 6)", "x^2 + 2*x + 1")
func_processed = preprocess_func(func_input)

# --- SymPy 설정 ---
x_sym = symbols('x')
try:
    expr = sympify(func_processed)
    factored_expr = factor(expr)
    diff_expr = diff(expr, x_sym)
    int_expr = integrate(expr, x_sym)
except:
    st.error("함수를 올바르게 입력해주세요.")
    st.stop()

st.subheader("함수 결과")
st.write(f"입력 함수: {func_input}")
st.write(f"인수분해 결과: {factored_expr}")
st.write(f"미분: {diff_expr}")
st.write(f"적분: {int_expr} + C")

# --- x 범위 설정 ---
x_vals = np.linspace(-10, 10, 400)

# --- y 값 계산 ---
y_orig = np.array([float(expr.subs(x_sym, val)) for val in x_vals])
y_fact = np.array([float(factored_expr.subs(x_sym, val)) for val in x_vals])
y_diff = np.array([float(diff_expr.subs(x_sym, val)) for val in x_vals])
y_int = np.array([float(int_expr.subs(x_sym, val)) for val in x_vals])

# --- 그래프 그리기 ---
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_vals, y_orig, label='원 함수', color='blue')
ax.plot(x_vals, y_fact, label='인수분해', color='green', linestyle='--')
ax.plot(x_vals, y_diff, label='미분', color='red', linestyle='-.')
ax.plot(x_vals, y_int, label='적분', color='purple', linestyle=':')

ax.set_title('함수 그래프 및 인수분해/미분/적분', fontsize=14)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.grid(True)
ax.legend(fontsize=10)

st.pyplot(fig)
plt.close(fig)  # 반복 렌더링 시 충돌 방지
