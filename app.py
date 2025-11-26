# app.py
import streamlit as st
import numpy as np
import re
from sympy import symbols, sympify, factor, diff, integrate, lambdify
import plotly.graph_objs as go

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
func_input = st.text_input("함수 입력 (예: x^2 + 5*x + 6)", "x^2 + 2*x + 1")
func_processed = preprocess_func(func_input)

# SymPy 설정
x = symbols('x')
try:
    expr = sympify(func_processed)
    factored_expr = factor(expr)
    diff_expr = diff(expr, x)
    int_expr = integrate(expr, x)
except Exception as e:
    st.error(f"함수 처리 오류: {e}")
    st.stop()

st.write(f"입력 함수: {func_input}")
st.write(f"인수분해 결과: {factored_expr}")
st.write(f"미분: {diff_expr}")
st.write(f"적분: {int_expr} + C")

# x 범위
x_vals = np.linspace(-10, 10, 400)

# SymPy 함수 → NumPy 함수 변환
try:
    f_orig = lambdify(x, expr, "numpy")
    f_fact = lambdify(x, factored_expr, "numpy")
    f_diff = lambdify(x, diff_expr, "numpy")
    f_int = lambdify(x, int_expr, "numpy")

    y_orig = f_orig(x_vals)
    y_fact = f_fact(x_vals)
    y_diff = f_diff(x_vals)
    y_int = f_int(x_vals)
except Exception as e:
    st.error(f"그래프 계산 오류: {e}")
    st.stop()

# --- Plotly 그래프 ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_vals, y=y_orig, mode='lines', name=f'원 함수: {func_input}'))
fig.add_trace(go.Scatter(x=x_vals, y=y_fact, mode='lines', name=f'인수분해: {factored_expr}', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=x_vals, y=y_diff, mode='lines', name=f'미분: {diff_expr}', line=dict(dash='dot')))
fig.add_trace(go.Scatter(x=x_vals, y=y_int, mode='lines', name=f'적분: {int_expr} + C', line=dict(dash='dashdot')))

fig.update_layout(title='함수 그래프 및 인수분해/미분/적분',
                  xaxis_title='x', yaxis_title='y',
                  legend=dict(font=dict(size=10)))

st.plotly_chart(fig, use_container_width=True)
st.info("그래프 위 포인트 클릭 시 좌표 확인 가능")
