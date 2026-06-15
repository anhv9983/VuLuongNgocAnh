"""
app.py — Dashboard Streamlit AIDEOM-VN
========================================
Giao diện tương tác 5 tab hỗ trợ ra quyết định chính sách:

  Tab 1: Tổng quan — KPI quốc gia & tóm tắt pipeline
  Tab 2: Dự báo   — GDP 2026-2035 theo kịch bản (M1)
  Tab 3: Phân bổ  — Tối ưu ngân sách ngành/vùng (M2 + M3)
  Tab 4: Kịch bản — So sánh 5 chính sách (M1 + M3 + M4)
  Tab 5: Rủi ro   — Monte Carlo & Stress Test (M5)

Chạy:
    streamlit run app.py

Author: AIDEOM-VN Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import SCENARIOS, KPI_TARGETS_2030, REGION_NAMES_VI, ITEM_NAMES_VI
from src.m6_dashboard import DashboardBuilder, format_vnd, get_scenario_color

# ─── Cấu hình trang ──────────────────────────────────────────
st.set_page_config(
    page_title="AIDEOM-VN Dashboard",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Hàm chuyển đổi đơn vị ────────────────────────────────────
# Tỷ giá: 1 USD = 25,000 VND
# 1 nghìn tỷ VND = 40 triệu USD = 0.04 tỷ USD
# Vậy để đổi từ nghìn tỷ VND sang tỷ USD: chia cho 25
def to_usd(vnd_trillion):
    """Chuyển đổi từ nghìn tỷ VND sang tỷ USD"""
    return vnd_trillion / 25

# ─── Màu kịch bản ────────────────────────────────────────────
SCENARIO_COLORS = {
    "S1": "#6B7280", "S2": "#3B82F6",
    "S3": "#8B5CF6", "S4": "#10B981", "S5": "#F59E0B",
}

# ═══════════════════════════════════════════════════════════════
# Cache: Chạy pipeline một lần, cache kết quả
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Đang khởi tạo mô hình AIDEOM-VN...")
def load_pipeline():
    """Nạp và chạy toàn bộ pipeline, cache vào session."""
    from src.pipeline import AIDEOMPipeline
    pipe = AIDEOMPipeline(scenario_id="S5", n_mc_simulations=5_000, verbose=False)
    outputs = pipe.run_all(save_charts=False)
    # Attach M6 builder to outputs
    outputs._m6_builder = DashboardBuilder(outputs)
    return outputs


# ═══════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """Render sidebar với bộ điều khiển tham số."""
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Flag_of_Vietnam.svg",
                 width=60)
        st.title("AIDEOM-VN")
        st.caption("AI-Driven Economic Optimization Model for Vietnam")
        st.divider()

        st.subheader("Cài đặt Kịch bản")
        selected_scenario = st.selectbox(
            "Kịch bản chính sách",
            options=list(SCENARIOS.keys()),
            format_func=lambda s: f"{s}: {SCENARIOS[s].name_vi}",
            index=4,  # S5 mặc định
        )

        st.subheader("Tham số Mô hình")
        year_end = st.slider("Năm dự báo kết thúc", 2028, 2035, 2030)
        budget   = st.number_input(
            "Ngân sách kinh tế số (tỷ VND)",
            min_value=20_000, max_value=100_000,
            value=50_000, step=5_000,
            format="%d",
        )
        mc_runs = st.select_slider(
            "Mô phỏng Monte Carlo",
            options=[1_000, 3_000, 5_000, 10_000],
            value=5_000,
        )
        with_equity = st.toggle("Ràng buộc công bằng vùng miền", value=True)

        st.divider()
        st.caption("Nguồn dữ liệu: GSO/NSO, World Bank, Bộ KH&CN, Bộ TT&TT")
        st.caption("© 2025 AIDEOM-VN | MIT License")

    return {
        "scenario": selected_scenario,
        "year_end": year_end,
        "budget": budget,
        "mc_runs": mc_runs,
        "with_equity": with_equity,
    }


# ═══════════════════════════════════════════════════════════════
# Tab 1: Tổng quan
# ═══════════════════════════════════════════════════════════════

def tab_overview(outputs, params: dict) -> None:
    """Tab 1 — KPI quốc gia & tóm tắt hệ thống."""
    st.header("🇻🇳 Tổng quan AIDEOM-VN")
    st.caption("Mô hình ra Quyết định Phát triển Kinh tế Việt Nam trong Kỷ nguyên AI")

    # ── Metric cards ─────────────────────────────────────────
    m1 = outputs.m1
    m3 = outputs.m3
    m4 = outputs.m4
    m5 = outputs.m5

    sid = params["scenario"]
    forecast_r = m1["forecast_results"][sid]
    gdp_2030  = forecast_r.gdp[4] if len(forecast_r.gdp) > 4 else forecast_r.gdp[-1]
    gdp_2030_usd = to_usd(gdp_2030)
    growth_avg = m1["comparison_df"].loc[
        m1["comparison_df"]["scenario_id"] == sid, "avg_growth_pct"
    ].values[0]
    z_star = m3["results_by_scenario"][sid].objective_value
    netjob = m4["result_optimal"].total_net
    p50    = m5["result_s5"].percentiles[50]
    p50_usd = to_usd(p50)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("GDP 2025 (thực tế)",  "514 tỷ USD",  "+8.02% YoY")
    col2.metric(f"GDP 2030 [{sid}]",   f"{gdp_2030_usd:,.0f} tỷ USD",  f"{growth_avg:+.1f}%/năm")
    col3.metric("GDP Gain tối ưu (M3)", f"{z_star:,.0f}",   "tỷ VND")
    col4.metric("NetJob ròng (M4)",     f"{netjob:,.0f}k",  "nghìn việc")
    col5.metric("GDP P50 2030 (MC)",    f"{p50_usd:,.0f}",  "tỷ USD")

    st.divider()

    # ── Bản đồ kiến trúc module ───────────────────────────────
    col_arch, col_kpi = st.columns([1, 1])

    with col_arch:
        st.subheader("Kiến trúc 5 Module")
        modules_info = [
            ("M1", "Dự báo Kinh tế", "Cobb-Douglas mở rộng, TFP, Growth Accounting", "#2563EB"),
            ("M2", "Sẵn sàng AI/Số", "TOPSIS + Entropy Weight, 6 vùng & 10 ngành",  "#7C3AED"),
            ("M3", "Tối ưu Phân bổ", "LP 24 biến, PuLP/CVXPY, Equity Constraint",   "#059669"),
            ("M4", "Lao động AI",    "NetJob simulation, LP, Markov Chain",           "#D97706"),
            ("M5", "Phân tích Rủi ro","Monte Carlo 5K, Stress Test, VaR/CVaR",       "#DC2626"),
        ]
        for code, name, desc, color in modules_info:
            st.markdown(
                f'<div style="border-left:4px solid {color}; padding:8px 12px; '
                f'margin:4px 0; border-radius:4px; background:#f8fafc">'
                f'<b>{code} — {name}</b><br>'
                f'<small style="color:#6b7280">{desc}</small></div>',
                unsafe_allow_html=True,
            )

    with col_kpi:
        st.subheader("KPI Mục tiêu 2030 vs Hiện trạng")
        kpi_df = pd.DataFrame([
            {"KPI": "Tăng trưởng GDP (%/năm)", "Mục tiêu": "7.0%",
             "Hiện tại": f"{growth_avg:.1f}%", "Trạng thái": "✅" if growth_avg >= 7 else "⚠️"},
            {"KPI": "Kinh tế số / GDP", "Mục tiêu": "30%",
             "Hiện tại": f"{forecast_r.digital[4]:.1f}%",
             "Trạng thái": "✅" if forecast_r.digital[4] >= 30 else "⚠️"},
            {"KPI": "DN Công nghệ số (nghìn)", "Mục tiêu": "100",
             "Hiện tại": f"{forecast_r.ai_cap[4]:.0f}",
             "Trạng thái": "✅" if forecast_r.ai_cap[4] >= 100 else "⚠️"},
            {"KPI": "Lao động qua đào tạo", "Mục tiêu": "35%",
             "Hiện tại": f"{forecast_r.human_cap[4]:.1f}%",
             "Trạng thái": "✅" if forecast_r.human_cap[4] >= 35 else "⚠️"},
            {"KPI": "NetJob ròng (nghìn)", "Mục tiêu": "> 0",
             "Hiện tại": f"{netjob:.0f}k",
             "Trạng thái": "✅" if netjob > 0 else "❌"},
        ])
        st.dataframe(kpi_df, hide_index=True, use_container_width=True)

    # ── Thời gian chạy pipeline ───────────────────────────────
    st.divider()
    st.subheader("Hiệu suất Pipeline")
    rt = outputs.run_time
    rt_df = pd.DataFrame([
        {"Module": k, "Thời gian (s)": v, "Trạng thái": "✅ OK"}
        for k, v in rt.items()
    ])
    if outputs.errors:
        for k, e in outputs.errors.items():
            rt_df.loc[rt_df["Module"] == k, "Trạng thái"] = f"❌ {e[:40]}"
    st.dataframe(rt_df, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Tab 2: Dự báo GDP
# ═══════════════════════════════════════════════════════════════

def tab_forecast(outputs, params: dict) -> None:
    """Tab 2 — Dự báo GDP & phân rã tăng trưởng."""
    st.header("📈 Dự báo Kinh tế Việt Nam 2026–2035")

    m1 = outputs.m1
    sid = params["scenario"]

    # ── GDP Fan Chart (tất cả kịch bản) ──────────────────────
    st.subheader("GDP dự báo theo 5 Kịch bản Chính sách")

    fig = go.Figure()
    # Lịch sử
    from src.data_loader import load_macro
    df_hist = load_macro()
    fig.add_trace(go.Scatter(
        x=df_hist["year"], y=to_usd(df_hist["GDP_trillion_VND"]),
        mode="lines+markers", name="Lịch sử 2020–2025",
        line=dict(color="black", width=2.5), marker=dict(size=7),
    ))
    for s_id, res in m1["forecast_results"].items():
        c = SCENARIO_COLORS[s_id]
        name = f"{s_id}: {SCENARIOS[s_id].name_vi}"
        dash = "solid" if s_id == sid else "dot"
        width = 3 if s_id == sid else 1.5
        fig.add_trace(go.Scatter(
            x=res.years, y=to_usd(res.gdp),
            mode="lines", name=name,
            line=dict(color=c, width=width, dash=dash),
        ))
    fig.add_vline(x=2030, line_dash="dash", line_color="red",
                  annotation_text="Mục tiêu 2030")
    fig.update_layout(
        xaxis_title="Năm", yaxis_title="GDP (tỷ USD)",
        height=420, legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Phân rã tăng trưởng ───────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Phân rã Tăng trưởng 2020–2025")
        decomp = m1["decomp_df"]
        contrib_cols = ["contrib_K_pp","contrib_L_pp","contrib_D_pp",
                        "contrib_AI_pp","contrib_H_pp","contrib_TFP_pp"]
        labels = ["Vốn K","Lao động L","Số hóa D","AI","Nhân lực H","TFP"]
        colors_bar = ["#1D4ED8","#7C3AED","#059669","#D97706","#DC2626","#374151"]

        fig2 = go.Figure()
        for col, label, color in zip(contrib_cols, labels, colors_bar):
            fig2.add_trace(go.Bar(
                x=decomp["year"], y=decomp[col],
                name=label, marker_color=color,
            ))
        fig2.add_trace(go.Scatter(
            x=decomp["year"], y=decomp["gdp_growth_pct"],
            mode="lines+markers", name="ΔlnY",
            line=dict(color="black", width=2),
        ))
        fig2.update_layout(
            barmode="relative", height=350,
            xaxis_title="Năm", yaxis_title="Điểm phần trăm",
            legend=dict(orientation="h", y=-0.3, font=dict(size=9)),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader(f"Quỹ đạo Chỉ số số hóa & AI [{sid}]")
        res = m1["forecast_results"][sid]
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(
            x=res.years, y=res.digital,
            name="Kinh tế số / GDP (%)",
            line=dict(color="#3B82F6", width=2),
        ), secondary_y=False)
        fig3.add_trace(go.Bar(
            x=res.years, y=res.ai_cap,
            name="DN Công nghệ số (nghìn)",
            marker_color="#8B5CF6", opacity=0.6,
        ), secondary_y=True)
        fig3.add_hline(y=30, line_dash="dash", line_color="#3B82F6",
                       annotation_text="Mục tiêu D=30%")
        fig3.update_layout(height=350, hovermode="x unified",
                           legend=dict(y=-0.3, orientation="h"))
        fig3.update_yaxes(title_text="Kinh tế số / GDP (%)", secondary_y=False)
        fig3.update_yaxes(title_text="DN Công nghệ số (nghìn)", secondary_y=True)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Bảng so sánh kịch bản ────────────────────────────────
    st.subheader("Bảng so sánh 5 Kịch bản đến 2030")
    cmp_df = m1["comparison_df"].copy()
    cmp_df["gdp_2030_tỷ_USD"] = to_usd(cmp_df["gdp_2030_tn_vnd"])
    cmp_df = cmp_df.drop(columns=["gdp_2030_tn_vnd"])
    cmp_df.columns = [c.replace("_", " ").title() for c in cmp_df.columns]
    st.dataframe(
        cmp_df.style.background_gradient(
            subset=["Avg Growth Pct"], cmap="RdYlGn"
        ).format(precision=1),
        hide_index=True, use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════
# Tab 3: Phân bổ Ngân sách
# ═══════════════════════════════════════════════════════════════

def tab_allocation(outputs, params: dict) -> None:
    """Tab 3 — TOPSIS Readiness & LP Budget Allocation."""
    st.header("💰 Phân bổ Ngân sách Kinh tế Số")

    m2 = outputs.m2
    m3 = outputs.m3
    sid = params["scenario"]

    col_left, col_right = st.columns(2)

    # ── TOPSIS xếp hạng vùng ─────────────────────────────────
    with col_left:
        st.subheader("Xếp hạng Sẵn sàng AI — 6 Vùng (TOPSIS)")
        r_region = m2["region_result"]
        rank_df = r_region.ranking_expert.copy()

        fig = px.bar(
            rank_df, x="topsis_score", y="name",
            orientation="h", color="topsis_score",
            color_continuous_scale="RdYlGn",
            labels={"topsis_score": "Điểm C*", "name": ""},
        )
        fig.update_layout(height=320, showlegend=False,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # ── TOPSIS xếp hạng ngành ─────────────────────────────────
    with col_right:
        st.subheader("Xếp hạng Sẵn sàng AI — 10 Ngành (TOPSIS)")
        r_sector = m2["sector_result"]
        sec_df = r_sector.ranking_expert.copy()

        fig2 = px.bar(
            sec_df, x="topsis_score", y="name",
            orientation="h", color="topsis_score",
            color_continuous_scale="Blues",
            labels={"topsis_score": "Điểm C*", "name": ""},
        )
        fig2.update_layout(height=400, showlegend=False,
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Phân bổ tối ưu ───────────────────────────────────────
    st.subheader(f"Ma trận Phân bổ Tối ưu [{sid}]")
    result = m3["results_by_scenario"][sid]
    alloc_mat = result.allocation_matrix.copy()
    alloc_mat.index = [REGION_NAMES_VI.get(r, r) for r in alloc_mat.index]
    alloc_mat.columns = [ITEM_NAMES_VI.get(j, j) for j in alloc_mat.columns]
    alloc_mat["TỔNG"] = alloc_mat.sum(axis=1)

    fig_heat = px.imshow(
        alloc_mat.drop(columns="TỔNG"),
        text_auto=".0f",
        color_continuous_scale="YlOrRd",
        labels=dict(color="Tỷ VND"),
        title=f"Phân bổ ngân sách tỷ VND — Z* = {result.objective_value:,.0f} tỷ VND",
    )
    fig_heat.update_layout(height=320)
    st.plotly_chart(fig_heat, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Theo vùng (tỷ VND)**")
        st.dataframe(
            alloc_mat.style.background_gradient(cmap="Blues").format("{:,.0f}"),
            use_container_width=True,
        )
    with col_b:
        st.markdown("**Chi phí Công bằng Vùng miền**")
        eq_cost = m3["equity_cost"]
        r_wo = m3.get("result_no_equity")
        z_with = result.objective_value
        st.metric("Z* (có equity)", f"{z_with:,.0f} tỷ VND")
        st.metric("Chi phí equity", f"{eq_cost:,.0f} tỷ VND",
                  delta=f"-{eq_cost/max(z_with+eq_cost,1)*100:.1f}%",
                  delta_color="inverse")
        st.caption("Chi phí kinh tế (GDP gain bị hi sinh) để đảm bảo "
                   "phát triển đồng đều giữa các vùng.")

    # ── Phân tích độ nhạy ngân sách ──────────────────────────
    st.subheader("Phân tích Độ nhạy Ngân sách")
    sens_df = m3["sensitivity_df"]
    fig_sens = make_subplots(rows=1, cols=2,
                              subplot_titles=("Tổng GDP Gain Z*(B)",
                                              "Hiệu quả Biên Z*/B"))
    fig_sens.add_trace(go.Scatter(
        x=sens_df["budget_trillion"], y=sens_df["z_star"],
        mode="lines+markers", name="Z*",
        line=dict(color="#2563EB", width=2), marker=dict(size=8)
    ), row=1, col=1)
    fig_sens.add_trace(go.Scatter(
        x=sens_df["budget_trillion"], y=sens_df["z_per_budget"],
        mode="lines+markers", name="Z*/B",
        line=dict(color="#DC2626", width=2), marker=dict(size=8)
    ), row=1, col=2)
    fig_sens.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_sens, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Tab 4: So sánh Kịch bản
# ═══════════════════════════════════════════════════════════════

def tab_scenarios(outputs, params: dict) -> None:
    """Tab 4 — So sánh 5 kịch bản chính sách tích hợp."""
    st.header("🔄 So sánh 5 Kịch bản Chính sách")

    m1 = outputs.m1
    m3 = outputs.m3
    m4 = outputs.m4

    # ── Bảng tổng hợp KPI ────────────────────────────────────
    st.subheader("Bảng KPI Tổng hợp 2030")
    cmp_m1 = m1["comparison_df"]
    cmp_m3 = pd.DataFrame([
        {"scenario_id": s,
         "z_star_gdp_gain": m3["results_by_scenario"][s].objective_value}
        for s in ["S1","S2","S3","S4","S5"]
    ])
    kpi_table = cmp_m1.merge(cmp_m3, on="scenario_id")
    kpi_table["netjob_k"] = m4["result_optimal"].total_net
    kpi_table["gdp_2030_tỷ_USD"] = to_usd(kpi_table["gdp_2030_tn_vnd"])

    display_cols = ["scenario_id","scenario_name_vi","avg_growth_pct",
                    "gdp_2030_tỷ_USD","z_star_gdp_gain","gdp_gain_vs_S1_pct"]
    avail = [c for c in display_cols if c in kpi_table.columns]
    st.dataframe(
        kpi_table[avail].style.background_gradient(
            subset=["avg_growth_pct"] if "avg_growth_pct" in avail else [],
            cmap="RdYlGn"
        ).format(precision=1),
        hide_index=True, use_container_width=True,
    )

    # ── Radar chart ───────────────────────────────────────────
    st.subheader("Radar Chart — Phân tích Đánh đổi")
    metrics = ["GDP 2030", "Kinh tế số", "AI capacity", "Nhân lực", "Z* LP"]
    fig_radar = go.Figure()

    for sid in ["S1","S2","S3","S4","S5"]:
        row = m1["comparison_df"][m1["comparison_df"]["scenario_id"] == sid].iloc[0]
        gdp_2030_usd = to_usd(row["gdp_2030_tn_vnd"])

        dig_2030 = row["digital_2030_pct"]
        ai_2030  = row["ai_firms_2030_k"]
        hc_2030  = row["human_cap_2030_pct"]
        z_val    = m3["results_by_scenario"][sid].objective_value / 1000  # nghìn tỷ

        # Chuẩn hóa về [0,1]
        vals_raw = [gdp_2030_usd / 800, dig_2030 / 40,
                    ai_2030 / 120,        hc_2030 / 50,
                    z_val / 100]
        vals = [max(0, min(1, v)) for v in vals_raw]
        vals += [vals[0]]  # đóng vòng

        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=metrics + [metrics[0]],
            fill="toself", name=f"{sid}: {SCENARIOS[sid].name_vi}",
            line_color=SCENARIO_COLORS[sid], opacity=0.65,
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=450,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Timeline phân bổ vốn ─────────────────────────────────
    st.subheader("Phân bổ Vốn theo Kịch bản")
    alloc_data = []
    for sid, scen in SCENARIOS.items():
        for comp, w in scen.allocation.items():
            alloc_data.append({
                "Kịch bản": f"{sid}: {scen.name_vi}",
                "Hạng mục": {"K":"Vốn vật chất","D":"Số hóa",
                              "AI":"AI","H":"Nhân lực"}[comp],
                "Tỷ trọng (%)": w * 100,
            })
    alloc_df = pd.DataFrame(alloc_data)
    fig_alloc = px.bar(
        alloc_df, x="Kịch bản", y="Tỷ trọng (%)",
        color="Hạng mục", barmode="stack",
        color_discrete_map={
            "Vốn vật chất": "#1D4ED8",
            "Số hóa": "#3B82F6",
            "AI": "#8B5CF6",
            "Nhân lực": "#10B981",
        },
    )
    fig_alloc.update_layout(height=350,
                             xaxis_tickangle=-20,
                             legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_alloc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Tab 5: Rủi ro
# ═══════════════════════════════════════════════════════════════

def tab_risk(outputs, params: dict) -> None:
    """Tab 5 — Phân tích Rủi ro Monte Carlo & Stress Test."""
    st.header("⚠️ Phân tích Rủi ro & Bất định")

    m5 = outputs.m5
    sid = params["scenario"]
    risk_res = m5["risk_by_scenario"].get(sid, m5["result_s5"])
    gdp_sims = risk_res.gdp_simulations
    gdp_sims_usd = to_usd(gdp_sims)
    p = risk_res.percentiles
    p_usd = {k: to_usd(v) for k, v in p.items()}

    # ── Metric row ────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P50 GDP 2030", f"{p_usd[50]:,.0f}", "tỷ USD")
    c2.metric("VaR 95%",      f"{to_usd(risk_res.var_95):,.0f}",  "tỷ USD (worst 5%)")
    c3.metric("CVaR 95%",     f"{to_usd(risk_res.cvar_95):,.0f}", "tỷ USD (tail mean)")
    c4.metric("P(hụt mục tiêu)", f"{risk_res.prob_below_target*100:.1f}%",
              delta_color="inverse")

    st.divider()
    col_dist, col_sens = st.columns(2)

    # ── Phân phối GDP ─────────────────────────────────────────
    with col_dist:
        st.subheader("Phân phối GDP 2030")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=gdp_sims_usd, nbinsx=60,
            marker_color=SCENARIO_COLORS.get(sid, "#6B7280"),
            opacity=0.75, name="Mô phỏng",
        ))
        for pct_val, color, label in [
            (p_usd[5],  "red",   f"P5={p_usd[5]:,.0f} tỷ USD"),
            (p_usd[50], "navy",  f"P50={p_usd[50]:,.0f} tỷ USD"),
            (p_usd[95], "green", f"P95={p_usd[95]:,.0f} tỷ USD"),
        ]:
            fig_hist.add_vline(x=pct_val, line_color=color,
                               line_dash="dash",
                               annotation_text=label,
                               annotation_position="top")
        fig_hist.update_layout(
            height=320,
            xaxis_title="GDP 2030 (tỷ USD)",
            yaxis_title="Tần suất",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Sensitivity ───────────────────────────────────────────
    with col_sens:
        st.subheader("Đóng góp Phương sai — Nguồn Rủi ro")
        sens_df = risk_res.sensitivity
        fig_sens = px.bar(
            sens_df, x="pct_of_total", y="factor",
            orientation="h", color="pct_of_total",
            color_continuous_scale="Reds",
            labels={"pct_of_total": "% Phương sai", "factor": ""},
            text="pct_of_total",
        )
        fig_sens.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_sens.update_layout(height=320, showlegend=False,
                               yaxis=dict(autorange="reversed"),
                               xaxis_title="Đóng góp Phương sai (%)")
        st.plotly_chart(fig_sens, use_container_width=True)

    # ── Stress Test ───────────────────────────────────────────
    st.subheader("Stress Test — Kịch bản Cực đoan")
    stress = risk_res.stress_results
    if stress:
        from src.m5_risk import RiskAnalyzer
        stress_rows = [
            {"Kịch bản cực đoan": RiskAnalyzer.STRESS_SCENARIOS[k]["label"],
             "GDP trung vị 2030": f"{to_usd(v):,.0f} tỷ USD",
             "vs P50 cơ sở": f"{(to_usd(v)/p_usd[50]-1)*100:+.1f}%",
             "Đánh giá": "✅ An toàn" if v >= p[50]*0.92 else
                         "⚠️ Chú ý" if v >= p[50]*0.85 else "❌ Rủi ro cao"}
            for k, v in stress.items()
        ]
        stress_df = pd.DataFrame(stress_rows)
        st.dataframe(stress_df, hide_index=True, use_container_width=True)

    # ── So sánh rủi ro 5 kịch bản ────────────────────────────
    st.subheader("So sánh Rủi ro qua 5 Kịch bản")
    risk_cmp = m5["risk_comparison"]
    fig_cmp = make_subplots(rows=1, cols=2,
                             subplot_titles=("Khoảng tin cậy P5–P95 (tỷ USD)",
                                             "Xác suất hụt mục tiêu (%)"))
    for _, row in risk_cmp.iterrows():
        c = SCENARIO_COLORS.get(row["scenario_id"], "gray")
        xi = list(risk_cmp["scenario_name"]).index(row["scenario_name"])
        fig_cmp.add_trace(go.Scatter(
            x=[xi, xi], y=[to_usd(row["p5_gdp_tn"]), to_usd(row["p95_gdp_tn"])],
            mode="lines", line=dict(color=c, width=8), opacity=0.35,
            showlegend=False,
        ), row=1, col=1)
        fig_cmp.add_trace(go.Scatter(
            x=[xi],
